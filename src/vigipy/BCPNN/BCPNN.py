import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.special import digamma, polygamma
from ..utils import Container

# The trigamma function is mathematically the first derivative of digamma (first-order polygamma).
def trigamma(x):
    return polygamma(1, x)

def bcpnn(
    container,
    min_events=4,
    MC=False,
    num_MC=10000,
):
    """
    A Bayesian Confidence Propogation Neural Network. 


    Arguments:
        container: A DataContainer object produced by the convert()
                    function from data_prep.py

        min_events: The min number of AE reports to be considered a signal

        MC (Bool): Use Monte Carlo simulations to make results more robust?

        num_mc (int): Number of MC simulations to run

    """
    input_params = locals()
    del input_params["container"]

    DATA = container.data
    N = container.N

    if min_events > 1:
        DATA = DATA.loc[DATA.events >= min_events]

    n11 = DATA["events"].to_numpy(dtype=np.float64)
    n1j = DATA["product_aes"].to_numpy(dtype=np.float64)
    ni1 = DATA["count_across_brands"].to_numpy(dtype=np.float64)

    n10 = n1j - n11
    n01 = ni1 - n11
    n00 = N - (n11 + n10 + n01)
    num_cell = len(n11)

    if not MC:
        p1 = 1 + n1j
        p2 = 1 + N - n1j
        q1 = 1 + ni1
        q2 = 1 + N - ni1
        r1 = 1 + n11
        r2b = N - n11 - 1 + (2 + N) ** 2 / (q1 * p1)
        # Calculate the Information Criterion
        digamma_term = (
            digamma(r1) - digamma(r1 + r2b) - (digamma(p1) - digamma(p1 + p2) + digamma(q1) - digamma(q1 + q2))
        )
        IC = np.asarray((np.log(2) ** -1) * digamma_term, dtype=np.float64)
        IC_variance = np.asarray(
            (np.log(2) ** -2)
            * (
                (trigamma(r1)
                - trigamma(r1 + r2b))
                + (trigamma(p1) - trigamma(p1 + p2)) + (trigamma(q1) - trigamma(q1 + q2))
            ),
            dtype=np.float64,
        )
        relative_risk=1
        posterior_prob = norm.cdf(np.log2(relative_risk), IC, np.sqrt(IC_variance))
        lower_bound = norm.ppf(0.025, IC, np.sqrt(IC_variance))
        upper_bound = norm.ppf(0.975, IC, np.sqrt(IC_variance))
    else:
        # Vectorized priors (direct one-step computation)
        q1j = (n1j + 0.5) / (N + 1)
        qi1 = (ni1 + 0.5) / (N + 1)
        qi0 = (N - ni1 + 0.5) / (N + 1)
        q0j = (N - n1j + 0.5) / (N + 1)

        a_ = 0.5 / (q1j * qi1)

        g11 = (q1j * qi1 * a_) + n11
        g10 = (q1j * qi0 * a_) + n10
        g01 = (q0j * qi1 * a_) + n01
        g00 = (q0j * qi0 * a_) + n00

        posterior_prob = np.empty(num_cell, dtype=np.float64)
        IC = np.empty(num_cell, dtype=np.float64)
        lower_bound = np.empty(num_cell, dtype=np.float64)
        upper_bound = np.empty(num_cell, dtype=np.float64)
        relative_risk=1

        # Adjustable block size. 2000 is ideal for RAM
        #---------------------------------------------
        chunk_size = 2000  
        
        for i in range(0, num_cell, chunk_size):
            # Selecting the current line block
            end = min(i + chunk_size, num_cell)
            n_chunk = end - i
            
            # Dirichlet simulation via Gamma laws ONLY for this block
            gamma11 = np.random.gamma(g11[i:end, np.newaxis], 1.0, size=(n_chunk, num_MC))
            gamma10 = np.random.gamma(g10[i:end, np.newaxis], 1.0, size=(n_chunk, num_MC))
            gamma01 = np.random.gamma(g01[i:end, np.newaxis], 1.0, size=(n_chunk, num_MC))
            gamma00 = np.random.gamma(g00[i:end, np.newaxis], 1.0, size=(n_chunk, num_MC))
            
            total_gamma = gamma11 + gamma10 + gamma01 + gamma00
            
            p11 = gamma11 / total_gamma
            p1_ = (gamma11 + gamma10) / total_gamma
            p_1 = (gamma11 + gamma01) / total_gamma
            
            # Immediate release of intermediate variables from the block
            del gamma11, gamma10, gamma01, gamma00, total_gamma
            
            ic_monte = np.log2(p11 / (p1_ * p_1))
            del p11, p1_, p_1
            
            # Calculation and storage of statistics for this block
            #  UMBA PARALLELIZED KERNEL CALL
            #--------------------------------------------------------
            posterior_prob, IC, lower_bound, upper_bound = _bcpnn_numba_kernel(
            g11, g10, g01, g00, num_MC
            )
            
            # Cleaning the main matrix of the iteration
            del ic_monte
    
    #-------------------------------
    # Compute FDR, FNR, Se and Sp
    #-------------------------------
    
    # 1. Compute Null (H0) and Alternative (H1) hypothesis probability
    #----------------------------------------------------------------
    p_h0 = np.asarray(posterior_prob)  # posterior_prob represents P(H0), the null hypothesis probability
    p_h1 = 1.0 - p_h0                  # Alternative hypothesis probability (true signal)
    
    # 2. Compute total expected masses in the baseline
    #-------------------------------------------------
    total_true_signals = np.sum(p_h1)
    total_true_negatives = np.sum(p_h0)
    num_cells = len(p_h0)
    
    # 3. Cumulative sums from left to right (from highest to lowest signal)
    #-----------------------------------------------------------------------
    true_positives_cum = np.cumsum(p_h1)
    false_positives_cum = np.cumsum(p_h0)
    post_range = np.arange(1, num_cells + 1)
    
    
    # FDR: proportion of false positives among the k raised alerts
    #--------------------------------------------------------------
    FDR = false_positives_cum / post_range
    
    # Sensitivity (Se): proportion of captured true signals out of the total available
    #----------------------------------------------------------------------------------
    Se = true_positives_cum / (total_true_signals + 1e-7)
    
    # FNR: proportion of missed true signals (those remaining to the right of the threshold)
    # Strictly equivalent to: FNR = 1.0 - Se
    #----------------------------------------------------------------------------------------
    missed_true_signals = total_true_signals - true_positives_cum
    FNR = missed_true_signals / (total_true_signals + 1e-7)

    # Specificity (Sp): proportion of correctly identified true negatives
    # Those are the true negatives that were NOT raised as alerts (remaining to the right)
    #---------------------------------------------------------------------------------------
    true_negatives_remaining = total_true_negatives - false_positives_cum
    Sp = true_negatives_remaining / (total_true_negatives + 1e-7)

    #---------------------------------------
    # Results
    #------------------------------
    name = DATA["product_name"]
    ae = DATA["ae_name"]
    count = n11
    RC = Container(params=True)
    RC.param["input_params"] = input_params

    # SIGNALS RESULTS and presentation
    #----------------------------------
    RC.all_signals = pd.DataFrame(
        {
            "Product": name,
            "Adverse Event": ae,
            "2**IC": 2**(IC),
            "lower_bound" : 2**(lower_bound),
            "upper_bound" : 2**(upper_bound),
            "p_{H0}" : p_h0,
            "N_{11}": n11,
            "N_{10}": n10,
            "N_{01}": n01,
            "N_{00}": n00,
            "FDR": FDR,
            "FNR": FNR,
            "Se": Se,
            "Sp": Sp,
        }
    ).sort_values(by=["lower_bound"], ascending=False)
    RC.signals = RC.all_signals.loc[RC.all_signals["lower_bound"] > 1]

    # Calculate total number of detected signals: criterion is lower_bound > 0
    # This is exactly stating that IC025 > 0
    #--------------------------------------------------------------------------
    num_signals = (lower_bound > 0).sum()

    # Number of signals
    RC.num_signals = num_signals
    return RC


#-----------------------------------
# parallelized for MC loop
#--------------------------------------
import numpy as np
from numba import njit, prange

@njit(parallel=True, fastmath=True)
def _bcpnn_numba_kernel(g11, g10, g01, g00, num_MC):
    num_cell = len(g11)
    
    # Pre-allocation of final results (one element per cell)
    posterior_prob = np.empty(num_cell, dtype=np.float64)
    IC = np.empty(num_cell, dtype=np.float64)
    lower_bound = np.empty(num_cell, dtype=np.float64)
    upper_bound = np.empty(num_cell, dtype=np.float64)
    
    # 'prange' tells Numba to parallelize this loop across all CPU cores
    for m in prange(num_cell):
        # Local array for each thread to store the draws of the current cell
        ic_monte = np.empty(num_MC, dtype=np.float64)
        
        # Monte Carlo simulation for cell 'm'
        for i in range(num_MC):
            # Tirage de lois Gamma thread-safe
            gamma11 = np.random.gamma(g11[m], 1.0)
            gamma10 = np.random.gamma(g10[m], 1.0)
            gamma01 = np.random.gamma(g01[m], 1.0)
            gamma00 = np.random.gamma(g00[m], 1.0)
            
            total = gamma11 + gamma10 + gamma01 + gamma00
            
            p11 = gamma11 / total
            p1_ = (gamma11 + gamma10) / total
            p_1 = (gamma11 + gamma01) / total
            
            # Avoid division by zero if the probabilities are zero
            if p1_ * p_1 > 0 and p11 > 0:
                ic_monte[i] = np.log2(p11 / (p1_ * p_1))
            else:
                ic_monte[i] = -np.inf # Numerical equivalent of an impossible value
        
        # Local rapid sorting (highly optimized by Numba)
        ic_monte.sort()
        
        # Calculation of the posterior probability (ic_monte < 0.0)
        under_zero = 0
        for i in range(num_MC):
            if ic_monte[i] < 0.0:
                under_zero += 1
                
        posterior_prob[m] = under_zero / num_MC
        
        # Extracting percentiles directly from the sorted table
        IC[m] = ic_monte[int(round(num_MC * 0.50))]
        lower_bound[m] = ic_monte[int(round(num_MC * 0.025))]
        upper_bound[m] = ic_monte[int(round(num_MC * 0.975))]
        
    return posterior_prob, IC, lower_bound, upper_bound

