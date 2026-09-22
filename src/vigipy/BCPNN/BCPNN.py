import numpy as np
import pandas as pd
from scipy.stats import norm
from sympy.functions.special import gamma_functions
from ..utils import Container

digamma = np.vectorize(gamma_functions.digamma)
trigamma = np.vectorize(gamma_functions.trigamma)


def bcpnn(
    container,
    min_events=1,
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
        num_MC = float(num_MC)
        # Priors for the contingency table
        q1j = (n1j + 0.5) / (N + 1)
        qi1 = (ni1 + 0.5) / (N + 1)
        qi0 = (N - ni1 + 0.5) / (N + 1)
        q0j = (N - n1j + 0.5) / (N + 1)

        a_ = 0.5 / (q1j * qi1)

        a11 = q1j * qi1 * a_
        a10 = q1j * qi0 * a_
        a01 = q0j * qi1 * a_
        a00 = q0j * qi0 * a_

        g11 = a11 + n11
        g10 = a10 + n10
        g01 = a01 + n01
        g00 = a00 + n00

        posterior_prob = []
        lower_bound = []
        upper_bound = []
        IC = []
        relative_risk=1
        for m in range(num_cell):
            alpha = [g11[m], g10[m], g01[m], g00[m]]
            p = np.random.dirichlet(alpha, int(num_MC))
            p11 = p[:, 0]
            p1_ = p11 + p[:, 1]
            p_1 = p11 + p[:, 2]
            ic_monte = np.log2(p11 / (p1_ * p_1))
            ic_monte = np.sort(ic_monte)
            temp = 1 * (ic_monte < np.log2(relative_risk))
            posterior_prob.append(sum(temp) / num_MC)
            IC.append(ic_monte[round(num_MC * 0.50)])    
            lower_bound.append(ic_monte[round(num_MC * 0.025)]) 
            upper_bound.append(ic_monte[round(num_MC * 0.975)])
        posterior_prob = np.asarray(posterior_prob)
        IC = np.asarray(IC)
        lower_bound = np.asarray(lower_bound)
        upper_bound = np.asarray(upper_bound)


    FDR = np.cumsum(posterior_prob) / np.arange(1, len(posterior_prob) + 1)
    FNR = (np.cumsum(1 - posterior_prob)[::-1]) / (num_cell - np.arange(1, len(posterior_prob) + 1) + 1e-7)
    Se = np.cumsum((1 - posterior_prob)) / (sum(1 - posterior_prob))
    Sp = (np.cumsum(posterior_prob)[::-1]) / (num_cell - sum(1 - posterior_prob))

    name = DATA["product_name"]
    ae = DATA["ae_name"]
    count = n11
    RC = Container(params=True)

    RC.param["input_params"] = input_params

    # SIGNALS RESULTS and presentation
    RC.all_signals = pd.DataFrame(
        {
            "Product": name,
            "Adverse Event": ae,
            "count/expected": 2**(IC),
            "lower_bound" : 2**(lower_bound),
            "upper_bound" : 2**(upper_bound),
            "Count": count,
            "product margin": n1j,
            "event margin": ni1,
            "fdr": FDR,
            "FNR": FNR,
            "Se": Se,
            "Sp": Sp,
        }
    ).sort_values(by=["lower_bound"], ascending=False)
    RC.signals = RC.all_signals.loc[RC.all_signals["lower_bound"] > 0]

    
    if num_signals > 0:
        num_signals -= 1
    else:
        num_signals = 0

    # Number of signals
    RC.num_signals = num_signals
    return RC
