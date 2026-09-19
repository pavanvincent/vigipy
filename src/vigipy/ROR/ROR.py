import warnings
import numpy as np
import pandas as pd
from scipy.stats import norm
from ..utils.lbe import lbe
from ..utils import Container
from ..utils import calculate_expected


def ror(
    container,
    relative_risk=1,
    min_events=1,
    # Vincent PAVAN
    # delete decision_metric and decision_thres
    # decision_metric="fdr",
    # decision_thres=0.05,
    # Vincent PAVAN
    # sort output results using Lower Bounb of 95% confidence interval
    ranking_statistic="LB",
    # do not calculate Expected in the ROR 
    # expected_method="mantel-haentzel",
    # method_alpha=1,
):
    """
    Calculate the proportional reporting ratio.

    Arguments:
        container: A DataContainer object produced by the convert()
                    function from data_prep.py

        relative_risk (int/float): The relative risk value

        min_events: The min number of AE reports to be considered a signal

        decision_metric (str): The metric used for detecting signals:
                            {fdr = false detection rate,
                            signals = number of signals,
                            rank = ranking statistic}

        decision_thres (float): The min thres value for the decision_metric

        ranking_statistic (str): How to rank signals:
                            {'p_value' = posterior prob of the null hypothesis,
                            'CI' = 95% CI lower boundary}

        expected_method: The method of calculating the expected counts for
                        the disproportionality analysis.

        method_alpha: If the expected_method is negative-binomial, this
                    parameter is the alpha parameter of the distribution.

    """
    DATA = container.data
    N = container.N

    if min_events > 1:
        DATA = DATA[DATA.events >= min_events]

    n11 = np.asarray(DATA["events"], dtype=np.float64)
    n1j = np.asarray(DATA["product_aes"], dtype=np.float64)
    ni1 = np.asarray(DATA["count_across_brands"], dtype=np.float64)
    num_cell = len(n11)
    # Vincent PAVAN, 18/09/2026
    # no need to compute expected anymore
    # expected = calculate_expected(N, n1j, ni1, n11, expected_method, method_alpha)

    n10 = n1j - n11
    n01 = ni1 - n11 + 1e-7
    n00 = N - (n11 + n10 + n01)

    log_ror = np.log(n11 * n00 / (n10 * n01))
    var_log_ror = 1.0 / n11 + 1.0 / n10 + 1.0 / n01 + 1.0 / n00
    pval_uni = 1 - norm.cdf(log_ror, np.log(relative_risk), np.sqrt(var_log_ror))
    # rankstat = (log_prr - np.log(relative_risk)) / np.sqrt(var_log_prr)
    pval_uni[pval_uni > 1] = 1
    pval_uni[pval_uni < 0] = 0
    # 
    # with warnings.catch_warnings():
    #    warnings.simplefilter("ignore")
    #    results = lbe(2 * np.minimum(pval_uni, 1 - pval_uni))
    # pi_c = results[1]

    # fdr = pi_c * np.sort(pval_uni[pval_uni <= 0.5]) / (np.arange(1, (pval_uni <= 0.5).sum() + 1) / num_cell)

    # fdr = np.concatenate(
    #    (
    #        fdr,
    #        (
    #            pi_c / (2 * np.arange(((pval_uni <= 0.5).sum()), num_cell) / num_cell)
    #            + 1
    #            - (pval_uni <= 0.5).sum() / np.arange((pval_uni <= 0.5).sum(), num_cell)
    #        ),
    #    ),
    #    axis=None,
    #)

    # Vincent PAVAN, 18/09/2026
    # non necessary computation
    # FDR = np.minimum(fdr, np.ones((len(fdr),)))
    # if ranking_statistic == "CI":
    #    FDR = np.empty((len(n11),))
    LB = norm.ppf(0.025, log_ror, np.sqrt(var_log_ror))
    # Vincent PAVAN, 17/09/2026
    # Add the computation of Upper Bound for log(Confidence Intervalle)
    UB = norm.ppf(0.975,log_ror, np.sqrt(var_log_ror))
    # if ranking_statistic == "p_value":
    #    RankStat = pval_uni
    # else:
    # Vincent PAVAN, 198/09/2026
    # decision is positive when LB > 0
    # RankStat = LB
    num_signals = (LB > 0).sum()
     # if decision_metric == "fdr":
     #   num_signals = (FDR <= decision_thres).sum()
     # elif decision_metric == "signals":
     #   num_signals = min((RankStat <= decision_thres).sum(), num_cell)
     # elif decision_metric == "rank":
     #   if ranking_statistic == "p_value":
     #       num_signals = (RankStat <= decision_thres).sum()
     #   else:
     #       num_signals = (RankStat >= decision_thres).sum()

    RC = Container()
    RC.all_signals = pd.DataFrame(
        {
            "Product": DATA["product_name"].values,
            "Adverse Event": DATA["ae_name"].values,
            "Count": n11,
            # Vincent PAVAN, 18/09/2026
            # delete Expected Count and Rankstat from output
            #"Expected Count": expected,
            #"p_value": RankStat,
            # Vincent PAVAN, 17/09/2026
            # correction: replace "PRR = np.exp(log_ror)" by "ROR = np.exp(log_ror)
            # "PRR": np.exp(log_ror),
            "ROR": np.exp(log_ror),
            "LB(IC 95%)" : np.exp(LB),
            "UP(IC 95%)" : np.exp(UB),
            "p-value" : pval_uni,
            # "product margin": n1j,
            # "event margin": ni1,
            # "fdr": FDR,
        },
        index=np.arange(len(n11)),
    ).sort_values(by=["LB(IC 95%)"], ascending = False)

    # if ranking_statistic == "CI":
    #    RC.all_signals = RC.all_signals.rename(columns={"p_value": "lower_bound_CI(95%)"}).sort_values(
    #        by=["lower_bound_CI(95%)"]
    #    )

    RC.signals = RC.all_signals.iloc[
        0:num_signals,
    ]
    RC.num_signals = num_signals
    return RC
