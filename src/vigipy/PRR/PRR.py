import warnings
import numpy as np
import pandas as pd
from scipy.stats import norm

from ..utils.lbe import lbe
from ..utils import Container
from ..utils import calculate_expected


def prr(
    container,
    min_events=1,
):
    """
    Calculate the proportional reporting ratio.

    Arguments:
        container: A DataContainer object produced by the convert()
                    function from data_prep.py

        min_events: The min number of AE reports to be considered a signal


    """
    DATA = container.data
    N = container.N

    if min_events > 1:
        DATA = DATA[DATA.events >= min_events]

    n11 = np.asarray(DATA["events"], dtype=np.float64)
    n1j = np.asarray(DATA["product_aes"], dtype=np.float64)
    ni1 = np.asarray(DATA["count_across_brands"], dtype=np.float64)
    num_cell = len(n11)
   
    n10 = n1j - n11
    n01 = ni1 - n11 + 1e-7
    n00 = N - (n11 + n10 + n01)

    log_prr = np.log((n11 / (n11 + n10)) / (n01 / (n01 + n00)))
    var_log_prr = 1 / n11 - 1 / (n11 + n10) + 1 / n01 - 1 / (n01 + n00)

    prr_H0 = 1
    pval_uni = 1 - norm.cdf(log_prr, np.log(prr_H0), np.sqrt(var_log_prr))
    pval_uni[pval_uni > 1] = 1
    pval_uni[pval_uni < 0] = 0

   
    log_LB = norm.ppf(0.025, log_prr, np.sqrt(var_log_prr))
    log_UB = norm.ppf(0.975, log_prr, np.sqrt(var_log_prr))
            
    num_signals = (log_LB > 0).sum()
    RC = Container()
    RC.all_signals = pd.DataFrame(
        {
            "Product": DATA["product_name"].values,
            "Adverse Event": DATA["ae_name"].values,
            "Count": n11,
            # Vincent PAVAN, 18/09/2026
            # delete Expected Count and Rankstat from output
            # "Expected Count": expected,
            # "p_value": RankStat,
            "PRR": np.exp(log_prr),
            # Vincent PAVAN, 16/09/2026
            # Add lower and upper bound for the Confident interval of PRR, and pvalue
            "LB(CI 95%)" : np.exp(LB),
            "UB(CI 95%)" : np.exp(UB),
            "p-value" : pval_uni,
            # "product margin": n1j,
            # "event margin": ni1,
            # "fdr": FDR,
        },
        index=np.arange(len(n11)),
    # Vincent PAVAN, 18/09/2026
    # sort by LB
    ).sort_values(by=["LB(CI 95%)"], ascending = False)

    # Vincent PAVAN, 18/09/2026
    # rankign_statistic is LB
    # if ranking_statistic == "CI":
    #    RC.all_signals = RC.all_signals.rename(columns={"p_value": "lower_bound_CI(95%)"}).sort_values(
    #        by=["lower_bound_CI(95%)"]
    #    )

    RC.signals = RC.all_signals.iloc[
        0:num_signals,
    ]
    RC.num_signals = num_signals
    return RC
