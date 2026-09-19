import numpy as np
import pandas as pd
from scipy.stats import fisher_exact, hypergeom, norm
from ..utils import Container


def rfet(
    container,
    
    min_events=1,
    mid_pval=False,
):
    """
    Calculate the reporting odds ratio.

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

    log_rfet = np.log(n11 * n00 / (n10 * n01))
    var_log_rfet = 1.0 / n11 + 1.0 / n10 + 1.0 / n01 + 1.0 / n00
    LB = norm.ppf(0.025, log_rfet, np.sqrt(var_log_rfet))
    UB = norm.ppf(0.975,log_rfet, np.sqrt(var_log_rfet))
    
    pval_fish_uni = np.empty((num_cell))
    for p in range(num_cell):
        table = [[n11[p], n10[p]], [n01[p], n00[p]]]
        pval_fish_uni[p] = fisher_exact(table, alternative="greater")[1]

    if mid_pval:
        for p in range(num_cell):
            pval_fish_uni[p] = pval_fish_uni[p] - 0.5 * hypergeom.pmf(
                n11[p], n11[p] + n10[p], n11[p] + n01[p], n10[p] + n00[p]
            )

    pval_uni = pval_fish_uni
    pval_uni[pval_uni > 1] = 1
    pval_uni[pval_uni < 0] = 0
    RankStat = pval_uni

   
    num_signals = (LB > 0).sum()

    RC = Container()
    RC.all_signals = pd.DataFrame(
        {
            "Product": DATA["product_name"].values,
            "Adverse Event": DATA["ae_name"].values,
            "Count": n11,
            # "Expected Count": expected,
            "ROR": np.exp(log_rfet),
            "LB(95 %)" : LB,
            "UB(95 %)" : UB,
            "p_value": pval_uni,
        },
        index=np.arange(len(n11)),
    ).sort_values(by=["LB(95 %)"], ascending = False)

    RC.signals = RC.all_signals.iloc[
        0:num_signals,
    ]
    RC.num_signals = num_signals
    return RC
