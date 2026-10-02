import pandas as pd
import numpy as np
import multiprocessing
from joblib import Parallel, delayed
from ..utils import convert, convert_binary, convert_multi_item

class LongitudinalModel:

    CONVERSION_TYPES = {"base", "binary", "multi-item"}

    def __init__(self, dataframe, time_unit):
        self.time_unit = time_unit
        dataframe["date"] = pd.to_datetime(dataframe["date"])
        # Sorting is mandatory and essential for np.searchsorted (binary search).
        self.data = dataframe.sort_values("date").reset_index(drop=True)
        self.date_groups = self.data.resample(time_unit, on="date")
        self.results = []

    def _convert(self, data, conversion_type, conversion_kwargs):
        if conversion_type not in self.CONVERSION_TYPES:
            raise ValueError(f"Provided `conversion_type` not in {self.CONVERSION_TYPES}")

        conversion_kwargs = conversion_kwargs or {}
        if conversion_type == "base":
            return convert(data, **conversion_kwargs)
        elif conversion_type == "binary":
            return convert_binary(data, **conversion_kwargs)
        elif conversion_type == "multi-item":
            return convert_multi_item(data, **conversion_kwargs)


    def _execute_disjoint_task(self, timestamp, end_idx, model, include_gaps, conversion_type, conversion_kwargs, kwargs):
        # If no record exists up to that date
        if end_idx == 0:
            return (timestamp, None) if include_gaps else None
            
        try:
            # Positional slicing (.iloc) is virtual and instantaneous here.
            subset = self.data.iloc[:end_idx]
            
            sub_container = self._convert(subset, conversion_type, conversion_kwargs)
            da_results = model(sub_container, **kwargs)
            return (timestamp, da_results)
            
        except ValueError:
            return (timestamp, None) if include_gaps else None

    def run_disjoint(self, model, include_gaps=True, conversion_type="base", conversion_kwargs=None, start_date=None, end_date=None, **kwargs):
    """Sequential version: Extraction of upstream data filtered over an interval [start_date, end_date]"""
    # 1. Sécurisation de conversion_kwargs contre l'erreur d'ambiguïté DataFrame
    if conversion_kwargs is None:
        conversion_kwargs = {}

    # 2. Extraction sécurisée des groupes via get_group() (évite le KeyError sur l'index)
    counts = self.date_groups.sum()["count"]
    group_data = [(timestamp, self.date_groups.get_group(timestamp)) for timestamp in self.date_groups.groups.keys()]
    
    # Filtrage par start_date et end_date
    if start_date is not None:
        start_ts = pd.to_datetime(start_date)
        group_data = [(ts, sub) for ts, sub in group_data if ts >= start_ts]
        
    if end_date is not None:
        end_ts = pd.to_datetime(end_date)
        group_data = [(ts, sub) for ts, sub in group_data if ts <= end_ts]
    
    # 3. Exécution séquentielle avec dépaquetage **kwargs
    raw_results = []
    for timestamp, subset in group_data:
        res = self._execute_disjoint_task(
            timestamp, subset, counts.get(timestamp, 0), model, include_gaps, conversion_type, conversion_kwargs, **kwargs
        )
        raw_results.append(res)
    
    self.results = [res for res in raw_results if res is not None]
    return self.results

    
    def run(self, model, include_gaps=True, conversion_type="base", conversion_kwargs=None, start_date=None, end_date=None, **kwargs):
        """
        Sequential cumulative version: Allows filtering on an interval [start_date, end_date].
        """
        timestamps = list(self.date_groups.groups.keys())
        
        # Filtering by start_date and end_date
        if start_date is not None:
            start_ts = pd.to_datetime(start_date)
            timestamps = [ts for ts in timestamps if ts >= start_ts]
            
        if end_date is not None:
            end_ts = pd.to_datetime(end_date)
            timestamps = [ts for ts in timestamps if ts <= end_ts]
        
        # We use Pandas' native .searchsorted() function on the Series
        end_indices = self.data["date"].searchsorted(timestamps, side="right")

        # Pure sequential execution
        raw_results = []
        for timestamp, end_idx in zip(timestamps, end_indices):
            res = self._execute_disjoint_task(
                timestamp, end_idx, model, include_gaps, conversion_type, conversion_kwargs, kwargs
            )
            raw_results.append(res)
        
        # Filtering of None entries if include_gaps=False
        self.results = [res for res in raw_results if res is not None]
        return self.results

    def regroup_dates(self, time_unit):
        self.time_unit = time_unit
        self.date_groups = self.data.resample(time_unit, on="date")

