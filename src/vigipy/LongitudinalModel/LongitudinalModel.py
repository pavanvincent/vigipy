import pandas as pd
import numpy as np
import multiprocessing
from joblib import Parallel, delayed
from ..utils import convert, convert_binary, convert_multi_item

class LongitudinalModel:

    CONVERSION_TYPES = {"base", "binary", "multi-item"}

   # def __init__(self, dataframe, time_unit):
   #     self.time_unit = time_unit
   #     dataframe["date"] = pd.to_datetime(dataframe["date"])
   #     # On trie obligatoirement par date pour accélérer les sélections vectorisées (.loc)
   #     self.data = dataframe.sort_values("date").reset_index(drop=True)
   #     self.date_groups = self.data.resample(time_unit, on="date")

    def __init__(self, dataframe, time_unit):
        self.time_unit = time_unit
        dataframe["date"] = pd.to_datetime(dataframe["date"])
        # Tri obligatoire indispensable pour np.searchsorted (recherche dichotomique)
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
        # Si aucun enregistrement n'existe jusqu'à cette date
        if end_idx == 0:
            return (timestamp, None) if include_gaps else None
            
        try:
            # Le slicing par position (.iloc) est virtuel et instantané ici
            subset = self.data.iloc[:end_idx]
            
            # Votre logique métier
            sub_container = self._convert(subset, conversion_type, conversion_kwargs)
            da_results = model(sub_container, **kwargs)
            return (timestamp, da_results)
            
        except ValueError:
            return (timestamp, None) if include_gaps else None

    def run_disjoint(self, model, include_gaps=True, conversion_type="base", conversion_kwargs=None, **kwargs):
        """Version séquentielle : Extraction des données en amont sans overhead de processus"""
        # 1. On extrait les données de manière statique
        counts = self.date_groups.sum()["count"]
        group_data = [(timestamp, self.data.loc[idx]) for timestamp, idx in self.date_groups.groups.items()]
        
        # 2. Exécution séquentielle simple
        raw_results = []
        for timestamp, subset in group_data:
            res = self._execute_disjoint_task(
                timestamp, subset, counts.get(timestamp, 0), model, include_gaps, conversion_type, conversion_kwargs, kwargs
            )
            raw_results.append(res)
        
        self.results = [res for res in raw_results if res is not None]
        return self.results

    def run(self, model, include_gaps=True, conversion_type="base", conversion_kwargs=None, **kwargs):
        """
        Version cumulée séquentielle : Cherche les index de découpe en O(log N) 
        et exécute les tâches l'une après l'autre de manière stable.
        """
        timestamps = list(self.date_groups.groups.keys())
        
        # On utilise le .searchsorted() natif de Pandas sur la Série
        end_indices = self.data["date"].searchsorted(timestamps, side="right")

        # Exécution séquentielle pure
        raw_results = []
        for timestamp, end_idx in zip(timestamps, end_indices):
            res = self._execute_disjoint_task(
                timestamp, end_idx, model, include_gaps, conversion_type, conversion_kwargs, kwargs
            )
            raw_results.append(res)
        
        # Filtrage des None si include_gaps=False
        self.results = [res for res in raw_results if res is not None]
        return self.results

    def regroup_dates(self, time_unit):
        self.time_unit = time_unit
        self.date_groups = self.data.resample(time_unit, on="date")

