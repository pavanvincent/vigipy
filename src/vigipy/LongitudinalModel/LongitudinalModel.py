# import pandas as pd
# import traceback
# from ..utils import convert, convert_binary, convert_multi_item
#
#
# class LongitudinalModel:
#
#    CONVERSION_TYPES = {"base", "binary", "multi-item"}
#
#    def __init__(self, dataframe, time_unit):
#        """
#        Initialize the longitudinal model with raw data and a time unit.
#
#        Arguments:
#            dataframe (Pandas DataFrame): A dataframe containing counts, AEs,
#                                          product/brands and AE dates.
#
#            time_unit (str): One of Pandas' time unit aliases found here: https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases
#
#        """
#        self.time_unit = time_unit
#        dataframe["date"] = pd.to_datetime(dataframe["date"])
#        self.date_groups = dataframe.resample(time_unit, on="date")
#        self.data = dataframe
#
#    def _convert(self, data, conversion_type, conversion_kwargs):
#        if conversion_type not in self.CONVERSION_TYPES:
#            raise ValueError(f"Provided `conversion_type` not in {self.CONVERSION_TYPES}")
#
#        if conversion_kwargs is None:
#            conversion_kwargs = {}
#
#        if conversion_type == "base":
#            return convert(data, **conversion_kwargs)
#        elif conversion_type == "binary":
#            return convert_binary(data, **conversion_kwargs)
#        elif conversion_type == "multi-item":
#            return convert_multi_item(data, **conversion_kwargs)
#
#
#    # def run(self, model, include_gaps=True, conversion_type="base", conversion_kwargs=None, **kwargs):
#    #    """
#    #    Run the longitudinal model as initialized.
#    #
#    #    Arguments:
#    #        model (vigipy model): One of the supported vigipy models
#    #                              from this module (i.e. gps, prr, etc)
#    #
#   #        include_gaps (bool): If a particular time slice has a
#    #                             zero-count sum, still run the model?
#    #
#    #        kwargs: key word arguments can be added to this function call
#    #                and they will be passed into the model at run time.
#    #
#    #    """
#    #    self.results = []
#    #    for timestamp, count in self.date_groups.sum()["count"].items():
#    #        if count == 0:
#    #            if include_gaps:
#    #                self.results.append((timestamp, None))
#    #            continue
#    #
#    #        subset = self.data.loc[self.data["date"] <= timestamp]
#    #        sub_container = self._convert(subset, conversion_type, conversion_kwargs)
#    #        self._run_model(model, sub_container, timestamp, include_gaps, kwargs)
#
#    def run(self, model, include_gaps=True, conversion_type="base", conversion_kwargs=None, n_jobs=-1, **kwargs):
#        """Version parallélisée de run() utilisant joblib"""
#        num_cores = multiprocessing.cpu_count() if n_jobs == -1 else n_jobs
#        
#        # Préparation des tâches à envoyer aux cœurs CPU
#        tasks = [
#            delayed(self._worker_run)(
#                timestamp, count, model, include_gaps, conversion_type, conversion_kwargs, kwargs
#            )
#            for timestamp, count in self.date_groups.sum()["count"].items()
#        ]
#        
#        # Exécution parallèle sur les 8 cœurs de Colab Pro
#        raw_results = Parallel(n_jobs=num_cores)(tasks)
#        
#        # Nettoyage des résultats (on filtre les valeurs None si include_gaps=False)
#        self.results = [res for res in raw_results if res is not None]
#
#    def run_disjoint(self, model, include_gaps=True, conversion_type="base", conversion_kwargs=None, **kwargs):
#        """
#        Run the longitudinal model as initialized.
#
#        Arguments:
#            model (vigipy model): One of the supported vigipy models
#                                  from this module (i.e. gps, prr, etc)
#
#            include_gaps (bool): If a particular time slice has a
#                                 zero-count sum, still run the model?
#
#            kwargs: key word arguments can be added to this function call
#                    and they will be passed into the model at run time.
#
#        """
#        self.results = []
#        for count, (timestamp, subset) in zip(self.date_groups.sum()["count"], self.date_groups):
#            if count == 0:
#                if include_gaps:
#                    self.results.append((timestamp, None))
#                continue
#
#            sub_container = self._convert(subset, conversion_type, conversion_kwargs)
#            self._run_model(model, sub_container, timestamp, include_gaps, kwargs)
#
#    def _run_model(self, model, sub_container, timestamp, include_gaps, kwargs):
#        try:
#            da_results = model(sub_container, **kwargs)
#            self.results.append((timestamp, da_results))
#        except ValueError as e:
#            print(traceback.format_exc())
#            if include_gaps:
#                self.results.append((timestamp, None))
#            print(f"Insufficient data for this model. Skipping this slice: {timestamp}")
#
#    def regroup_dates(self, time_unit):
#        """
#        Regroup the data by a new time unit.
#
#        Arguments:
#            time_unit (str): One of Pandas' time unit aliases. (Q, A, QS, etc.)
#
#        """
#        self.time_unit = time_unit
#        self.date_groups = self.data.resample(time_unit, on="date")

import pandas as pd
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

    # def _worker_disjoint(self, timestamp, subset, count, model, include_gaps, conversion_type, conversion_kwargs, kwargs):
    #    if count == 0:
    #        return (timestamp, None) if include_gaps else None
    #    try:
    #        sub_container = self._convert(subset, conversion_type, conversion_kwargs)
    #        da_results = model(sub_container, **kwargs)
    #        return (timestamp, da_results)
    #    except ValueError:
    #        return (timestamp, None) if include_gaps else None

     def _worker_disjoint(self, timestamp, end_idx, model, include_gaps, conversion_type, conversion_kwargs, kwargs):
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



    def run_disjoint(self, model, include_gaps=True, conversion_type="base", conversion_kwargs=None, n_jobs=-1, **kwargs):
        """Version optimisée : Extraction des données en amont pour éviter l'overhead de resample"""
        num_cores = multiprocessing.cpu_count() if n_jobs == -1 else n_jobs
        
        # 1. On extrait les données de manière statique (très léger pour la mémoire)
        counts = self.date_groups.sum()["count"]
        # Extraction propre sans itérer sur l'objet resample vivant
        group_data = [(timestamp, self.data.loc[idx]) for timestamp, idx in self.date_groups.groups.items()]
        
        # 2. Préparation des tâches
        tasks = [
            delayed(self._worker_disjoint)(
                timestamp, subset, counts.get(timestamp, 0), model, include_gaps, conversion_type, conversion_kwargs, kwargs
            )
            for timestamp, subset in group_data
        ]
        
        # 3. Exécution parallèle pure
        raw_results = Parallel(n_jobs=num_cores, backend="multiprocessing")(tasks)
        self.results = [res for res in raw_results if res is not None]
        return self.results

    # def run(self, model, include_gaps=True, conversion_type="base", conversion_kwargs=None, n_jobs=-1, **kwargs):
    #    """
    #    Version cumulée optimisée : Si le modèle sous-jacent (ex: GPS) accepte l'accumulation,
    #    on devrait sommer les conteneurs disjoints au lieu de re-calculer les gros DataFrames.
    #    Sinon, l'extraction est ici fiabilisée via les index natifs de Pandas.
    #    """
    #    num_cores = multiprocessing.cpu_count() if n_jobs == -1 else n_jobs
    #    timestamps = list(self.date_groups.groups.keys())
    #   
    #    
    #    # Optimisation : On utilise la recherche dichotomique ou le slicing d'index pré-triés
    #    tasks = []
    #    for timestamp in timestamps:
    #        # Slicing ultra-rapide sur DataFrame préalablement trié par date
    #        subset = self.data[self.data["date"] <= timestamp]
    #        count = len(subset)
    #        
    #        tasks.append(
    #            delayed(self._worker_disjoint)(
    #                timestamp, subset, count, model, include_gaps, conversion_type, conversion_kwargs, kwargs
    #            )
    #        )
    #        
    #    raw_results = Parallel(n_jobs=num_cores, backend="multiprocessing")(tasks)
    #    self.results = [res for res in raw_results if res is not None]
    #    return self.results

    def run(self, model, include_gaps=True, conversion_type="base", conversion_kwargs=None, n_jobs=-1, **kwargs):
        """
        Version cumulée optimisée : Cherche les index de découpe en O(log N) 
        et transmet uniquement des pointeurs d'index aux workers pour éviter le crash mémoire.
        """
        num_cores = multiprocessing.cpu_count() if n_jobs == -1 else n_jobs
        timestamps = list(self.date_groups.groups.keys())
        
        # Recherche dichotomique ultra-rapide (O(log N)) pour trouver la position de fin
        end_indices = np.searchsorted(self.data["date"].values, timestamps, side="right")
        
        # Préparation des tâches : on passe l'entier `end_idx` au lieu du DataFrame lourd `subset`
        tasks = [
            delayed(self._worker_disjoint)(
                timestamp, end_idx, model, include_gaps, conversion_type, conversion_kwargs, kwargs
            )
            for timestamp, end_idx in zip(timestamps, end_indices)
        ]
        
        # Utilisation de "loky" (gestion efficace de la mémoire partagée en lecture seule)
        raw_results = Parallel(n_jobs=num_cores, backend="loky")(tasks)
        
        # Filtrage des None si include_gaps=False (car votre worker renvoie None dans ce cas)
        self.results = [res for res in raw_results if res is not None]
        return self.results

    def regroup_dates(self, time_unit):
        self.time_unit = time_unit
        self.date_groups = self.data.resample(time_unit, on="date")

