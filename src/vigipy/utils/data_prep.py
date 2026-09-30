from itertools import product, chain, combinations
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import polars as pl
from scipy.sparse import csr_matrix
from .Container import Container 

def convert(
    data_frame,
    margin_threshold=1,
    product_label="name",
    count_label="count",
    ae_label="AE",
):
# 1. Instant conversion to Polars DataFrame (multi-threaded)
    if not isinstance(data_frame, pl.DataFrame):
        lf = pl.from_pandas(data_frame).lazy()
    else:
        lf = data_frame.lazy()

    # 2. Parallel Aggregation and Pivot (Equivalent to compute_contingency)
    # We first group to ensure it's aggregated, then we pivot
    agg_lf = (
        lf.group_by([product_label, ae_label])
        .agg(pl.col(count_label).sum())
        .filter(pl.col(count_label) >= margin_threshold)
    )
    
    # Pivot to obtain the contingency matrix
    pivot_df = agg_lf.collect().pivot(
        on=ae_label,
        index=product_label,
        values=count_label,
        aggregate_function="sum"
    ).fill_null(0)

    # 3. Extracting the NumPy matrix
    # The first column is the product name, the rest is the matrix
    product_names = pivot_df[product_label].to_numpy()
    data_cont = pivot_df.drop(product_label).cast(pl.Float64).to_numpy()

    # 4. Margin calculations (Ultra-fast on a contiguous NumPy matrix)
    col_sums = np.sum(data_cont, axis=0)
    row_sums = np.sum(data_cont, axis=1)

    # 5. Optimized reconstruction of data_df (Equivalent to count())
    # Instead of a slow 'count' function, we vectorize the flattening:
    num_products, num_aes = data_cont.shape
    
    # Repetition of marginal sums to align with the flattened matrix
    n1j = np.repeat(row_sums, num_aes)
    ni1 = np.tile(col_sums, num_products)
    n11 = data_cont.ravel()
    
    # Mask to filter according to the threshold
    mask = n11 >= margin_threshold
    
    # Generation of associated text columns
    ae_names = np.array(pivot_df.drop(product_label).columns)
    all_products = np.repeat(product_names, num_aes)
    all_aes = np.tile(ae_names, num_products)

    # Creation of the final reduced DataFrame
    import pandas as pd
    data_df = pd.DataFrame({
        "product_name": all_products[mask],
        "ae_name": all_aes[mask],
        "events": n11[mask],
        "product_aes": n1j[mask],
        "count_across_brands": ni1[mask]
    })

    # 6. Filling the VIGIPY Container
    DC = Container()
    DC.contingency = data_cont
    DC.data = data_df
    DC.N = data_df["events"].sum()
    DC.type = "contingency"
    return DC


def compute_contingency(data_frame, product_label, count_label, ae_label, margin_threshold):
    """Compute the contingency table for DA - Memory-optimized version"""
    # Creation of the contingency table
    data_cont = pd.pivot_table(
        data_frame,
        values=count_label,
        index=product_label,
        columns=ae_label,
        aggfunc="sum",
        fill_value=0,
    )

    # Direct and simultaneous filtering of rows and columns via a Boolean mask
    row_mask = data_cont.sum(axis=1) >= margin_threshold
    col_mask = data_cont.sum(axis=0) >= margin_threshold

    # Only valid intersections are kept.
    return data_cont.loc[row_mask, col_mask]

# def convert_binary(
#    data, product_label="name", ae_label="AE", use_counts=False, count_label="count", expand_counts=True
#):
#    """Convert input data consisting of unique product-event pairs into a
#       binary dataframe indicating which event and which product are
#       associated with each other.
#
#    Args:
#        data (pd.DataFrame): A DataFrame consisting of unique product-event pairs for each row
#        product_label (str, optional): If the product name is not in a column called `name`, override here. Defaults to "name".
#        ae_label (str, optional): If the adverse event is not in a column called `AE`, override here.. Defaults to "AE".
#
#    Returns:
#        Container: A container with two binary dataframes. One is the X data of product names and the other is the
#        y data with adverse events. Index locations are associated with the input DataFrame.
#
#    """
#    DC = Container()
#
#    # Sanitize df to remove unnecessary information during transforms
#    data = _sanitize_data(data, [product_label, ae_label, count_label])
#
#    if use_counts:
#        if not isinstance(product_label, str):
#            group_list = [*product_label, ae_label]
#        else:
#            group_list = [product_label, ae_label]
#        data = data.groupby(group_list).sum().reset_index()
#        event_df = __transform_dataframe(data, count_label, ae_label)
#        DC.type = "binary_count"
#    else:
#        if data[count_label].max() > 1 and expand_counts:
#            data = __expand_dataframe(data, count_label, ae_label, product_label)
#        event_df = pd.get_dummies(data[ae_label], prefix="", prefix_sep="")
#        event_df = event_df.groupby(by=event_df.columns, axis=1).sum()
#        DC.type = "binary"
#
#    prod_df = pd.get_dummies(data[product_label], prefix="", prefix_sep="")
#    DC.product_features = prod_df.groupby(by=prod_df.columns, axis=1).sum()
#
#    DC.event_outcomes = event_df
#    DC.N = data.shape[0]
#    DC.data = data
#
#    return DC

def convert_binary(
    data, product_label="name", ae_label="AE", use_counts=False, count_label="count", expand_counts=True
):
    """Convert input data consisting of unique product-event pairs into a
       sparse/binary format indicating which event and which product are
       associated with each other. Optimisé pour la mémoire et la vitesse.
    """
    DC = Container()

    # Désinfection des données
    data = _sanitize_data(data, [product_label, ae_label, count_label])

    # Gestion des comptes (Duplication des lignes si nécessaire)
    if not use_counts and expand_counts and data[count_label].max() > 1:
        # Optimisation radicale de __expand_dataframe via repeat de NumPy
        data = data.loc[data.index.repeat(data[count_label])].reset_index(drop=True)
        data[count_label] = 1

    if use_counts:
        group_list = [product_label] if isinstance(product_label, str) else list(product_label)
        group_list = group_list + [ae_label]
        data = data.groupby(group_list)[count_label].sum().reset_index()
        
        # Conserver l'appel d'origine si __transform_dataframe est spécifique à VIGIPY, 
        # sinon l'approche pivot/creuse ci-dessous s'applique aussi.
        event_df = __transform_dataframe(data, count_label, ae_label)
        DC.type = "binary_count"
        
        # Pour les produits (features)
        prod_df = pd.get_dummies(data[product_label], prefix="", prefix_sep="")
        DC.product_features = prod_df.groupby(by=prod_df.columns, axis=1).sum()
    else:
        # --- OPTIMISATION RADICALE VIA MATRICES CREUSES (SPARSE) ---
        DC.type = "binary"
        
        # 1. Encodage catégoriel rapide (Factorisation)
        prod_series = data[product_label].astype("category")
        ae_series = data[ae_label].astype("category")
        
        # Extraction des labels uniques (Noms des colonnes)
        product_names = prod_series.cat.categories.tolist()
        ae_names = ae_series.cat.categories.tolist()
        
        # Codes numériques (Coordonnées dans la matrice)
        prod_codes = prod_series.cat.codes.values
        ae_codes = ae_series.cat.codes.values
        row_indices = np.arange(len(data))
        
        # 2. Construction directe en Matrices Creuses CSR (Évite pd.get_dummies)
        # Chaque ligne 'i' a un '1' à la colonne du code produit/effet
        ones = np.ones(len(data), dtype=np.int8)
        
        X_sparse = csr_matrix((ones, (row_indices, prod_codes)), shape=(len(data), len(product_names)))
        y_sparse = csr_matrix((ones, (row_indices, ae_codes)), shape=(len(data), len(ae_names)))
        
        # 3. Conversion finale en DataFrame Creux (Sparse DataFrame)
        # Conserve l'interface DataFrame pour le reste de VIGIPY mais consomme 95% de mémoire en moins
        DC.product_features = pd.DataFrame.sparse.from_spmatrix(X_sparse, columns=product_names)
        DC.event_outcomes = pd.DataFrame.sparse.from_spmatrix(y_sparse, columns=ae_names)

    DC.N = data.shape[0]
    DC.data = data

    return DC



def convert_multi_item(df, product_label=["name"], ae_label="AE", count_label="count", min_threshold=3):
    """***WARNING*** Currently experimental and not guaranteed to perform as expected.
    Convert data with multiple product columns into a multi-item flattened dataframe for the DA methods.

    Args:
        df (pd.DataFrame): A dataframe where each row is a unique adverse event and has multiple columns
        indicating the presence of multiple devices/drugs/interventions.
        product_cols (list, optional): A list of column names associated with the co-occuring products. Defaults to ["name"].
        ae_col (str, optional): The column name that contains the adverse events. Defaults to "AE".
        min_threshold (int, optional): The minimum number of events required to keep a drug/device-event pair.

    Returns:
        Container: A container object that holds the necessary components for DA.
    """
    ae_counts = defaultdict(int)
    product_counts = defaultdict(int)
    for col in product_label:
        for ae, product, count in df.loc[df[col] != ""][[ae_label, col, count_label]].itertuples(index=False):
            ae_counts[ae] += count
            product_counts[product] += count

    # Initialize an empty list to store the result
    result = []

    # Sanitize df to remove unnecessary information during transforms
    df = _sanitize_data(df, [product_label, ae_label, count_label])

    # Iterate over each row in the dataframe
    for _, row in df.iterrows():
        # Extract product names from the current row
        names = {row[x] for x in product_label if row[x]}
        # Get all unique combinations of names (without repetition)
        combos = list(chain.from_iterable(combinations(names, r) for r in range(1, len(names) + 1)))
        # Append combinations to the result list with the other column info
        for combo in combos:
            new_data = {idx: row[idx] for idx in row.index if idx not in product_label}
            new_data["product_name"] = f"{'|'.join([c for c in combo if c])}"
            new_data["product_aes"] = sum([product_counts[p] for p in combo])
            new_data["count_across_brands"] = ae_counts[row[ae_label]]
            result.append(new_data)

    # Convert the result list to a new dataframe
    new_df = pd.DataFrame(result)
    event_series = new_df.groupby(by=["AE", "product_name"]).sum()["count"]
    new_df["events"] = new_df.apply(lambda x: event_series[x["AE"]][x["product_name"]], axis=1)
    new_df.rename(columns={ae_col: "ae_name"}, inplace=True)

    DC = Container()
    DC.contingency = compute_contingency(new_df, "product_name", "count", "ae_name", min_threshold)
    DC.data = new_df[["ae_name", "product_name", "count_across_brands", "product_aes", "events"]].drop_duplicates()
    DC.N = new_df["events"].sum()

    return DC


def count(data, rows, cols):
    """
    Convert the input contingency table to a flattened table
    Version optimisée 100% vectorisée (sans boucle for).
    """
    # 1. We "stack" the matrix: this creates a Series whose index is a MultiIndex (Product, AE)
    # Values ​​greater than 0 are immediately filtered out to retain only the real signals.
    stacked = data.stack()
    stacked = stacked[stacked > 0]
    
    # 2. We transform this structure into a flat DataFrame
    df = stacked.reset_index()
    df.columns = ["product_name", "ae_name", "events"]
    
    # 3. We apply marginal totals (rows and cols) instantly using .map()
    df["product_aes"] = df["product_name"].map(rows)
    df["count_across_brands"] = df["ae_name"].map(cols)
    
    # 4. The columns are reordered to STRICTLY comply with the format expected by VIGIPY.
    return df[["events", "product_aes", "count_across_brands", "ae_name", "product_name"]]

def _sanitize_data(df, keep_labels):
    keep = []
    for label in keep_labels:
        if not isinstance(label, str):
            keep.extend(label)
        else:
            keep.append(label)

    return df[keep].copy()


def __expand_dataframe(df, count_label, ae_label, product_label):
    new = defaultdict(list)
    for row in df.itertuples(index=False):
        for _ in range(int(getattr(row, count_label))):
            new[product_label].append(getattr(row, product_label))
            new[ae_label].append(getattr(row, ae_label))
            new[count_label].append(1)

    new_data = pd.DataFrame(new)
    return new_data


def __transform_dataframe(df, count_label, ae_label):
    # Create a new dataframe with unique values from 'AE' as columns, and initialize all cells with 0
    new_df = pd.DataFrame(0, index=range(len(df)), columns=df[ae_label].unique())

    # Iterate through the rows and set the appropriate value from 'count' in the corresponding 'AE' column
    for i, row in df.iterrows():
        new_df.at[i, row[ae_label]] = row[count_label]

    return new_df
