import warnings
import openml
import pandas as pd
import numpy as np
from typing import Dict, Tuple, Union
import matplotlib.pyplot as plt
import seaborn as sns
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, AdaBoostRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from gplearn.genetic import SymbolicTransformer, SymbolicRegressor
from evolutionary_forest.forest import EvolutionaryForestRegressor
from evolutionary_forest.utils import get_feature_importance
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.base import clone
from xgboost import XGBRegressor

from Functions import select_st_features_by_threshold, select_ef_features_by_threshold, categorize_to_numeric, \
    apply_z_score_standardization


def get_algorithm_scores(X_train, y_train, X_test, y_test, new_train, new_test, dataset_name):
    regressor_list = ['RF', 'XGBoost']

    regressor_dict = {
        'RF': RandomForestRegressor(n_estimators=200, n_jobs=-1),
        'XGBoost': XGBRegressor(n_jobs=1, n_estimators=200, verbosity=0),
    }

    results = []

    for regr_name in regressor_list:
        try:
            base_model = clone(regressor_dict[regr_name])
            base_model.fit(X_train, y_train)
            b_score = r2_score(y_test, base_model.predict(X_test))

            enh_model = clone(regressor_dict[regr_name])
            enh_model.fit(new_train, y_train)
            e_score = r2_score(y_test, enh_model.predict(new_test))

            diff = e_score - b_score

            results.append({
                'Algorithm': regr_name,
                'Dataset': dataset_name,
                'Base_Score': b_score,
                'Improved_Score': e_score,
                'Difference': diff
            })

        except Exception as e:
            print(f"Hata ({regr_name} - {dataset_name}): {e}")
            results.append({
                'Algorithm': regr_name, 'Dataset': dataset_name,
                'Base_Score': 0, 'Improved_Score': 0, 'Difference': 0
            })

    return results

def ef_fe(sets_id, threshold, min_importance_threshold):
    np.seterr(divide='ignore', invalid='ignore')
    all_results = []

    for set_id_val in sets_id:
        try:
            dataset = openml.datasets.get_dataset(set_id_val)
            d_name = f"{dataset.name}({set_id_val})"
            X, y, _, _ = dataset.get_data(dataset_format="dataframe", target=dataset.default_target_attribute)

            X = categorize_to_numeric(X)
            X, y = apply_z_score_standardization(X, y)

            X_vals = np.nan_to_num(X.values.astype(np.float64))
            y_vals = y.values if isinstance(y, pd.Series) else y
            y_vals = y_vals.astype(np.float64)

            X_train, X_test, y_train, y_test = train_test_split(X_vals, y_vals, test_size=0.2, random_state=42)

            if threshold:
                ef = EvolutionaryForestRegressor()
                with np.errstate(divide='ignore', invalid='ignore'):
                    ef.fit(X_train, y_train)
                    X_train_ef = ef.transform(X_train)
                    X_test_ef = ef.transform(X_test)
                    X_train_ef = np.nan_to_num(X_train_ef)
                    X_test_ef = np.nan_to_num(X_test_ef)

                feature_importance_dict = get_feature_importance(ef)

                X_train_top_ef, X_test_top_ef = select_ef_features_by_threshold(
                    feature_importance_dict=feature_importance_dict,
                    X_train_transformed=X_train_ef,
                    X_test_transformed=X_test_ef,
                    min_importance_threshold=min_importance_threshold,
                )

                new_train = np.hstack((X_train, X_train_top_ef))
                new_test = np.hstack((X_test, X_test_top_ef))

                dataset_results = get_algorithm_scores(X_train, y_train, X_test, y_test, new_train, new_test, d_name)
                all_results.extend(dataset_results)

            else:

                ef = EvolutionaryForestRegressor()
                with np.errstate(divide='ignore', invalid='ignore'):
                    ef.fit(X_train, y_train)
                    X_train_ef = ef.transform(X_train)
                    X_test_ef = ef.transform(X_test)
                    X_train_ef = np.nan_to_num(X_train_ef)
                    X_test_ef = np.nan_to_num(X_test_ef)

                if X_train_ef.shape[1] > 10:
                    X_train_ef = X_train_ef[:, :10]
                    X_test_ef = X_test_ef[:, :10]

                new_train = np.hstack((X_train, X_train_ef))
                new_test = np.hstack((X_test, X_test_ef))

                dataset_results = get_algorithm_scores(X_train, y_train, X_test, y_test, new_train, new_test, d_name)
                all_results.extend(dataset_results)

        except Exception as e:
            print(f"Hata (ID: {set_id_val}): {e}")
            continue

    if not all_results:
        return None

    df_raw = pd.DataFrame(all_results)

    pivot_df = pd.pivot_table(
        df_raw,
        index='Algorithm',
        columns='Dataset',
        values=['Base_Score', 'Improved_Score', 'Difference']
    )
    pivot_df = pivot_df.swaplevel(0, 1, axis=1)

    desired_order = ['Base_Score', 'Improved_Score', 'Difference']

    unique_datasets = sorted(df_raw['Dataset'].unique())

    new_columns = pd.MultiIndex.from_product([unique_datasets, desired_order], names=['Dataset', 'Metric'])

    pivot_df = pivot_df.reindex(columns=new_columns)

    general_base = df_raw.groupby('Algorithm')['Base_Score'].mean()
    general_imp = df_raw.groupby('Algorithm')['Improved_Score'].mean()
    general_diff = df_raw.groupby('Algorithm')['Difference'].mean()

    pivot_df[('General', 'Base_Score')] = general_base
    pivot_df[('General', 'Improved_Score')] = general_imp
    pivot_df[('General', 'Difference')] = general_diff

    return pivot_df

def stgp_and_ef_fe(sets_id, threshold, min_importance_threshold):
    np.seterr(divide='ignore', invalid='ignore')
    all_results = []

    for set_id_val in sets_id:
        try:
            dataset = openml.datasets.get_dataset(set_id_val)
            d_name = f"{dataset.name}({set_id_val})"

            X, y, _, _ = dataset.get_data(dataset_format="dataframe", target=dataset.default_target_attribute)

            X = categorize_to_numeric(X)
            X, y = apply_z_score_standardization(X, y)

            X_vals = np.nan_to_num(X.values.astype(np.float64))
            y_vals = y.values if isinstance(y, pd.Series) else y
            y_vals = y_vals.astype(np.float64)

            X_train, X_test, y_train, y_test = train_test_split(X_vals, y_vals, test_size=0.2, random_state=42)

            if threshold:

                ef = EvolutionaryForestRegressor()
                with np.errstate(divide='ignore', invalid='ignore'):
                    ef.fit(X_train, y_train)
                    X_train_ef = ef.transform(X_train)
                    X_test_ef = ef.transform(X_test)
                    X_train_ef = np.nan_to_num(X_train_ef)
                    X_test_ef = np.nan_to_num(X_test_ef)

                feature_importance_dict = get_feature_importance(ef)

                X_train_top_ef, X_test_top_ef = select_ef_features_by_threshold(
                    feature_importance_dict=feature_importance_dict,
                    X_train_transformed=X_train_ef,
                    X_test_transformed=X_test_ef,
                    min_importance_threshold=min_importance_threshold,
                )

                stgp = SymbolicTransformer(parsimony_coefficient=0.005, n_components=5)
                with np.errstate(divide='ignore', invalid='ignore'):
                    stgp.fit(X_train, y_train)
                    X_train_top_st = stgp.transform(X_train)
                    X_test_top_st = stgp.transform(X_test)

                new_train = np.hstack((X_train, X_train_top_st, X_train_top_ef))
                new_test = np.hstack((X_test, X_test_top_st, X_test_top_ef))

                dataset_results = get_algorithm_scores(X_train, y_train, X_test, y_test, new_train, new_test, d_name)
                all_results.extend(dataset_results)

            else:

                ef = EvolutionaryForestRegressor()
                with np.errstate(divide='ignore', invalid='ignore'):
                    ef.fit(X_train, y_train)
                    X_train_ef = ef.transform(X_train)
                    X_test_ef = ef.transform(X_test)
                    X_train_ef = np.nan_to_num(X_train_ef)
                    X_test_ef = np.nan_to_num(X_test_ef)

                if X_train_ef.shape[1] > 10:
                    X_train_ef = X_train_ef[:, :10]
                    X_test_ef = X_test_ef[:, :10]

                stgp = SymbolicTransformer()
                with np.errstate(divide='ignore', invalid='ignore'):
                    stgp.fit(X_train, y_train)
                    X_train_st = stgp.transform(X_train)
                    X_test_st = stgp.transform(X_test)

                new_train = np.hstack((X_train, X_train_st, X_train_ef))
                new_test = np.hstack((X_test, X_test_st, X_test_ef))

                dataset_results = get_algorithm_scores(X_train, y_train, X_test, y_test, new_train, new_test, d_name)
                all_results.extend(dataset_results)

        except Exception as e:
            print(f"Hata (ID: {set_id_val}): {e}")
            continue

    if not all_results:
        return None

    df_raw = pd.DataFrame(all_results)

    pivot_df = pd.pivot_table(
        df_raw,
        index='Algorithm',
        columns='Dataset',
        values=['Base_Score', 'Improved_Score', 'Difference']
    )

    pivot_df = pivot_df.swaplevel(0, 1, axis=1)


    desired_order = ['Base_Score', 'Improved_Score', 'Difference']

    unique_datasets = sorted(df_raw['Dataset'].unique())

    new_columns = pd.MultiIndex.from_product([unique_datasets, desired_order], names=['Dataset', 'Metric'])

    pivot_df = pivot_df.reindex(columns=new_columns)

    general_base = df_raw.groupby('Algorithm')['Base_Score'].mean()
    general_imp = df_raw.groupby('Algorithm')['Improved_Score'].mean()
    general_diff = df_raw.groupby('Algorithm')['Difference'].mean()

    pivot_df[('General', 'Base_Score')] = general_base
    pivot_df[('General', 'Improved_Score')] = general_imp
    pivot_df[('General', 'Difference')] = general_diff

    return pivot_df