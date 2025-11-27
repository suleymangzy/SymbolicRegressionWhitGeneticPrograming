import warnings
import openml
import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, AdaBoostRegressor, GradientBoostingRegressor
from gplearn.genetic import SymbolicTransformer, SymbolicRegressor
from evolutionary_forest.forest import EvolutionaryForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.base import clone
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from Functions.Comparison import categorize_to_numeric

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
pd.set_option('display.float_format', '{:.4f}'.format)


def get_algorithm_scores(X_train, y_train, X_test, y_test, new_train, new_test, dataset_name):
    regressor_list = ['RF', 'ET', 'AdaBoost', 'GBDT', 'DART', 'XGBoost', 'LightGBM', 'CatBoost']

    regressor_dict = {
        'RF': RandomForestRegressor(n_estimators=200, n_jobs=-1),
        'ET': ExtraTreesRegressor(n_estimators=200, n_jobs=-1),
        'AdaBoost': AdaBoostRegressor(n_estimators=200),
        'GBDT': GradientBoostingRegressor(n_estimators=200),
        'DART': LGBMRegressor(n_jobs=1, n_estimators=200, boosting_type='dart', xgboost_dart_mode=True, verbose=-1),
        'XGBoost': XGBRegressor(n_jobs=1, n_estimators=200, verbosity=0),
        'LightGBM': LGBMRegressor(n_jobs=1, n_estimators=200, verbose=-1),
        'CatBoost': CatBoostRegressor(n_estimators=200, thread_count=1, verbose=False, allow_writing_files=False),
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

def srgp_fe(sets_id):
    np.seterr(divide='ignore', invalid='ignore')
    all_results = []

    for set_id_val in sets_id:
        try:
            dataset = openml.datasets.get_dataset(set_id_val)
            d_name = f"{dataset.name}({set_id_val})"

            X, y, _, _ = dataset.get_data(dataset_format="dataframe", target=dataset.default_target_attribute)

            X = categorize_to_numeric(X)

            X_vals = np.nan_to_num(X.values.astype(np.float64))
            y_vals = y.values if isinstance(y, pd.Series) else y
            y_vals = y_vals.astype(np.float64)

            X_train, X_test, y_train, y_test = train_test_split(X_vals, y_vals, test_size=0.2, random_state=42)

            scaler_x = StandardScaler()
            X_train = scaler_x.fit_transform(X_train)
            X_test = scaler_x.transform(X_test)

            scaler_y = StandardScaler()
            y_train = scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()
            y_test = scaler_y.transform(y_test.reshape(-1, 1)).ravel()

            srgp = SymbolicRegressor()
            with np.errstate(divide='ignore', invalid='ignore'):
                srgp.fit(X_train, y_train)
                expr = srgp._program
                X_train_sr = np.nan_to_num(expr.execute(X_train).reshape(-1, 1))
                X_test_sr = np.nan_to_num(expr.execute(X_test).reshape(-1, 1))

            new_train = np.hstack((X_train, X_train_sr))
            new_test = np.hstack((X_test, X_test_sr))

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

def stgp_fe(sets_id):
    np.seterr(divide='ignore', invalid='ignore')
    all_results = []

    for set_id_val in sets_id:
        try:
            dataset = openml.datasets.get_dataset(set_id_val)
            d_name = f"{dataset.name}({set_id_val})"

            X, y, _, _ = dataset.get_data(dataset_format="dataframe", target=dataset.default_target_attribute)

            X = categorize_to_numeric(X)

            X_vals = np.nan_to_num(X.values.astype(np.float64))
            y_vals = y.values if isinstance(y, pd.Series) else y
            y_vals = y_vals.astype(np.float64)

            X_train, X_test, y_train, y_test = train_test_split(X_vals, y_vals, test_size=0.2, random_state=42)

            scaler_x = StandardScaler()
            X_train = scaler_x.fit_transform(X_train)
            X_test = scaler_x.transform(X_test)

            scaler_y = StandardScaler()
            y_train = scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()
            y_test = scaler_y.transform(y_test.reshape(-1, 1)).ravel()

            stgp = SymbolicTransformer()
            with np.errstate(divide='ignore', invalid='ignore'):
                stgp.fit(X_train, y_train)

                X_train_st = stgp.transform(X_train)
                X_test_st = stgp.transform(X_test)

                X_train_st = np.nan_to_num(X_train_st)
                X_test_st = np.nan_to_num(X_test_st)

            new_train = np.hstack((X_train, X_train_st))
            new_test = np.hstack((X_test, X_test_st))
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

def ef_fe(sets_id):
    np.seterr(divide='ignore', invalid='ignore')
    all_results = []

    for set_id_val in sets_id:
        try:
            dataset = openml.datasets.get_dataset(set_id_val)
            d_name = f"{dataset.name}({set_id_val})"

            X, y, _, _ = dataset.get_data(dataset_format="dataframe", target=dataset.default_target_attribute)

            X = categorize_to_numeric(X)

            X_vals = np.nan_to_num(X.values.astype(np.float64))
            y_vals = y.values if isinstance(y, pd.Series) else y
            y_vals = y_vals.astype(np.float64)

            X_train, X_test, y_train, y_test = train_test_split(X_vals, y_vals, test_size=0.2, random_state=42)

            scaler_x = StandardScaler()
            X_train = scaler_x.fit_transform(X_train)
            X_test = scaler_x.transform(X_test)

            scaler_y = StandardScaler()
            y_train = scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()
            y_test = scaler_y.transform(y_test.reshape(-1, 1)).ravel()

            ef = EvolutionaryForestRegressor()
            with np.errstate(divide='ignore', invalid='ignore'):
                ef.fit(X_train, y_train)
                X_train_ef = ef.transform(X_train)
                X_test_ef = ef.transform(X_test)

                if X_train_ef.shape[1] > 10:
                    X_train_ef = X_train_ef[:, :10]
                    X_test_ef = X_test_ef[:, :10]

                X_train_ef = np.nan_to_num(X_train_ef)
                X_test_ef = np.nan_to_num(X_test_ef)

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

def srgp_and_stgp_fe(sets_id):
    np.seterr(divide='ignore', invalid='ignore')
    all_results = []

    for set_id_val in sets_id:
        try:
            dataset = openml.datasets.get_dataset(set_id_val)
            d_name = f"{dataset.name}({set_id_val})"

            X, y, _, _ = dataset.get_data(dataset_format="dataframe", target=dataset.default_target_attribute)

            X = categorize_to_numeric(X)

            X_vals = np.nan_to_num(X.values.astype(np.float64))
            y_vals = y.values if isinstance(y, pd.Series) else y
            y_vals = y_vals.astype(np.float64)

            X_train, X_test, y_train, y_test = train_test_split(X_vals, y_vals, test_size=0.2, random_state=42)

            scaler_x = StandardScaler()
            X_train = scaler_x.fit_transform(X_train)
            X_test = scaler_x.transform(X_test)

            scaler_y = StandardScaler()
            y_train = scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()
            y_test = scaler_y.transform(y_test.reshape(-1, 1)).ravel()

            srgp = SymbolicRegressor()
            with np.errstate(divide='ignore', invalid='ignore'):
                srgp.fit(X_train, y_train)
                expr = srgp._program
                X_train_sr = np.nan_to_num(expr.execute(X_train).reshape(-1, 1))
                X_test_sr = np.nan_to_num(expr.execute(X_test).reshape(-1, 1))

            stgp = SymbolicTransformer()
            with np.errstate(divide='ignore', invalid='ignore'):
                stgp.fit(X_train, y_train)
                X_train_st = stgp.transform(X_train)
                X_test_st = stgp.transform(X_test)
                X_train_st = np.nan_to_num(X_train_st)
                X_test_st = np.nan_to_num(X_test_st)

            new_train = np.hstack((X_train, X_train_sr, X_train_st))
            new_test = np.hstack((X_test, X_test_sr, X_test_st))

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

def srgp_and_ef_fe(sets_id):
    np.seterr(divide='ignore', invalid='ignore')
    all_results = []

    for set_id_val in sets_id:
        try:
            dataset = openml.datasets.get_dataset(set_id_val)
            d_name = f"{dataset.name}({set_id_val})"

            X, y, _, _ = dataset.get_data(dataset_format="dataframe", target=dataset.default_target_attribute)

            X = categorize_to_numeric(X)

            X_vals = np.nan_to_num(X.values.astype(np.float64))
            y_vals = y.values if isinstance(y, pd.Series) else y
            y_vals = y_vals.astype(np.float64)

            X_train, X_test, y_train, y_test = train_test_split(X_vals, y_vals, test_size=0.2, random_state=42)

            scaler_x = StandardScaler()
            X_train = scaler_x.fit_transform(X_train)
            X_test = scaler_x.transform(X_test)

            scaler_y = StandardScaler()
            y_train = scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()
            y_test = scaler_y.transform(y_test.reshape(-1, 1)).ravel()

            srgp = SymbolicRegressor()
            with np.errstate(divide='ignore', invalid='ignore'):
                srgp.fit(X_train, y_train)
                expr = srgp._program
                X_train_sr = np.nan_to_num(expr.execute(X_train).reshape(-1, 1))
                X_test_sr = np.nan_to_num(expr.execute(X_test).reshape(-1, 1))

            ef = EvolutionaryForestRegressor()
            with np.errstate(divide='ignore', invalid='ignore'):
                ef.fit(X_train, y_train)

                X_train_ef = ef.transform(X_train)
                X_test_ef = ef.transform(X_test)

                if X_train_ef.shape[1] > 10:
                    X_train_ef = X_train_ef[:, :10]
                    X_test_ef = X_test_ef[:, :10]

                X_train_ef = np.nan_to_num(X_train_ef)
                X_test_ef = np.nan_to_num(X_test_ef)

            new_train = np.hstack((X_train, X_train_sr, X_train_ef))
            new_test = np.hstack((X_test, X_test_sr, X_test_ef))

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

def stgp_and_ef_fe(sets_id):
    np.seterr(divide='ignore', invalid='ignore')
    all_results = []

    for set_id_val in sets_id:
        try:
            dataset = openml.datasets.get_dataset(set_id_val)
            d_name = f"{dataset.name}({set_id_val})"

            X, y, _, _ = dataset.get_data(dataset_format="dataframe", target=dataset.default_target_attribute)

            X = categorize_to_numeric(X)

            X_vals = np.nan_to_num(X.values.astype(np.float64))
            y_vals = y.values if isinstance(y, pd.Series) else y
            y_vals = y_vals.astype(np.float64)

            X_train, X_test, y_train, y_test = train_test_split(X_vals, y_vals, test_size=0.2, random_state=42)

            scaler_x = StandardScaler()
            X_train = scaler_x.fit_transform(X_train)
            X_test = scaler_x.transform(X_test)

            scaler_y = StandardScaler()
            y_train = scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()
            y_test = scaler_y.transform(y_test.reshape(-1, 1)).ravel()

            stgp = SymbolicTransformer()
            with np.errstate(divide='ignore', invalid='ignore'):
                stgp.fit(X_train, y_train)

                X_train_st = stgp.transform(X_train)
                X_test_st = stgp.transform(X_test)

                X_train_st = np.nan_to_num(X_train_st)
                X_test_st = np.nan_to_num(X_test_st)

            ef = EvolutionaryForestRegressor()
            with np.errstate(divide='ignore', invalid='ignore'):
                ef.fit(X_train, y_train)

                X_train_ef = ef.transform(X_train)
                X_test_ef = ef.transform(X_test)

                if X_train_ef.shape[1] > 10:
                    X_train_ef = X_train_ef[:, :10]
                    X_test_ef = X_test_ef[:, :10]

                X_train_ef = np.nan_to_num(X_train_ef)
                X_test_ef = np.nan_to_num(X_test_ef)

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

def srgp_and_stgp_and_ef_fe(sets_id):
    np.seterr(divide='ignore', invalid='ignore')
    all_results = []

    for set_id_val in sets_id:
        try:
            dataset = openml.datasets.get_dataset(set_id_val)
            d_name = f"{dataset.name}({set_id_val})"

            X, y, _, _ = dataset.get_data(dataset_format="dataframe", target=dataset.default_target_attribute)

            X = categorize_to_numeric(X)

            X_vals = np.nan_to_num(X.values.astype(np.float64))
            y_vals = y.values if isinstance(y, pd.Series) else y
            y_vals = y_vals.astype(np.float64)

            X_train, X_test, y_train, y_test = train_test_split(X_vals, y_vals, test_size=0.2, random_state=42)

            scaler_x = StandardScaler()
            X_train = scaler_x.fit_transform(X_train)
            X_test = scaler_x.transform(X_test)

            scaler_y = StandardScaler()
            y_train = scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()
            y_test = scaler_y.transform(y_test.reshape(-1, 1)).ravel()

            srgp = SymbolicRegressor()
            with np.errstate(divide='ignore', invalid='ignore'):
                srgp.fit(X_train, y_train)
                expr = srgp._program
                X_train_sr = np.nan_to_num(expr.execute(X_train).reshape(-1, 1))
                X_test_sr = np.nan_to_num(expr.execute(X_test).reshape(-1, 1))

            stgp = SymbolicTransformer()
            with np.errstate(divide='ignore', invalid='ignore'):
                stgp.fit(X_train, y_train)

                X_train_st = stgp.transform(X_train)
                X_test_st = stgp.transform(X_test)

                X_train_st = np.nan_to_num(X_train_st)
                X_test_st = np.nan_to_num(X_test_st)

            ef = EvolutionaryForestRegressor()
            with np.errstate(divide='ignore', invalid='ignore'):
                ef.fit(X_train, y_train)

                X_train_ef = ef.transform(X_train)
                X_test_ef = ef.transform(X_test)

                if X_train_ef.shape[1] > 10:
                    X_train_ef = X_train_ef[:, :10]
                    X_test_ef = X_test_ef[:, :10]

                X_train_ef = np.nan_to_num(X_train_ef)
                X_test_ef = np.nan_to_num(X_test_ef)

            new_train = np.hstack((X_train, X_train_sr, X_train_st, X_train_ef))
            new_test = np.hstack((X_test, X_test_sr, X_test_st, X_test_ef))

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