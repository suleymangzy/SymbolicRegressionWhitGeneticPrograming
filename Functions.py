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


def categorize_to_numeric(df: pd.DataFrame) -> pd.DataFrame:
    df_transformed = df.copy()

    non_numeric_cols = df_transformed.select_dtypes(include=['object', 'category']).columns.tolist()

    df_transformed = pd.get_dummies(df_transformed, columns=non_numeric_cols)

    return df_transformed


def apply_z_score_standardization(X: pd.DataFrame, y: Union[pd.Series, pd.DataFrame]) -> Tuple[pd.DataFrame, pd.Series]:
    X_transformed = X.copy()
    y_transformed = y.copy()

    numeric_cols = X_transformed.select_dtypes(include=np.number).columns.tolist()

    scaler_X = StandardScaler()
    X_transformed[numeric_cols] = scaler_X.fit_transform(X_transformed[numeric_cols])

    if isinstance(y_transformed, pd.Series):
        y_array = y_transformed.values.reshape(-1, 1)
        scaler_y = StandardScaler()
        y_scaled = scaler_y.fit_transform(y_array)

        y_transformed = pd.Series(y_scaled.flatten(), index=y_transformed.index, name=y_transformed.name)

    return X_transformed, y_transformed


def select_ef_features_by_threshold(feature_importance_dict: Dict[str, float], X_train_transformed: np.ndarray,
                                    X_test_transformed: np.ndarray, min_importance_threshold: float
                                    ) -> Tuple[np.ndarray, np.ndarray]:
    sorted_features = sorted(feature_importance_dict.items(), key=lambda x: x[1], reverse=True)

    selected = [(name, imp) for name, imp in sorted_features if imp >= min_importance_threshold]

    if not selected:
        print(f"Warning: No features found with importance ≥ {min_importance_threshold} in EF.")
        selected = sorted_features[:3]
        print(f"Selecting top 3 features instead: {[name for name, _ in selected]}")

    all_feature_names = list(feature_importance_dict.keys())
    selected_indices = [all_feature_names.index(name) for name, _ in selected]

    X_train_selected = X_train_transformed[:, selected_indices]
    X_test_selected = X_test_transformed[:, selected_indices]

    print(f"Selected {len(selected_indices)} out of {len(all_feature_names)} EF features.")

    return X_train_selected, X_test_selected


def select_st_features_by_threshold(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, st_train: np.ndarray,
                                    st_test: np.ndarray, min_importance_threshold: float) -> Tuple[
    np.ndarray, np.ndarray]:
    X_train_transformed = np.hstack([X_train, st_train])
    X_test_transformed = np.hstack([X_test, st_test])

    rf = RandomForestRegressor(n_estimators=200, random_state=42)
    rf.fit(X_train_transformed, y_train)

    importances = rf.feature_importances_

    orig_names = [f"X{i}" for i in range(X_train.shape[1])]
    st_names = [f"ST_feat_{i}" for i in range(st_train.shape[1])]
    all_feature_names = orig_names + st_names

    feature_importance_dict = dict(zip(all_feature_names, importances))

    sorted_features = sorted(feature_importance_dict.items(), key=lambda x: x[1], reverse=True)

    selected = [(name, imp) for name, imp in sorted_features if imp >= min_importance_threshold]

    if not selected:
        print(f"Warning: No features found with importance ≥ {min_importance_threshold} in EF.")
        selected = sorted_features[:3]
        print(f"Selecting top 3 features instead: {[name for name, _ in selected]}")

    all_feature_names = list(feature_importance_dict.keys())
    selected_indices = [all_feature_names.index(name) for name, _ in selected]

    X_train_selected = X_train_transformed[:, selected_indices]
    X_test_selected = X_test_transformed[:, selected_indices]

    print(f"Selected {len(selected_indices)} out of {len(all_feature_names)}  ST features.")

    return X_train_selected, X_test_selected


def comparison_ef_srgp(sets_id: list) -> pd.DataFrame:
    results_list = []

    for set_id_val in sets_id:
        dataset = openml.datasets.get_dataset(set_id_val)
        X, y, _, _ = dataset.get_data(dataset_format="dataframe", target=dataset.default_target_attribute)

        X = categorize_to_numeric(X)
        X, y = apply_z_score_standardization(X, y)

        X_train, X_test, y_train, y_test = train_test_split(
            X.values,
            y.values if isinstance(y, pd.Series) else y,
            test_size=0.2,
            random_state=42
        )

        r = EvolutionaryForestRegressor()
        r.fit(X_train, y_train)

        y_train_pred_ef = r.predict(X_train)
        y_test_pred_ef = r.predict(X_test)

        est_gp = SymbolicRegressor()
        est_gp.fit(X_train, y_train)

        y_train_pred_est_gp = est_gp.predict(X_train)
        y_test_pred_est_gp = est_gp.predict(X_test)

        ef_train_r2 = r2_score(y_train, y_train_pred_ef)
        ef_test_r2 = r2_score(y_test, y_test_pred_ef)

        est_gp_train_r2 = r2_score(y_train, y_train_pred_est_gp)
        est_gp_test_r2 = r2_score(y_test, y_test_pred_est_gp)

        results_list.append({
            'Set_ID': set_id_val,
            'Dataset_Name': dataset.name,
            'EF_Train_R2_Score': ef_train_r2,
            'EF_Test_R2_Score': ef_test_r2,
            'EST_GP_Train_R2_Score': est_gp_train_r2,
            'EST_GP_Test_R2_Score': est_gp_test_r2,
        })

    results_df = pd.DataFrame(results_list)

    results_df_sorted = results_df.sort_values(by='Set_ID', ascending=False)

    print(results_df_sorted.to_markdown(index=False, floatfmt=".4f", tablefmt="github"))

    return results_df_sorted


def comparison_whit_gp_transgormer(sets_id: list, min_importance_threshold: float,
                                   orginal_features: bool) -> pd.DataFrame:
    results_list = []

    for set_id_val in sets_id:
        dataset = openml.datasets.get_dataset(set_id_val)

        X, y, _, _ = dataset.get_data(dataset_format="dataframe", target=dataset.default_target_attribute)

        X = categorize_to_numeric(X)
        X, y = apply_z_score_standardization(X, y)

        X_train, X_test, y_train, y_test = train_test_split(X.values, y.values if isinstance(y, pd.Series) else y,
                                                            test_size=0.2, random_state=42)

        st = SymbolicTransformer(
            generations=20,
            population_size=2000,
            hall_of_fame=100,
            n_components=50,
            random_state=42
        )

        st.fit(X_train, y_train)

        X_train_st = st.transform(X_train)
        X_test_st = st.transform(X_test)

        X_train_top_st, X_test_top_st = select_st_features_by_threshold(
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            st_train=X_train_st,
            st_test=X_test_st,
            min_importance_threshold=min_importance_threshold
        )

        if orginal_features:
            X_train_new = np.hstack((X_train, X_train_top_st))
            X_test_new = np.hstack((X_test, X_test_top_st))
        else:
            X_train_new = X_train_top_st
            X_test_new = X_test_top_st

        r = EvolutionaryForestRegressor()
        r.fit(X_train_new, y_train)

        y_train_pred_ef = r.predict(X_train_new)
        y_test_pred_ef = r.predict(X_test_new)

        est_gp = SymbolicRegressor()
        est_gp.fit(X_train_new, y_train)

        y_train_pred_est_gp = est_gp.predict(X_train_new)
        y_test_pred_est_gp = est_gp.predict(X_test_new)

        ef_train_r2 = r2_score(y_train, y_train_pred_ef)
        ef_test_r2 = r2_score(y_test, y_test_pred_ef)

        est_gp_train_r2 = r2_score(y_train, y_train_pred_est_gp)
        est_gp_test_r2 = r2_score(y_test, y_test_pred_est_gp)

        results_list.append({
            'Set_ID': set_id_val,
            'Dataset_Name': dataset.name,
            'EF_Train_R2_Score': ef_train_r2,
            'EF_Test_R2_Score': ef_test_r2,
            'EF_GP_Train_R2_Score': est_gp_train_r2,
            'EF_GP_Test_R2_Score': est_gp_test_r2,
        })

    results_df = pd.DataFrame(results_list)

    results_df_sorted = results_df.sort_values(by='Set_ID', ascending=False)

    print(results_df_sorted.to_markdown(index=False, floatfmt=".4f", tablefmt="github"))

    return results_df_sorted


def feature_extraction_and_model_training(set_id: list, min_importance_threshold: float, aos: bool,
                                          orginal_features: bool) -> pd.DataFrame:
    results_list = []

    for set_id_val in set_id:

        dataset = openml.datasets.get_dataset(set_id_val)

        X, y, _, _ = dataset.get_data(dataset_format="dataframe", target=dataset.default_target_attribute)

        X = categorize_to_numeric(X)

        X, y = apply_z_score_standardization(X, y)

        X_train, X_test, y_train, y_test = train_test_split(
            X.values,
            y.values if isinstance(y, pd.Series) else y,
            test_size=0.2,
            random_state=42
        )

        if aos:
            r = EvolutionaryForestRegressor(
                max_height=5,
                normalize=True,
                select='AutomaticLexicase',
                gene_num=10,
                boost_size=100,
                n_gen=20,
                n_pop=200,
                cross_pb=1,
                base_learner='Random-DT',
                verbose=False,
                n_process=1
            )

            r.fit(X_train, y_train)
        else:
            r = EvolutionaryForestRegressor()
            r.fit(X_train, y_train)

        X_train_ef = r.transform(X_train)
        X_test_ef = r.transform(X_test)

        feature_importance_dict = get_feature_importance(r)

        X_train_top_ef, X_test_top_ef = select_ef_features_by_threshold(
            feature_importance_dict=feature_importance_dict,
            X_train_transformed=X_train_ef,
            X_test_transformed=X_test_ef,
            min_importance_threshold=min_importance_threshold,
        )

        st = SymbolicTransformer(
            generations=20,
            population_size=2000,
            hall_of_fame=100,
            n_components=50,
            random_state=42
        )

        st.fit(X_train, y_train)

        X_train_st = st.transform(X_train)
        X_test_st = st.transform(X_test)

        X_train_top_st, X_test_top_st = select_st_features_by_threshold(
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            st_train=X_train_st,
            st_test=X_test_st,
            min_importance_threshold=min_importance_threshold
        )

        if orginal_features:

            new_train = np.hstack((X_train, X_train_top_ef, X_train_top_st))
            new_test = np.hstack((X_test, X_test_top_ef, X_test_top_st))
        else:
            new_train = np.hstack((X_train_top_ef, X_train_top_st))
            new_test = np.hstack((X_test_top_ef, X_test_top_st))

        est_gp = SymbolicRegressor()

        est_gp.fit(new_train, y_train)

        y_train_pred = est_gp.predict(new_train)
        y_test_pred = est_gp.predict(new_test)

        train_r2 = r2_score(y_train, y_train_pred)
        test_r2 = r2_score(y_test, y_test_pred)

        results_list.append({
            'Set_ID': set_id_val,
            'Dataset_Name': dataset.name,
            'Train_R2_Score': train_r2,
            'Test_R2_Score': test_r2
        })

    results_df = pd.DataFrame(results_list)

    results_df_sorted = results_df.sort_values(by='Set_ID', ascending=False)

    print(results_df_sorted.to_markdown(index=False, floatfmt=".4f", tablefmt="github"))

    return results_df_sorted


def feature_extraction_ef(sets_id: list) -> pd.DataFrame:
    results_list = []

    for set_id_val in sets_id:
        dataset = openml.datasets.get_dataset(set_id_val)

        X, y, _, _ = dataset.get_data(dataset_format="dataframe", target=dataset.default_target_attribute)

        X = categorize_to_numeric(X)

        X, y = apply_z_score_standardization(X, y)

        X_train, X_test, y_train, y_test = train_test_split(X.values, y.values if isinstance(y, pd.Series) else y,
                                                            test_size=0.2,
                                                            random_state=42
                                                            )

        srgp = SymbolicRegressor()
        srgp.fit(X_train, y_train)

        srgp_train_r2 = r2_score(y_train, srgp.predict(X_train))
        srgp_test_r2 = r2_score(y_test, srgp.predict(X_test))

        ef = EvolutionaryForestRegressor()
        ef.fit(X_train, y_train)

        X_train_ef = ef.transform(X_train)
        X_test_ef = ef.transform(X_test)

        k_features_to_select = X_train.shape[1]

        if k_features_to_select > X_train_ef.shape[1]:

            size = k_features_to_select - X_train_ef.shape[1]
            X_train_selected = np.hstack((X_train[:, :size], X_train_ef))
            X_test_selected = np.hstack((X_test[:, :size], X_test_ef))

        else:
            n_ef_features = X_train_ef.shape[1]

            feature_importance_dict = get_feature_importance(ef)
            all_feature_names = list(feature_importance_dict.keys())

            sorted_indices = sorted(
                range(len(all_feature_names)),
                key=lambda k: feature_importance_dict[all_feature_names[k]],
                reverse=True
            )

            valid_indices = [i for i in sorted_indices if i < n_ef_features]
            selected_indices = valid_indices[:k_features_to_select]

            X_train_selected = X_train_ef[:, selected_indices]
            X_test_selected = X_test_ef[:, selected_indices]

        srgp.fit(X_train_selected, y_train)

        srgp_ef_train_r2 = r2_score(y_train, srgp.predict(X_train_selected))
        srgp_ef_test_r2 = r2_score(y_test, srgp.predict(X_test_selected))

        results_list.append({
            'Set_ID': set_id_val,
            'Dataset_Name': dataset.name,
            'SRGP_Train_R2_Score': srgp_train_r2,
            'SRGP_Test_R2_Score': srgp_test_r2,
            'SRGP_EF_Train_R2_Score': srgp_ef_train_r2,
            'SRGP_EF_Test_R2_Score': srgp_ef_test_r2,
        })

    results_df = pd.DataFrame(results_list)

    results_df_sorted = results_df.sort_values(by='Set_ID', ascending=False)

    print(results_df_sorted.to_markdown(index=False, floatfmt=".4f", tablefmt="github"))

    return results_df_sorted





