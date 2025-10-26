import openml
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Union
from sklearn.preprocessing import StandardScaler
from gplearn.genetic import SymbolicTransformer, SymbolicRegressor
from evolutionary_forest.forest import EvolutionaryForestRegressor
from evolutionary_forest.utils import get_feature_importance
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score


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


def select_top_features(feature_importance_dict: Dict[str, float], X_train_transformed: np.ndarray,
                        X_test_transformed: np.ndarray, features_size: int) -> Tuple[np.ndarray, np.ndarray]:

    sorted_features: List[Tuple[str, float]] = sorted(feature_importance_dict.items(), key=lambda x: x[1], reverse=True)

    top_n_features: List[Tuple[str, float]] = sorted_features[:features_size]

    feature_names_list = list(feature_importance_dict.keys())

    top_indices: List[int] = [feature_names_list.index(f[0]) for f in top_n_features]

    X_train_top: np.ndarray = X_train_transformed[:, top_indices]
    X_test_top: np.ndarray = X_test_transformed[:, top_indices]

    return X_train_top, X_test_top


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


def comparison_whit_gp_transgormer(sets_id: list, orginal_features: bool) -> pd.DataFrame:
    results_list = []

    for set_id_val in sets_id:
        dataset = openml.datasets.get_dataset(set_id_val)

        X, y, _, _ = dataset.get_data(dataset_format="dataframe", target=dataset.default_target_attribute)

        X = categorize_to_numeric(X)
        X, y = apply_z_score_standardization(X, y)

        X_train, X_test, y_train, y_test = train_test_split(X.values, y.values if isinstance(y, pd.Series) else y, test_size=0.2, random_state=42)

        st = SymbolicTransformer()
        st.fit(X_train, y_train)

        X_train_st = st.transform(X_train)
        X_test_st = st.transform(X_test)

        if orginal_features:
            X_train_new = np.hstack((X_train, X_train_st))
            X_test_new = np.hstack((X_test, X_test_st))
        else:
            X_train_new = X_train_st
            X_test_new = X_test_st

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


def feature_extraction_and_model_training(set_id: list, aos: bool, orginal_features: bool) -> pd.DataFrame:
    results_list = []

    for set_id_val in set_id:

        dataset = openml.datasets.get_dataset(set_id_val)

        X, y, _, _ = dataset.get_data(dataset_format="dataframe", target=dataset.default_target_attribute)

        features_size = X.shape[1]

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

        feature_importance_dict = get_feature_importance(r)

        X_train_top, X_test_top = select_top_features(
            feature_importance_dict=feature_importance_dict,
            X_train_transformed=X_train,
            X_test_transformed=X_test,
            features_size=features_size
        )

        st = SymbolicTransformer()

        st.fit(X_train, y_train)

        X_train_st = st.transform(X_train)
        X_test_st = st.transform(X_test)

        if orginal_features:

            new_train = np.hstack((X_train, X_train_top, X_train_st))
            new_test = np.hstack((X_test, X_test_top, X_test_st))
        else:
            new_train = np.hstack((X_train_top, X_train_st))
            new_test = np.hstack((X_test_top, X_test_st))

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
