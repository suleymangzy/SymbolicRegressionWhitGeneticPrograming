import warnings
import openml
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.base import clone
from sklearn.preprocessing import StandardScaler
from gplearn.genetic import SymbolicTransformer
from evolutionary_forest.forest import EvolutionaryForestRegressor
from evolutionary_forest.utils import get_feature_importance

from Functions import (
    select_ef_features_by_threshold,
    categorize_to_numeric, select_st_features_by_threshold
)

warnings.filterwarnings("ignore")


def evaluate_models(X_train_new, X_test_new, y_train, y_test, dataset_name, method_name):
    """
    Verilen feature setleri ile modelleri eğitir ve sonuçları döndürür.
    """
    regressor_dict = {
        'RF': RandomForestRegressor(n_estimators=200, n_jobs=-1, random_state=42),
        'XGBoost': XGBRegressor(n_jobs=1, n_estimators=200, verbosity=0, random_state=42),
    }

    results = []

    for regr_name, model_instance in regressor_dict.items():
        try:
            # # 1. Base Model (Ham veri)
            # # Not: Base skoru her method için tekrar hesaplamak yerine dışarıda bir kez hesaplanabilir
            # # ancak karşılaştırma kolaylığı için buraya ekliyoruz.
            # base_model = clone(model_instance)
            # base_model.fit(X_train_base, y_train)
            # b_score = r2_score(y_test, base_model.predict(X_test_base))

            # 2. Enhanced Model (Yeni özelliklerle)
            enh_model = clone(model_instance)
            enh_model.fit(X_train_new, y_train)
            e_score = r2_score(y_test, enh_model.predict(X_test_new))

            # diff = e_score - b_score

            results.append({
                'Algorithm': regr_name,
                'Dataset': dataset_name,
                'Method': method_name,
                'Score': e_score  # Tabloda görünmesini istediğiniz değer (Improved Score)
                # 'Difference': diff # Eğer tabloda farkı görmek isterseniz bunu kullanın
            })

        except Exception as e:
            print(f"Hata ({regr_name} - {dataset_name}): {e}")
            results.append({
                'Algorithm': regr_name,
                'Dataset': dataset_name,
                'Method': method_name,
                'Score': np.nan
            })

    return results


def run_comparative_analysis(sets_id, use_threshold=True, min_importance_threshold_ef=0.05, min_importance_threshold_stgp=0.60):
    all_results = []

    for set_id_val in sets_id:
        try:
            print(f"Procesing: Dataset ID {set_id_val}...")
            dataset = openml.datasets.get_dataset(set_id_val)
            d_name = f"{dataset.name}"

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

            print(f'Raw {d_name} columns: {X_train.shape[1]}')

            ef = EvolutionaryForestRegressor(random_state=42)
            ef.fit(X_train, y_train)

            X_train_ef = ef.transform(X_train)
            X_test_ef = ef.transform(X_test)

            if use_threshold:
                limit = min(5, X_train_ef.shape[1])
                X_train_ef = X_train_ef[:, :limit]
                X_test_ef = X_test_ef[:, :limit]
            else:
                limit = min(10, X_train_ef.shape[1])
                X_train_ef = X_train_ef[:, :limit]
                X_test_ef = X_test_ef[:, :limit]

            new_train_ef = np.hstack((X_train, X_train_ef))
            new_test_ef = np.hstack((X_test, X_test_ef))

            print(f'EF {d_name} columns: {new_train_ef.shape[1]}')

            all_results.extend(evaluate_models(new_train_ef, new_test_ef,
                                               y_train, y_test, d_name, "EF"))


            stgp = SymbolicTransformer()
            stgp.fit(X_train, y_train)

            X_train_st = stgp.transform(X_train)
            X_test_st = stgp.transform(X_test)

            if use_threshold:
                limit = min(5, X_train_st.shape[1])
                X_train_st = X_train_st[:, :limit]
                X_test_st = X_test_st[:, :limit]

            new_train_stgp_ef = np.hstack((X_train, X_train_st, X_train_ef))
            new_test_stgp_ef = np.hstack((X_test, X_test_st, X_test_ef))

            print(f'STGP+EF {d_name} columns: {new_train_stgp_ef.shape[1]}')

            all_results.extend(evaluate_models(new_train_stgp_ef, new_test_stgp_ef,
                                               y_train, y_test, d_name, "STGP+EF"))

        except Exception as e:
            print(f"Genel Hata (ID: {set_id_val}): {e}")
            continue

    if not all_results:
        return None

    # --- TABLO OLUŞTURMA ---
    df_results = pd.DataFrame(all_results)

    # İstenilen format: Index=Algorithm, Columns=[Dataset, Method]
    pivot_df = pd.pivot_table(
        df_results,
        index='Algorithm',
        columns=['Dataset', 'Method'],
        values='Score'  # Buraya 'Difference' yazarsanız artış miktarını gösterir.
    )

    return pivot_df

# --- KULLANIM ÖRNEĞİ ---
# dataset_ids = [531, 537] # Örnek ID'ler
# final_table = run_comparative_analysis(dataset_ids, use_threshold=True)
# print(final_table)
