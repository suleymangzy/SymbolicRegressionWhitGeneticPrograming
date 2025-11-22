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

def categorize_to_numeric1(df: pd.DataFrame) -> pd.DataFrame:
    df_transformed = df.copy()

    non_numeric_cols = df_transformed.select_dtypes(include=['object', 'category']).columns.tolist()

    df_transformed = pd.get_dummies(df_transformed, columns=non_numeric_cols)

    return df_transformed


def apply_z_score_standardization1(X: pd.DataFrame, y: Union[pd.Series, pd.DataFrame]) -> Tuple[pd.DataFrame, pd.Series]:
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


warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
pd.set_option('display.float_format', '{:.4f}'.format)


def get_algorithm_scores(X_train, y_train, X_test, y_test, new_train, new_test, dataset_name):
    """
    Tek bir veri seti için skorları ve FARKI hesaplar.
    """
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
            # 1. Base Model
            base_model = clone(regressor_dict[regr_name])
            base_model.fit(X_train, y_train)
            b_score = r2_score(y_test, base_model.predict(X_test))

            # 2. Enhanced Model
            enh_model = clone(regressor_dict[regr_name])
            enh_model.fit(new_train, y_train)
            e_score = r2_score(y_test, enh_model.predict(new_test))

            # Fark Hesabı
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


def visualize_general_results(base_series, imp_series, diff_series):
    """
    General ortalamaları ve farkı grafiğe döker.
    """
    sns.set(style="whitegrid", font_scale=1.1)
    plot_df = pd.DataFrame({
        'Base': base_series,
        'Improved': imp_series,
        'Difference': diff_series
    }).sort_index()

    x = np.arange(len(plot_df))
    width = 0.35
    fig, ax = plt.subplots(figsize=(12, 6))

    # Base Bar (Mavi)
    ax.bar(x - width / 2, plot_df['Base'], width, label='General Mean Base', color='skyblue')

    # Improved Bar (Yeşil/Kırmızı)
    colors = np.where(plot_df['Difference'] > 0, 'mediumseagreen', 'salmon')
    ax.bar(x + width / 2, plot_df['Improved'], width, label='General Mean Improved', color=colors)

    ax.set_ylabel('Average $R^2$ Score')
    ax.set_title('General Performance Comparison (Difference Based)')
    ax.set_xticks(x)
    ax.set_xticklabels(plot_df.index, rotation=45)
    ax.legend()

    # Barların üzerine FARK değerini yazdır
    for i, v in enumerate(plot_df['Difference']):
        height = max(plot_df['Base'].iloc[i], plot_df['Improved'].iloc[i])
        color = 'green' if v > 0 else 'red'
        ax.text(i + width / 2, height + 0.02, f"{v:+.4f}",
                ha='center', va='bottom', color=color, fontweight='bold', fontsize=10)

    plt.tight_layout()
    plt.show()


def srgp_fe(sets_id):
    np.seterr(divide='ignore', invalid='ignore')
    all_results = []

    print(f"{'=' * 20} Analiz Başlıyor {'=' * 20}")

    for set_id_val in sets_id:
        try:
            dataset = openml.datasets.get_dataset(set_id_val)
            d_name = f"{dataset.name[:8]}..({set_id_val})"
            print(f">> İşleniyor: {d_name}")

            X, y, _, _ = dataset.get_data(dataset_format="dataframe", target=dataset.default_target_attribute)

            X = categorize_to_numeric1(X)
            X, y = apply_z_score_standardization1(X, y)

            X_vals = np.nan_to_num(X.values.astype(np.float64))
            y_vals = y.values if isinstance(y, pd.Series) else y
            y_vals = y_vals.astype(np.float64)

            X_train, X_test, y_train, y_test = train_test_split(X_vals, y_vals, test_size=0.2, random_state=42)

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

    # 1. PIVOT: Verileri satır/sütun formatına getir
    pivot_df = pd.pivot_table(
        df_raw,
        index='Algorithm',
        columns='Dataset',
        values=['Base_Score', 'Improved_Score', 'Difference']
    )

    # 2. SWAP LEVEL: (Dataset, Metric) hiyerarşisine çevir
    pivot_df = pivot_df.swaplevel(0, 1, axis=1)

    # --- DÜZELTME BURADA: ÖZEL SÜTUN SIRALAMASI ---
    # Alfabetik sıralama (sort_index) yapmıyoruz.
    # İstediğimiz sıra: [Base_Score, Improved_Score, Difference]

    desired_order = ['Base_Score', 'Improved_Score', 'Difference']

    # Mevcut veri setlerinin isimlerini alıp sıralayalım
    unique_datasets = sorted(df_raw['Dataset'].unique())

    # Pandas MultiIndex ile tüm datasetler için bu özel sırayı oluştur
    new_columns = pd.MultiIndex.from_product([unique_datasets, desired_order], names=['Dataset', 'Metric'])

    # Tabloyu bu yeni sıraya göre yeniden indeksle
    pivot_df = pivot_df.reindex(columns=new_columns)

    # 3. GENERAL SÜTUNLARI (Aynı sırayla ekle)
    general_base = df_raw.groupby('Algorithm')['Base_Score'].mean()
    general_imp = df_raw.groupby('Algorithm')['Improved_Score'].mean()
    general_diff = df_raw.groupby('Algorithm')['Difference'].mean()

    # General sütunlarını manuel olarak istediğimiz sırada ekliyoruz
    pivot_df[('General', 'Base_Score')] = general_base
    pivot_df[('General', 'Improved_Score')] = general_imp
    pivot_df[('General', 'Difference')] = general_diff

    print("\n" + "#" * 30 + " SONUÇ MATRİSİ (DÜZENLENMİŞ) " + "#" * 30)
    print(pivot_df)

    visualize_general_results(general_base, general_imp, general_diff)

    return pivot_df

def stgp_fe(sets_id):
    np.seterr(divide='ignore', invalid='ignore')
    all_results = []

    print(f"{'=' * 20} Analiz Başlıyor {'=' * 20}")

    for set_id_val in sets_id:
        try:
            dataset = openml.datasets.get_dataset(set_id_val)
            d_name = f"{dataset.name[:8]}..({set_id_val})"
            print(f">> İşleniyor: {d_name}")

            X, y, _, _ = dataset.get_data(dataset_format="dataframe", target=dataset.default_target_attribute)

            X = categorize_to_numeric1(X)
            X, y = apply_z_score_standardization1(X, y)

            X_vals = np.nan_to_num(X.values.astype(np.float64))
            y_vals = y.values if isinstance(y, pd.Series) else y
            y_vals = y_vals.astype(np.float64)

            X_train, X_test, y_train, y_test = train_test_split(X_vals, y_vals, test_size=0.2, random_state=42)

            stgp = SymbolicTransformer()
            with np.errstate(divide='ignore', invalid='ignore'):
                stgp.fit(X_train, y_train)

                # 2. Transform işlemi
                # Varsayılan olarak (n_samples, 10) boyutunda matris döner.
                X_train_st = stgp.transform(X_train)
                X_test_st = stgp.transform(X_test)

                # 3. NaN/Inf Temizliği
                X_train_st = np.nan_to_num(X_train_st)
                X_test_st = np.nan_to_num(X_test_st)

            # 4. Birleştirme (Stacking)
            # DİKKAT: .reshape(-1, 1) KALDIRILDI.
            # Çünkü çıktı zaten 2 boyutlu (Samples x 10) formatındadır.
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

    # 1. PIVOT: Verileri satır/sütun formatına getir
    pivot_df = pd.pivot_table(
        df_raw,
        index='Algorithm',
        columns='Dataset',
        values=['Base_Score', 'Improved_Score', 'Difference']
    )

    # 2. SWAP LEVEL: (Dataset, Metric) hiyerarşisine çevir
    pivot_df = pivot_df.swaplevel(0, 1, axis=1)

    # --- DÜZELTME BURADA: ÖZEL SÜTUN SIRALAMASI ---
    # Alfabetik sıralama (sort_index) yapmıyoruz.
    # İstediğimiz sıra: [Base_Score, Improved_Score, Difference]

    desired_order = ['Base_Score', 'Improved_Score', 'Difference']

    # Mevcut veri setlerinin isimlerini alıp sıralayalım
    unique_datasets = sorted(df_raw['Dataset'].unique())

    # Pandas MultiIndex ile tüm datasetler için bu özel sırayı oluştur
    new_columns = pd.MultiIndex.from_product([unique_datasets, desired_order], names=['Dataset', 'Metric'])

    # Tabloyu bu yeni sıraya göre yeniden indeksle
    pivot_df = pivot_df.reindex(columns=new_columns)

    # 3. GENERAL SÜTUNLARI (Aynı sırayla ekle)
    general_base = df_raw.groupby('Algorithm')['Base_Score'].mean()
    general_imp = df_raw.groupby('Algorithm')['Improved_Score'].mean()
    general_diff = df_raw.groupby('Algorithm')['Difference'].mean()

    # General sütunlarını manuel olarak istediğimiz sırada ekliyoruz
    pivot_df[('General', 'Base_Score')] = general_base
    pivot_df[('General', 'Improved_Score')] = general_imp
    pivot_df[('General', 'Difference')] = general_diff

    print("\n" + "#" * 30 + " SONUÇ MATRİSİ (DÜZENLENMİŞ) " + "#" * 30)
    print(pivot_df)

    visualize_general_results(general_base, general_imp, general_diff)

    return pivot_df