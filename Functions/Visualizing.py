import warnings
import openml
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.base import clone
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor, AdaBoostRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from gplearn.genetic import SymbolicTransformer
from evolutionary_forest.forest import EvolutionaryForestRegressor

# Dummy function check
try:
    from Functions.Comparison import categorize_to_numeric
except ImportError:
    def categorize_to_numeric(df):
        return df.select_dtypes(include=[np.number])

warnings.filterwarnings('ignore')


def evaluate_single_model(X_train, X_test, y_train, y_test, model_name, model_instance, dataset_name, method_name):
    try:
        model = clone(model_instance)
        if model_name == 'CatBoost':
            model = CatBoostRegressor(n_estimators=200, thread_count=1, verbose=False, allow_writing_files=False)

        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        score = r2_score(y_test, preds)
        return {
            'Algorithm': model_name,
            'Dataset': dataset_name,
            'Method': method_name,
            'Score': score
        }
    except Exception as e:
        return None


def run_pipeline_and_visualize(sets_id):
    """
    Veri setlerini işler, modelleri eğitir ve Transformer, EF, Hybrid karşılaştırmalı
    geniş aralıklı grafikler oluşturur.
    """
    warnings.filterwarnings('ignore')

    all_results = []
    dataset_stats = []

    col_names = {
        'base': '0. Base',
        'st': '2. Transformer',
        'ef': '3. EF',
        'hy': '4. Hybrid'
    }

    judges = {
        'RF': RandomForestRegressor(n_estimators=100, n_jobs=-1, random_state=42),
        'XGBoost': XGBRegressor(n_jobs=1, n_estimators=100, verbosity=0, random_state=42),
        'ET': ExtraTreesRegressor(n_estimators=100, n_jobs=-1, random_state=42),
        'AdaBoost': AdaBoostRegressor(n_estimators=100, random_state=42),
        'GBDT': GradientBoostingRegressor(n_estimators=100, random_state=42),
        'DART': LGBMRegressor(n_jobs=1, n_estimators=200, boosting_type='dart', xgboost_dart_mode=True, verbose=-1),
        'LightGBM': LGBMRegressor(n_jobs=1, n_estimators=100, verbose=-1, random_state=42),
        'CatBoost': CatBoostRegressor(n_estimators=100, thread_count=1, verbose=False, allow_writing_files=False,
                                      random_state=42),
    }

    for set_id_val in sets_id:
        print(f"\n{'=' * 60}\nProcessing Dataset ID: {set_id_val}")

        try:
            # --- 1. Veri Hazırlığı ---
            dataset = openml.datasets.get_dataset(set_id_val, download_data=True)
            d_name = dataset.name
            X, y, _, _ = dataset.get_data(dataset_format="dataframe", target=dataset.default_target_attribute)

            X = categorize_to_numeric(X)
            X_vals = np.nan_to_num(X.values.astype(np.float64))
            y_vals = y.values if isinstance(y, pd.Series) else y
            y_vals = np.nan_to_num(y_vals.astype(np.float64))

            X_train, X_test, y_train, y_test = train_test_split(X_vals, y_vals, test_size=0.2, random_state=42)

            scaler_x = StandardScaler()
            X_train = scaler_x.fit_transform(X_train)
            X_test = scaler_x.transform(X_test)

            scaler_y = StandardScaler()
            y_train = scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()
            y_test = scaler_y.transform(y_test.reshape(-1, 1)).ravel()

            n_samples = X_train.shape[0]
            n_features = X_train.shape[1]
            dataset_stats.append({'Dataset': d_name, 'Samples': n_samples, 'Features': n_features})
            print(f"Dataset: {d_name} | Samples: {n_samples} | Features: {n_features}")

            # --- 2. Feature Extraction ---
            # A) ST
            print("> Applying: Symbolic Transformer...")
            stgp = SymbolicTransformer(n_components=5, generations=5, n_jobs=1, random_state=42)
            with np.errstate(divide='ignore', invalid='ignore'):
                stgp.fit(X_train, y_train)
                X_train_st = np.nan_to_num(stgp.transform(X_train))
                X_test_st = np.nan_to_num(stgp.transform(X_test))

            new_train_st = np.hstack((X_train, X_train_st))
            new_test_st = np.hstack((X_test, X_test_st))

            # B) EF
            print("> Applying: Evolutionary Forest...")
            ef = EvolutionaryForestRegressor(max_height=4, n_jobs=1, random_state=42)
            with np.errstate(divide='ignore', invalid='ignore'):
                ef.fit(X_train, y_train)
                X_train_ef = np.nan_to_num(ef.transform(X_train))
                X_test_ef = np.nan_to_num(ef.transform(X_test))

            if X_train_ef.shape[1] > 10:
                X_train_ef = X_train_ef[:, :10]
                X_test_ef = X_test_ef[:, :10]

            new_train_ef = np.hstack((X_train, X_train_ef))
            new_test_ef = np.hstack((X_test, X_test_ef))

            # C) Hybrid
            new_train_hy = np.hstack((X_train, X_train_st, X_train_ef))
            new_test_hy = np.hstack((X_test, X_test_st, X_test_ef))

            # --- 3. Evaluation ---
            print("> Evaluating Judges...")
            datasets_to_eval = {
                col_names['base']: (X_train, X_test),
                col_names['st']: (new_train_st, new_test_st),
                col_names['ef']: (new_train_ef, new_test_ef),
                col_names['hy']: (new_train_hy, new_test_hy)
            }

            for judge_name, judge_model in judges.items():
                for method_key, (train_data, test_data) in datasets_to_eval.items():
                    res = evaluate_single_model(train_data, test_data, y_train, y_test,
                                                judge_name, judge_model, d_name, method_key)
                    if res:
                        all_results.append(res)

        except Exception as e:
            print(f"!!! Error Processing {set_id_val}: {e}")
            continue

    if not all_results:
        print("Sonuç üretilemedi.")
        return None

    # --- 4. Veri İşleme ve Fark Hesaplama ---
    df_results = pd.DataFrame(all_results)
    df_pivot = df_results.pivot_table(index=['Dataset', 'Algorithm'], columns='Method', values='Score').reset_index()

    # Tüm metodlar için fark hesapla (Method - Base)
    # Eğer kolon yoksa farkı 0 kabul et (hata almamak için)
    if col_names['base'] in df_pivot.columns:
        # Transformer Farkı
        if col_names['st'] in df_pivot.columns:
            df_pivot['Transformer_Diff'] = df_pivot[col_names['st']] - df_pivot[col_names['base']]
        else:
            df_pivot['Transformer_Diff'] = np.nan

        # EF Farkı
        if col_names['ef'] in df_pivot.columns:
            df_pivot['EF_Diff'] = df_pivot[col_names['ef']] - df_pivot[col_names['base']]
        else:
            df_pivot['EF_Diff'] = np.nan

        # Hybrid Farkı
        if col_names['hy'] in df_pivot.columns:
            df_pivot['Hybrid_Diff'] = df_pivot[col_names['hy']] - df_pivot[col_names['base']]
        else:
            df_pivot['Hybrid_Diff'] = np.nan
    else:
        print("Base model skorları bulunamadı, fark hesaplanamıyor.")
        return df_pivot

    # İstatistikleri birleştir
    df_stats = pd.DataFrame(dataset_stats)
    df_final = pd.merge(df_pivot, df_stats, on='Dataset', how='left')

    # --- 5. Görselleştirme (GÜNCELLENMİŞ VERSİYON) ---
    print("\n> Generating Comprehensive Plots...")

    # Gerekli sütunların varlığını kontrol et
    diff_cols = ['Transformer_Diff', 'EF_Diff', 'Hybrid_Diff']
    available_diffs = [col for col in diff_cols if col in df_final.columns]

    if not available_diffs:
        print("Görselleştirme için gerekli fark sütunları (Diff) bulunamadı.")
        return df_final

    algorithms = df_final['Algorithm'].unique()
    n_algos = len(algorithms)
    n_methods = 3  # Transformer, EF, Hybrid

    # Her algoritma bir satır, metodlar sütun
    fig, axes = plt.subplots(n_algos, n_methods, figsize=(18, 5 * n_algos))

    # Tek algoritma varsa boyut hatasını önle
    if n_algos == 1:
        axes = axes.reshape(1, -1)

    # Renk skalası için global simetrik limit belirle (0 ortada kalsın diye)
    all_values = []
    for col in available_diffs:
        all_values.extend(df_final[col].dropna().values)

    if all_values:
        limit = max(abs(min(all_values)), abs(max(all_values)))
        vmin, vmax = -limit, limit
    else:
        vmin, vmax = -0.1, 0.1

    method_keys = ['Transformer', 'EF', 'Hybrid']
    method_titles = ['Transformer', 'Evolutionary Forest', 'Hybrid (Trans+EF)']
    diff_map = {
        'Transformer': 'Transformer_Diff',
        'EF': 'EF_Diff',
        'Hybrid': 'Hybrid_Diff'
    }

    # Ana Döngü: Algoritmalar
    for i, algo in enumerate(algorithms):
        subset = df_final[df_final['Algorithm'] == algo]

        # Alt Döngü: Metodlar
        for j, m_key in enumerate(method_keys):
            ax = axes[i, j]
            col_name = diff_map.get(m_key)

            # Veri veya Sütun yoksa
            if subset.empty or col_name not in df_final.columns:
                ax.text(0.5, 0.5, 'No Data', ha='center', transform=ax.transAxes)
                ax.set_axis_off()
                continue

            # 1. Scatter Çizimi
            sc = ax.scatter(subset['Samples'], subset['Features'],
                            c=subset[col_name],
                            cmap='RdYlGn',  # Kırmızı-Sarı-Yeşil
                            s=150, edgecolors='black', alpha=0.85,
                            vmin=vmin, vmax=vmax)

            # 2. Değerleri Yazdırma (Tüm noktalar için)
            for _, row in subset.iterrows():
                val = row[col_name]

                # Değeri biçimlendir (örn: +0.05)
                label_text = f"{val:+.2f}"

                # Logaritmik eksende yukarı öteleme (çarpma ile yapılır)
                # 1.20 çarpanı metni noktanın %20 yukarısına koyar.
                # Çakışma olursa bu değeri 1.25 veya 1.30 yapabilirsiniz.
                ax.text(row['Samples'], row['Features'] * 1.20,
                        label_text,
                        fontsize=9, ha='center', va='bottom',
                        fontweight='bold', color='black')

            ax.set_xscale('log')
            ax.set_yscale('log')

            # Ortalama İyileşme Değerini Hesapla ve Yazdır
            mean_diff = subset[col_name].mean()
            stats_text = f"Avg Imp: {mean_diff:+.4f}"

            # Yazı rengi: Pozitifse yeşil, negatifse kırmızı
            text_color = 'green' if mean_diff > 0 else 'red'

            # Köşedeki bilgi kutusu
            ax.text(0.05, 0.95, stats_text, transform=ax.transAxes,
                    fontsize=11, fontweight='bold', color=text_color,
                    verticalalignment='top', bbox=dict(boxstyle="round", fc="white", ec="gray", alpha=0.8))

            # Başlık ve Eksenler
            ax.set_title(f'{algo}\n{method_titles[j]}', fontsize=12, fontweight='bold')

            # Sadece en alt satırda X ekseni etiketi, sadece en sol sütunda Y ekseni etiketi
            if i == n_algos - 1:
                ax.set_xlabel('Samples (Log)', fontsize=10)
            if j == 0:
                ax.set_ylabel('Features (Log)', fontsize=10)

            ax.grid(True, which="both", ls="--", alpha=0.3)

    # --- Yerleşim Ayarları ---
    plt.subplots_adjust(top=0.95, bottom=0.08, left=0.05, right=0.92, hspace=0.3, wspace=0.2)

    # Ortak Colorbar (Sağ Taraf)
    cbar_ax = fig.add_axes([0.93, 0.15, 0.015, 0.7])
    cbar = fig.colorbar(sc, cax=cbar_ax)
    cbar.set_label('R2 Score Improvement (Method - Base)', fontsize=13, fontweight='bold')

    # 0 noktasına belirgin bir çizgi ekle (Colorbar üzerinde)
    cbar.ax.plot([0, 1], [0.5, 0.5], 'k-', lw=2, transform=cbar.ax.transAxes)

    plt.suptitle('Performance Improvement by Algorithm & Method (Relative to Base)', fontsize=18, y=0.98)
    plt.show()

    print("\nİşlem Tamamlandı. Görselleştirme oluşturuldu.")
    return df_final