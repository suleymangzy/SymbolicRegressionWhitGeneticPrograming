import warnings
import openml
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from gplearn.genetic import SymbolicTransformer, SymbolicRegressor
from evolutionary_forest.forest import EvolutionaryForestRegressor

# EF isimlendirmesi için gerekli import
try:
    from evolutionary_forest.utils import get_feature_importance
except ImportError:
    # Eğer kütüphane versiyonunda utils altında değilse alternatif
    from evolutionary_forest.utils import get_feature_importance

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.base import clone
from sklearn.preprocessing import StandardScaler
from Functions.Comparison import categorize_to_numeric

# Ayarlar
warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
pd.set_option('display.float_format', '{:.4f}'.format)


def evaluate_single_model(X_train, X_test, y_train, y_test, model_name, model_instance, dataset_name, method_name):
    """
    Verilen eğitim verisiyle modeli eğitir ve test verisi üzerinde R2 skorunu hesaplar.
    """
    try:
        model = clone(model_instance)
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
        print(f"Evaluation Error ({model_name} - {method_name}): {e}")
        return None


def truncate_string(s, length=85):
    s = str(s)
    if len(s) > length:
        return s[:length - 3] + "..."
    return s


def print_feature_table(features, title):
    """
    Üretilen özellikleri (Feature Selection yapmadan) listeler.
    """
    w_rank, w_type, w_expr = 6, 15, 90
    header = f"| {'Index':^{w_rank}} | {'Source':^{w_type}} | {'Expression / Name':<{w_expr}} |"
    divider = "+" + "-" * (w_rank + 2) + "+" + "-" * (w_type + 2) + "+" + "-" * (w_expr + 2) + "+"

    print(f"\n   {title}")
    print("   " + divider)
    print("   " + header)
    print("   " + divider)

    if not features:
        print(f"   | {'No features generated.':^{w_rank + w_type + w_expr + 6}} |")
    else:
        for idx, item in enumerate(features, 1):
            c_type = item.get('type', '-')
            c_expr = truncate_string(item.get('name', '-'), length=w_expr)
            row = f"| {idx:^{w_rank}} | {c_type:^{w_type}} | {c_expr:<{w_expr}} |"
            print("   " + row)

    print("   " + divider)


def run_comprehensive_analysis(sets_id):
    all_results = []

    # Hakem Algoritmalar (Değerlendiriciler)
    judges = {
        'RF': RandomForestRegressor(n_estimators=200, n_jobs=-1, random_state=42),
        'XGBoost': XGBRegressor(n_jobs=1, n_estimators=200, verbosity=0, random_state=42)
    }

    # Sütun İsimlendirmeleri
    col_names = {
        'gp': '1. GP (Symb. Reg)',
        'st': '2. Transformer',
        'ef': '3. EF (Evo. Forest)',
        'hy': '4. Transformer+EF'
    }

    for set_id_val in sets_id:
        print(f"\n{'#' * 80}\nProcessing Dataset ID: {set_id_val}\n{'#' * 80}")

        try:
            # --- 1. Veri Hazırlığı ---
            dataset = openml.datasets.get_dataset(set_id_val, download_data=True)
            d_name = dataset.name
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

            print(f"Dataset: {d_name} | Samples: {X_train.shape[0]} | Original Features: {X_train.shape[1]}")

            # --- 2. Feature Extraction İşlemleri ---

            # A) Symbolic Regressor (GP)
            print("\n> Applying: Symbolic Regressor (SRGP)...")
            srgp = SymbolicRegressor(n_jobs=1, random_state=42)
            with np.errstate(divide='ignore', invalid='ignore'):
                srgp.fit(X_train, y_train)
                expr = srgp._program
                X_train_sr = np.nan_to_num(expr.execute(X_train).reshape(-1, 1))
                X_test_sr = np.nan_to_num(expr.execute(X_test).reshape(-1, 1))

            # Tablo
            feat_list_sr = [{'type': 'SRGP', 'name': str(expr)}]
            print_feature_table(feat_list_sr, f"Generated Features: {col_names['gp']}")

            new_train_sr = np.hstack((X_train, X_train_sr))
            new_test_sr = np.hstack((X_test, X_test_sr))

            # B) Symbolic Transformer (ST)
            print("\n> Applying: Symbolic Transformer (STGP)...")
            stgp = SymbolicTransformer(n_jobs=1, random_state=42)
            with np.errstate(divide='ignore', invalid='ignore'):
                stgp.fit(X_train, y_train)
                X_train_st = stgp.transform(X_train)
                X_test_st = stgp.transform(X_test)
                X_train_st = np.nan_to_num(X_train_st)
                X_test_st = np.nan_to_num(X_test_st)

            # Formüller
            feat_list_st = [{'type': 'STGP', 'name': str(p)} for p in stgp._best_programs]
            print_feature_table(feat_list_st, f"Generated Features: {col_names['st']}")

            new_train_st = np.hstack((X_train, X_train_st))
            new_test_st = np.hstack((X_test, X_test_st))

            # C) Evolutionary Forest (EF)
            print("\n> Applying: Evolutionary Forest (EF)...")
            ef = EvolutionaryForestRegressor(random_state=42)
            with np.errstate(divide='ignore', invalid='ignore'):
                ef.fit(X_train, y_train)
                X_train_ef = ef.transform(X_train)
                X_test_ef = ef.transform(X_test)

            # --- EF İSİMLENDİRME DÜZELTMESİ (ADIM 1) ---
            try:
                fi_dict = get_feature_importance(ef)
                ef_keys = list(fi_dict.keys())

                if len(ef_keys) > 0:
                    ef_names = [str(k) for k in ef_keys]
                    # Boyut Uyuşmazlığı Kontrolü
                    if len(ef_names) < X_train_ef.shape[1]:
                        missing = X_train_ef.shape[1] - len(ef_names)
                        for i in range(missing):
                            ef_names.append(f"Unknown_EF_{i}")
                    elif len(ef_names) > X_train_ef.shape[1]:
                        ef_names = ef_names[:X_train_ef.shape[1]]
                else:
                    ef_names = [f"EF_Gen_{i}" for i in range(X_train_ef.shape[1])]
            except Exception as ef_err:
                print(f"EF İsimlendirme Hatası: {ef_err}")
                ef_names = [f"EF_Gen_{i}" for i in range(X_train_ef.shape[1])]

            # Özellik Sayısını Sınırlama (Tablo ve Veri için)
            if X_train_ef.shape[1] > 10:
                X_train_ef = X_train_ef[:, :10]
                X_test_ef = X_test_ef[:, :10]
                ef_names = ef_names[:10]

            X_train_ef = np.nan_to_num(X_train_ef)
            X_test_ef = np.nan_to_num(X_test_ef)

            # İsimleri Listeye Çevir
            feat_list_ef = [{'type': 'EF', 'name': name} for name in ef_names]
            print_feature_table(feat_list_ef, f"Generated Features: {col_names['ef']}")

            new_train_ef = np.hstack((X_train, X_train_ef))
            new_test_ef = np.hstack((X_test, X_test_ef))

            # D) Hybrid (Transformer + EF) - OPTİMİZE EDİLMİŞ HALİ
            print("\n> Applying: Hybrid (Transformer + EF)...")

            # Modelleri tekrar eğitmek yerine, B ve C adımlarında üretilen
            # ve halihazırda bellekte olan özellikleri birleştiriyoruz.

            # 1. Özellik İsim Listelerini Birleştir
            # feat_list_st: Step B'den geliyor
            # feat_list_ef: Step C'den geliyor (zaten ilk 10 seçim ve isimlendirme yapılmış hali)
            feat_list_hy = feat_list_st + feat_list_ef

            print_feature_table(feat_list_hy, f"Generated Features: {col_names['hy']}")

            # 2. Veri Setlerini Birleştir (Stacking)
            # X_train: Orijinal veriler
            # X_train_st: Step B'den gelen Symbolic Transformer özellikleri
            # X_train_ef: Step C'den gelen (ve sınırlanmış) EF özellikleri

            new_train_hy = np.hstack((X_train, X_train_st, X_train_ef))
            new_test_hy = np.hstack((X_test, X_test_st, X_test_ef))

            # --- 3. Değerlendirme Döngüsü ---
            print("\n>>> Evaluating Models with Augmented Datasets...")

            for judge_name, judge_model in judges.items():

                # 0. Base Model (Ham Veri)
                res_base = evaluate_single_model(X_train, X_test, y_train, y_test,
                                                 judge_name, judge_model, d_name, "0. Base")
                if res_base: all_results.append(res_base)

                # 1. GP (Symbolic Regressor)
                res_gp = evaluate_single_model(new_train_sr, new_test_sr, y_train, y_test,
                                               judge_name, judge_model, d_name, col_names['gp'])
                if res_gp: all_results.append(res_gp)

                # 2. Transformer
                res_st = evaluate_single_model(new_train_st, new_test_st, y_train, y_test,
                                               judge_name, judge_model, d_name, col_names['st'])
                if res_st: all_results.append(res_st)

                # 3. EF
                res_ef = evaluate_single_model(new_train_ef, new_test_ef, y_train, y_test,
                                               judge_name, judge_model, d_name, col_names['ef'])
                if res_ef: all_results.append(res_ef)

                # 4. Hybrid
                res_hy = evaluate_single_model(new_train_hy, new_test_hy, y_train, y_test,
                                               judge_name, judge_model, d_name, col_names['hy'])
                if res_hy: all_results.append(res_hy)

        except Exception as e:
            print(f"Hata (Dataset ID {set_id_val}): {e}")
            import traceback
            traceback.print_exc()
            continue

    if not all_results:
        print("Sonuç üretilemedi.")
        return None

    # --- Pivot Tablo ---
    df_results = pd.DataFrame(all_results)

    pivot_df = pd.pivot_table(df_results,
                              index='Dataset',
                              columns=['Algorithm', 'Method'],
                              values='Score')

    pivot_df = pivot_df.fillna('-')

    # Sıralama
    try:
        pivot_df = pivot_df.reindex(sorted(pivot_df.columns), axis=1)
    except:
        pass

    return pivot_df