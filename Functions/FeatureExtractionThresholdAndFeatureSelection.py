import time
import warnings
import openml
import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
from evolutionary_forest.utils import get_feature_importance
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, AdaBoostRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.base import clone
from sklearn.preprocessing import StandardScaler
from gplearn.genetic import SymbolicTransformer
from evolutionary_forest.forest import EvolutionaryForestRegressor
from Functions.Comparison import categorize_to_numeric

warnings.filterwarnings("ignore")

def evaluate_models(X_train_new, X_test_new, y_train, y_test, dataset_name, method_name):

    regressor_dict = {
        'RF': RandomForestRegressor(n_estimators=200, n_jobs=-1, random_state=42),
        'XGBoost': XGBRegressor(n_jobs=1, n_estimators=200, verbosity=0, random_state=42),
    }

    results = []

    for regr_name, model_instance in regressor_dict.items():
        try:
            enh_model = clone(model_instance)
            enh_model.fit(X_train_new, y_train)
            e_score = r2_score(y_test, enh_model.predict(X_test_new))

            results.append({
                'Algorithm': regr_name,
                'Dataset': dataset_name,
                'Method': method_name,
                'Score': e_score
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


def run_comparative_analysis_rf_selection(sets_id, use_threshold=True, min_importance_feature_ef=0.05,
                                          min_importance_feature_st=0.05):
    """
    Hem STGP hem de EF için özellik seçiminde Random Forest Feature Importance yöntemini kullanır.
    STGP aşamasında EF ve STGP özelliklerini ortak havuzda değerlendirir.
    """
    all_results = []

    for set_id_val in sets_id:
        try:
            print(f"Processing: Dataset ID {set_id_val}...")
            # --- Veri İndirme ve Ön İşleme Bloğu ---
            max_retries = 3
            retry_delay = 5
            X = None

            for attempt in range(max_retries):
                try:
                    dataset = openml.datasets.get_dataset(set_id_val, download_data=True)
                    d_name = f"{dataset.name}"
                    X, y, _, _ = dataset.get_data(
                        dataset_format="dataframe", target=dataset.default_target_attribute
                    )
                    break
                except Exception as e:
                    print(f"Hata: Dataset ID {set_id_val} alınırken sorun (Deneme {attempt + 1}) - {e}")
                    if attempt < max_retries - 1: time.sleep(retry_delay)

            if X is None: continue

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

            # Base Model Değerlendirmesi
            all_results.extend(evaluate_models(X_train, X_test, y_train, y_test, d_name, "Base"))

            # ==========================================
            # 1. Evolutionary Forest (EF) - Feature Selection with Random Forest
            # ==========================================
            print("\n--- Running Evolutionary Forest ---")
            ef = EvolutionaryForestRegressor(random_state=42)
            ef.fit(X_train, y_train)

            # EF'in ürettiği ham özellik matrisini al (Transform)
            X_train_ef_candidates = ef.transform(X_train)
            X_test_ef_candidates = ef.transform(X_test)

            # Formül isimlerini sözlükten çek
            try:
                feature_importance_dict = get_feature_importance(ef)
                ef_formulas_list = list(feature_importance_dict.keys())

                # Güvenlik Kontrolü
                if len(ef_formulas_list) == X_train_ef_candidates.shape[1]:
                    ef_feature_names = [str(f) for f in ef_formulas_list]
                else:
                    print("Warning: EF formula count mismatch. Using generic names.")
                    ef_feature_names = [f"EF_Feat_{i}" for i in range(X_train_ef_candidates.shape[1])]
            except Exception as e:
                print(f"Warning: Could not get EF formulas ({e}). Using generic names.")
                ef_feature_names = [f"EF_Feat_{i}" for i in range(X_train_ef_candidates.shape[1])]

            print(f"EF generated {X_train_ef_candidates.shape[1]} features.")

            # RF ile Puanlama
            rf_selector_ef = RandomForestRegressor(n_estimators=100, n_jobs=-1, random_state=42)
            rf_selector_ef.fit(X_train_ef_candidates, y_train)
            importances_ef = rf_selector_ef.feature_importances_

            # (İsim, Önem, İndeks) listesi oluştur
            ef_features_list = []
            for idx, imp in enumerate(importances_ef):
                ef_features_list.append((ef_feature_names[idx], imp, idx))

            # Önem derecesine göre sırala
            sorted_ef = sorted(ef_features_list, key=lambda x: x[1], reverse=True)

            selected_ef_indices = []

            if use_threshold:
                print("Selecting EF features using Threshold...")
                selected_ef_indices = [idx for name, imp, idx in sorted_ef if imp >= min_importance_feature_ef]

                if not selected_ef_indices:
                    print(f"Warning: No EF features passed threshold {min_importance_feature_ef}. Selecting top 5.")
                    selected_ef_indices = [idx for name, imp, idx in sorted_ef[:5]]
            else:
                print("Selecting TOP 10 EF features (Ranking Mode)...")
                top_k = min(10, len(sorted_ef))
                selected_ef_indices = [idx for name, imp, idx in sorted_ef[:top_k]]

            # Seçilenleri Listele
            print("-" * 80)
            print(f"{'Rank':<4} | {'Importance':<12} | {'Feature Formula / Name'}")
            print("-" * 80)
            for rank, idx in enumerate(selected_ef_indices[:10]):
                imp = importances_ef[idx]
                name = ef_feature_names[idx]
                disp_name = (name[:55] + '...') if len(name) > 58 else name
                print(f"{rank + 1:<4} | {imp:.6f}      | {disp_name}")
            print("-" * 80)

            # 1. Aşama İçin Veriyi Filtrele
            X_train_ef = X_train_ef_candidates[:, selected_ef_indices]
            X_test_ef = X_test_ef_candidates[:, selected_ef_indices]

            # İkinci aşama için seçilen isimleri sakla
            selected_ef_names = [ef_feature_names[i] for i in selected_ef_indices]

            # EF Sonuçlarını Değerlendir
            new_train_ef = np.hstack((X_train, X_train_ef))
            new_test_ef = np.hstack((X_test, X_test_ef))
            print(f'EF Selected Columns: {X_train_ef.shape[1]} (Total: {new_train_ef.shape[1]})')
            all_results.extend(evaluate_models(new_train_ef, new_test_ef, y_train, y_test, d_name, "EF"))

            # ==========================================
            # 2. Symbolic Transformer (STGP) & Joint Selection with EF
            # ==========================================
            print("\n--- Running Symbolic Transformer (STGP) & Joint Selection with Evolutionary Forest (EF) ---")
            stgp = SymbolicTransformer(random_state=42)
            stgp.fit(X_train, y_train)

            best_programs = stgp._best_programs

            # STGP adaylarını matris haline getir
            if best_programs:
                candidate_outputs_train = [prog.execute(X_train) for prog in best_programs]
                candidate_outputs_test = [prog.execute(X_test) for prog in best_programs]

                X_train_st_all = np.column_stack(candidate_outputs_train)
                X_test_st_all = np.column_stack(candidate_outputs_test)
                st_feature_names = [str(prog) for prog in best_programs]
            else:
                X_train_st_all = np.zeros((X_train.shape[0], 0))
                X_test_st_all = np.zeros((X_test.shape[0], 0))
                st_feature_names = []

            print(f"STGP generated {X_train_st_all.shape[1]} features.")

            # 2. HAVUZU OLUŞTUR: [SEÇİLMİŞ EF Çıktıları] + [STGP Adayları]
            X_pool_train = np.hstack((X_train_ef, X_train_st_all))

            num_ef = X_train_ef.shape[1]
            num_st = X_train_st_all.shape[1]

            print(f"Joint Pool Created: {num_ef} EF features vs {num_st} STGP candidates.")
            print("Calculating feature importance on the JOINT pool (Random Forest)...")

            # 3. Hakem Model: Random Forest (Ortak Havuz Üzerinde)
            rf_selector_joint = RandomForestRegressor(n_estimators=100, n_jobs=-1, random_state=42)
            rf_selector_joint.fit(X_pool_train, y_train)
            importances_joint = rf_selector_joint.feature_importances_

            # Listeyi Oluştur (Tip, İndeks, Önem, İsim)
            joint_features = []

            # --- EF Özelliklerini Ekleme ---
            for i in range(num_ef):
                name = selected_ef_names[i] if i < len(selected_ef_names) else f"EF_Sel_{i}"
                joint_features.append({
                    'type': 'EF',
                    'idx': i,
                    'score': importances_joint[i],
                    'name': name
                })

            # --- STGP Ekle ---
            for i in range(num_st):
                name = st_feature_names[i]
                joint_features.append({
                    'type': 'STGP',
                    'idx': i,  # Kendi matrisi içindeki indeksi
                    'score': importances_joint[num_ef + i],  # Joint içindeki offsetli skor
                    'name': name
                })

            # 5. Sıralama (En yüksek puandan en düşüğe) -> Düzeltme: Dict key 'score'
            sorted_joint = sorted(joint_features, key=lambda x: x['score'], reverse=True)

            # -------------------------------------------------------------
            # B) SEÇİM MANTIĞI (Threshold vs Top-10)
            # -------------------------------------------------------------

            final_ef_indices = []
            final_stgp_indices = []
            seen_formulas = set()

            if use_threshold:
                print(f"Selecting from Joint Pool with Threshold >= {min_importance_feature_st}...")

                for item in sorted_joint:
                    if item['score'] >= min_importance_feature_st:
                        if item['name'] not in seen_formulas:
                            if item['type'] == 'EF':
                                final_ef_indices.append(item['idx'])
                            elif item['type'] == 'STGP':
                                final_stgp_indices.append(item['idx'])
                            seen_formulas.add(item['name'])

                # Fallback
                if not final_ef_indices and not final_stgp_indices:
                    print("Warning: No features passed threshold in joint selection. Fallback to top 5.")
                    for item in sorted_joint[:5]:
                        if item['name'] not in seen_formulas:
                            if item['type'] == 'EF':
                                final_ef_indices.append(item['idx'])
                            elif item['type'] == 'STGP':
                                final_stgp_indices.append(item['idx'])
                            seen_formulas.add(item['name'])

            else:
                # --- Ranking Mode: En iyi 10'u seç ---
                print("Selecting TOP 10 features from Joint Pool (Ranking Mode)...")
                top_k = 10
                count = 0

                for item in sorted_joint:
                    if count >= top_k:
                        break

                    if item['name'] not in seen_formulas:
                        if item['type'] == 'EF':
                            final_ef_indices.append(item['idx'])
                        elif item['type'] == 'STGP':
                            final_stgp_indices.append(item['idx'])
                        seen_formulas.add(item['name'])
                        count += 1

            # -------------------------------------------------------------
            # C) LİSTELEME VE MATRİS OLUŞTURMA
            # -------------------------------------------------------------

            print("-" * 100)
            print(f"{'Rank':<4} | {'Type':<6} | {'Importance':<12} | {'Feature Formula / Name'}")
            print("-" * 100)

            rank_disp = 0
            for item in sorted_joint:
                is_selected = (item['type'] == 'EF' and item['idx'] in final_ef_indices) or \
                              (item['type'] == 'STGP' and item['idx'] in final_stgp_indices)

                if is_selected and rank_disp < 20:
                    disp_name = (item['name'][:70] + '..') if len(item['name']) > 70 else item['name']
                    print(f"{rank_disp + 1:<4} | {item['type']:<6} | {item['score']:.6f}      | {disp_name}")
                    rank_disp += 1

            print("-" * 100)
            print(
                f"Result: Selected {len(final_ef_indices)} features from EF and {len(final_stgp_indices)} features from STGP.")

            # 1. EF Final Matrisini Oluştur
            # Not: X_train_ef zaten Step 1'den filtrelenmiş geldiği için, buradaki indeksler ona göredir.
            final_ef_indices.sort()
            if final_ef_indices:
                X_train_ef_final = X_train_ef[:, final_ef_indices]
                X_test_ef_final = X_test_ef[:, final_ef_indices]
            else:
                X_train_ef_final = np.zeros((X_train.shape[0], 0))
                X_test_ef_final = np.zeros((X_test.shape[0], 0))

            # 2. STGP Final Matrisini Oluştur
            # Not: X_train_st_all Step 2'de hesaplanmıştı.
            final_stgp_indices.sort()
            if final_stgp_indices:
                X_train_st_final = X_train_st_all[:, final_stgp_indices]
                X_test_st_final = X_test_st_all[:, final_stgp_indices]
            else:
                X_train_st_final = np.zeros((X_train.shape[0], 0))
                X_test_st_final = np.zeros((X_test.shape[0], 0))

            # ==========================================
            # 3. Final Birleştirme (Raw + Selected EF + Selected STGP)
            # ==========================================
            new_train_stgp_ef = np.hstack((X_train, X_train_ef_final, X_train_st_final))
            new_test_stgp_ef = np.hstack((X_test, X_test_ef_final, X_test_st_final))

            print(f'Final Combined Columns (Raw + EF + STGP): {new_train_stgp_ef.shape[1]}')

            all_results.extend(evaluate_models(new_train_stgp_ef, new_test_stgp_ef,
                                               y_train, y_test, d_name, "STGP+EF"))

        except Exception as e:
            print(f"Genel Hata (Dataset ID: {set_id_val}): {e}")
            import traceback
            traceback.print_exc()
            continue

    if not all_results:
        return None

    df_results = pd.DataFrame(all_results)
    pivot_df = pd.pivot_table(
        df_results,
        index='Algorithm',
        columns=['Dataset', 'Method'],
        values='Score'
    )

    return pivot_df


# Yardımcı Fonksiyon: Tek bir modeli değerlendirmek için
def evaluate_single_model(X_train, X_test, y_train, y_test, model_name, model_instance, dataset_name, method_name):
    try:
        model = clone(model_instance)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        score = r2_score(y_test, preds)
        return {
            'Algorithm': model_name,
            'Dataset': dataset_name,
            'Method': method_name,  # Örn: "STGP+EF (Judge: XGB)"
            'Score': score
        }
    except Exception as e:
        print(f"Evaluation Error ({model_name}): {e}")
        return None


def run_hybrid_comparative_analysis(sets_id, use_threshold=True, min_importance=0.01):
    all_results = []

    judges = {
        'RF': RandomForestRegressor(n_estimators=200, n_jobs=-1, random_state=42),
        'XGBoost': XGBRegressor(n_jobs=1, n_estimators=200, verbosity=0, random_state=42)
    }

    for set_id_val in sets_id:

        print(f"\n{'=' * 40}\nProcessing: Dataset ID {set_id_val}\n{'=' * 40}")

        dataset = openml.datasets.get_dataset(set_id_val, download_data=True)

        d_name = dataset.name
        X, y, _, _ = dataset.get_data(dataset_format="dataframe", target=dataset.default_target_attribute)

        X = categorize_to_numeric(X)
        X_vals = np.nan_to_num(X.values.astype(np.float64))
        y_vals = y.values.astype(np.float64)

        X_train, X_test, y_train, y_test = train_test_split(X_vals, y_vals, test_size=0.2, random_state=42)

        scaler_x = StandardScaler()
        X_train = scaler_x.fit_transform(X_train)
        X_test = scaler_x.transform(X_test)


        print(f"Raw Data Columns: {X_train.shape[1]}")

            # BASE MODEL (Sadece Ham Veri)
        for name, model in judges.items():
            res = evaluate_single_model(X_train, X_test, y_train, y_test, name, model, d_name, "Base (Raw Only)")
            if res: all_results.append(res)

            # --- 2. ÖZELLİK ÜRETİMİ (TEK SEFERLİK) ---

            # A) Evolutionary Forest (EF) Üretimi
        print("\nGenerating EF Candidates...")
        ef = EvolutionaryForestRegressor(random_state=42)
        ef.fit(X_train, y_train)
        X_train_ef_all = ef.transform(X_train)
        X_test_ef_all = ef.transform(X_test)


        fi_dict = get_feature_importance(ef)
        ef_formulas = list(fi_dict.keys())
        ef_names = [str(f) for f in ef_formulas] if len(ef_formulas) == X_train_ef_all.shape[1] else [f"EF_{i}" for i in range(X_train_ef_all.shape[1])]

        print("Generating STGP Candidates...")
        stgp = SymbolicTransformer(random_state=42)
        stgp.fit(X_train, y_train)
        best_progs = stgp._best_programs

        if best_progs:
            out_train = [p.execute(X_train) for p in best_progs]
            out_test = [p.execute(X_test) for p in best_progs]
            X_train_st_all = np.column_stack(out_train)
            X_test_st_all = np.column_stack(out_test)
            st_names = [str(p) for p in best_progs]
        else:
            X_train_st_all = np.zeros((X_train.shape[0], 0))
            X_test_st_all = np.zeros((X_test.shape[0], 0))
            st_names = []

        for judge_name, judge_model in judges.items():
            print(f"\n>>> JUDGE: {judge_name} <<<")

            selector_ef = clone(judge_model)
            selector_joint = clone(judge_model)

            print(f"  [1] Selection from EF Pool ({X_train_ef_all.shape[1]} features)...")
            selector_ef.fit(X_train_ef_all, y_train)
            imp_ef = selector_ef.feature_importances_

            ef_candidates = [(i, imp, ef_names[i]) for i, imp in enumerate(imp_ef)]
            ef_candidates.sort(key=lambda x: x[1], reverse=True)

            sel_idx_ef = []
            seen_ef = set()

            if use_threshold:
                for idx, score, name in ef_candidates:
                    if score >= min_importance and name not in seen_ef:
                        sel_idx_ef.append(idx)
                        seen_ef.add(name)
                if not sel_idx_ef:  # Fallback
                    sel_idx_ef = [x[0] for x in ef_candidates[:5]]
            else:
                sel_idx_ef = [x[0] for x in ef_candidates[:10]]

                # Birleştirme (Raw + Selected EF)
            X_train_sel_ef = X_train_ef_all[:, sel_idx_ef]
            X_test_sel_ef = X_test_ef_all[:, sel_idx_ef]

            X_train_final_ef = np.hstack((X_train, X_train_sel_ef))
            X_test_final_ef = np.hstack((X_test, X_test_sel_ef))

            print(f"  -> Evaluating EF (Raw + {len(sel_idx_ef)} EF features)...")
            res_ef = evaluate_single_model(
                X_train_final_ef, X_test_final_ef, y_train, y_test,
                judge_name, judge_model, d_name, f"EF (Judge: {judge_name})"
            )
            if res_ef: all_results.append(res_ef)

            X_pool_train = np.hstack((X_train_ef_all, X_train_st_all))
            X_pool_test = np.hstack((X_test_ef_all, X_test_st_all))

            num_ef = X_train_ef_all.shape[1]
            num_st = X_train_st_all.shape[1]

            print(f"  [2] Selection from Joint Pool ({X_pool_train.shape[1]} features)...")
            selector_joint.fit(X_pool_train, y_train)
            imp_joint = selector_joint.feature_importances_

            joint_cands = []
            for i in range(num_ef):
                joint_cands.append({'type': 'EF', 'idx': i, 'score': imp_joint[i], 'name': ef_names[i]})
            for i in range(num_st):
                joint_cands.append({'type': 'STGP', 'idx': i, 'score': imp_joint[num_ef + i], 'name': st_names[i]})

            joint_cands.sort(key=lambda x: x['score'], reverse=True)

            sel_idx_pool = []
            seen_joint = set()

            if use_threshold:
                for item in joint_cands:
                    if item['score'] >= min_importance and item['name'] not in seen_joint:
                        pool_idx = item['idx'] if item['type'] == 'EF' else (num_ef + item['idx'])
                        sel_idx_pool.append(pool_idx)
                        seen_joint.add(item['name'])
                if not sel_idx_pool:
                    for item in joint_cands[:5]:
                        pool_idx = item['idx'] if item['type'] == 'EF' else (num_ef + item['idx'])
                        sel_idx_pool.append(pool_idx)
            else:
                count = 0
                for item in joint_cands:
                    if count >= 10: break
                    if item['name'] not in seen_joint:
                        pool_idx = item['idx'] if item['type'] == 'EF' else (num_ef + item['idx'])
                        sel_idx_pool.append(pool_idx)
                        seen_joint.add(item['name'])
                        count += 1

            if sel_idx_pool:
                X_train_sel_joint = X_pool_train[:, sel_idx_pool]
                X_test_sel_joint = X_pool_test[:, sel_idx_pool]
            else:
                X_train_sel_joint = np.zeros((X_train.shape[0], 0))
                X_test_sel_joint = np.zeros((X_test.shape[0], 0))

            X_train_final_joint = np.hstack((X_train, X_train_sel_joint))
            X_test_final_joint = np.hstack((X_test, X_test_sel_joint))

            print(f"  -> Evaluating STGP+EF (Raw + {len(sel_idx_pool)} Joint features)...")
            res_joint = evaluate_single_model(
                X_train_final_joint, X_test_final_joint, y_train, y_test,
                judge_name, judge_model, d_name, f"STGP+EF (Judge: {judge_name})"
            )
            if res_joint: all_results.append(res_joint)

    if not all_results: return None
    return pd.pivot_table(pd.DataFrame(all_results), index='Algorithm', columns=['Dataset', 'Method'], values='Score')


def truncate_string(s, length=85):
    """Uzun formülleri tabloyu bozmaması için kısaltır."""
    if len(s) > length:
        return s[:length - 3] + "..."
    return s


def print_feature_table(candidates, title):
    """
    Seçilen özellikleri hizalı ve çerçeveli bir tablo halinde basar.
    """
    if not candidates:
        print(f"\n   {title}")
        print("   " + "-" * 50)
        print("   (Eşik değeri geçilemedi veya seçim yapılmadı)")
        print("   " + "-" * 50)
        return

    # Sütun Genişlikleri
    w_rank = 6
    w_type = 8
    w_score = 10
    w_expr = 90  # Formül alanı genişliği

    # Başlık ve Çizgiler
    header = f"| {'Rank':^{w_rank}} | {'Type':^{w_type}} | {'Score':^{w_score}} | {'Expression':<{w_expr}} |"
    divider = "+" + "-" * (w_rank + 2) + "+" + "-" * (w_type + 2) + "+" + "-" * (w_score + 2) + "+" + "-" * (
                w_expr + 2) + "+"

    print(f"\n   {title}")
    print("   " + divider)
    print("   " + header)
    print("   " + divider)

    for rank, item in enumerate(candidates, 1):
        # Verileri hazırla
        c_type = item['type']
        c_score = item['score']
        # Formülü temizle ve sığdır
        c_expr = truncate_string(str(item['name']), length=w_expr)

        # Satırı bas
        row = f"| {rank:^{w_rank}} | {c_type:^{w_type}} | {c_score:^{w_score}.4f} | {c_expr:<{w_expr}} |"
        print("   " + row)

    print("   " + divider)


def run_hybrid_comparative_analysis1(sets_id, use_threshold=True, min_importance=0.01):
    all_results = []

    judges = {
        'RF': RandomForestRegressor(n_estimators=200, n_jobs=-1, random_state=42),
        'XGBoost': XGBRegressor(n_jobs=1, n_estimators=200, verbosity=0, random_state=42)
    }

    for set_id_val in sets_id:
        print(f"\n{'#' * 60}\nProcessing: Dataset ID {set_id_val}\n{'#' * 60}")

        try:
            dataset = openml.datasets.get_dataset(set_id_val, download_data=True)
            d_name = dataset.name
            X, y, _, _ = dataset.get_data(dataset_format="dataframe", target=dataset.default_target_attribute)

            X = categorize_to_numeric(X)
            X_vals = np.nan_to_num(X.values.astype(np.float64))
            y_vals = y.values.astype(np.float64)

            X_train, X_test, y_train, y_test = train_test_split(X_vals, y_vals, test_size=0.2, random_state=42)

            scaler_x = StandardScaler()
            X_train = scaler_x.fit_transform(X_train)
            X_test = scaler_x.transform(X_test)

            print(f"Dataset: {d_name} | Raw Features: {X_train.shape[1]}")

            # 1. BASE MODEL
            for name, model in judges.items():
                res = evaluate_single_model(X_train, X_test, y_train, y_test, name, model, d_name, "1. Base")
                if res: all_results.append(res)

            # --- 2. ÖZELLİK ÜRETİMİ ---

            # A) EF Üretimi
            print("\n> Generating EF Candidates...")
            ef = EvolutionaryForestRegressor(random_state=42)
            ef.fit(X_train, y_train)
            X_train_ef_all = ef.transform(X_train)
            X_test_ef_all = ef.transform(X_test)

            # --- EF İSİMLENDİRME DÜZELTMESİ ---
            fi_dict = get_feature_importance(ef)
            ef_keys = list(fi_dict.keys())

            # Burada 'EF_Feat_X' yerine doğrudan anahtarları string'e çeviriyoruz.
            # Sayı uyuşmazlığı olsa bile, elimizdeki anahtarları kullanmaya çalışacağız.
            if len(ef_keys) > 0:
                ef_names = [str(k) for k in ef_keys]

                # Eğer transform edilen sütun sayısı ile isim sayısı tutmuyorsa
                # (Genelde EF en iyi özellikleri seçer ama transform hepsini döndürebilir)
                # Eksik kalanları doldurmak veya fazlaları kırpmak gerekebilir.
                if len(ef_names) < X_train_ef_all.shape[1]:
                    missing = X_train_ef_all.shape[1] - len(ef_names)
                    for i in range(missing):
                        ef_names.append(f"Unknown_EF_{i}")
                elif len(ef_names) > X_train_ef_all.shape[1]:
                    ef_names = ef_names[:X_train_ef_all.shape[1]]
            else:
                # Hiç isim gelmezse mecburen numaralandır
                ef_names = [f"EF_Gen_{i}" for i in range(X_train_ef_all.shape[1])]

            # B) STGP Üretimi
            print("> Generating STGP Candidates...")
            stgp = SymbolicTransformer(random_state=42)
            stgp.fit(X_train, y_train)
            best_progs = stgp._best_programs

            if best_progs:
                out_train = [p.execute(X_train) for p in best_progs]
                out_test = [p.execute(X_test) for p in best_progs]
                X_train_st_all = np.column_stack(out_train)
                X_test_st_all = np.column_stack(out_test)
                st_names = [str(p) for p in best_progs]
            else:
                X_train_st_all = np.zeros((X_train.shape[0], 0))
                X_test_st_all = np.zeros((X_test.shape[0], 0))
                st_names = []

            # --- 3. SEÇİM VE DEĞERLENDİRME ---
            for judge_name, judge_model in judges.items():
                print(f"\n   >>> JUDGE: {judge_name} <<<")

                # --- Sadece EF Seçimi ---
                selector_ef = clone(judge_model)
                selector_ef.fit(X_train_ef_all, y_train)
                imp_ef = selector_ef.feature_importances_

                # Aday listesi oluştur: (index, importance, name, type)
                ef_candidates_list = []
                for i, score in enumerate(imp_ef):
                    ef_candidates_list.append({
                        'idx': i,
                        'score': score,
                        'name': ef_names[i],
                        'type': 'EF'
                    })

                # Puana göre sırala
                ef_candidates_list.sort(key=lambda x: x['score'], reverse=True)

                sel_idx_ef = []
                final_ef_display = []  # Tablo için seçilenler
                seen_ef = set()

                # Seçim Döngüsü
                if use_threshold:
                    for item in ef_candidates_list:
                        if item['score'] >= min_importance and item['name'] not in seen_ef:
                            sel_idx_ef.append(item['idx'])
                            final_ef_display.append(item)
                            seen_ef.add(item['name'])
                    # Fallback (Eşik çok yüksekse en iyi 3)
                    if not sel_idx_ef:
                        for item in ef_candidates_list[:3]:
                            sel_idx_ef.append(item['idx'])
                            final_ef_display.append(item)
                            seen_ef.add(item['name'])
                else:
                    for item in ef_candidates_list[:10]:
                        sel_idx_ef.append(item['idx'])
                        final_ef_display.append(item)
                        seen_ef.add(item['name'])

                # --- TABLO YAZDIR ---
                print_feature_table(final_ef_display, f"[EF Selection] - {judge_name}")

                X_train_sel_ef = X_train_ef_all[:, sel_idx_ef]
                X_test_sel_ef = X_test_ef_all[:, sel_idx_ef]

                X_train_final_ef = np.hstack((X_train, X_train_sel_ef))
                X_test_final_ef = np.hstack((X_test, X_test_sel_ef))

                res_ef = evaluate_single_model(
                    X_train_final_ef, X_test_final_ef, y_train, y_test,
                    judge_name, judge_model, d_name, "2. EF"
                )
                if res_ef: all_results.append(res_ef)

                # --- Hybrid (EF + STGP) Ortak Havuz ---
                X_pool_train = np.hstack((X_train_ef_all, X_train_st_all))
                X_pool_test = np.hstack((X_test_ef_all, X_test_st_all))

                num_ef = X_train_ef_all.shape[1]
                num_st = X_train_st_all.shape[1]

                selector_joint = clone(judge_model)
                selector_joint.fit(X_pool_train, y_train)
                imp_joint = selector_joint.feature_importances_

                joint_candidates_list = []
                # EF'leri havuza ekle
                for i in range(num_ef):
                    joint_candidates_list.append({
                        'type': 'EF',
                        'idx': i,
                        'score': imp_joint[i],
                        'name': ef_names[i]
                    })
                # STGP'leri havuza ekle (offset: num_ef)
                for i in range(num_st):
                    joint_candidates_list.append({
                        'type': 'STGP',
                        'idx': i,
                        'score': imp_joint[num_ef + i],
                        'name': st_names[i]
                    })

                joint_candidates_list.sort(key=lambda x: x['score'], reverse=True)

                sel_idx_pool = []
                final_joint_display = []
                seen_joint = set()

                if use_threshold:
                    for item in joint_candidates_list:
                        if item['score'] >= min_importance and item['name'] not in seen_joint:
                            pool_idx = item['idx'] if item['type'] == 'EF' else (num_ef + item['idx'])
                            sel_idx_pool.append(pool_idx)
                            final_joint_display.append(item)
                            seen_joint.add(item['name'])
                    if not sel_idx_pool:
                        for item in joint_candidates_list[:5]:
                            pool_idx = item['idx'] if item['type'] == 'EF' else (num_ef + item['idx'])
                            sel_idx_pool.append(pool_idx)
                            final_joint_display.append(item)
                            seen_joint.add(item['name'])
                else:
                    count = 0
                    for item in joint_candidates_list:
                        if count >= 10: break
                        if item['name'] not in seen_joint:
                            pool_idx = item['idx'] if item['type'] == 'EF' else (num_ef + item['idx'])
                            sel_idx_pool.append(pool_idx)
                            final_joint_display.append(item)
                            seen_joint.add(item['name'])
                            count += 1

                # --- HİBRİT TABLO YAZDIR ---
                print_feature_table(final_joint_display, f"[Hybrid Selection] - {judge_name}")

                if sel_idx_pool:
                    X_train_sel_joint = X_pool_train[:, sel_idx_pool]
                    X_test_sel_joint = X_pool_test[:, sel_idx_pool]
                else:
                    X_train_sel_joint = np.zeros((X_train.shape[0], 0))
                    X_test_sel_joint = np.zeros((X_test.shape[0], 0))

                X_train_final_joint = np.hstack((X_train, X_train_sel_joint))
                X_test_final_joint = np.hstack((X_test, X_test_sel_joint))

                res_joint = evaluate_single_model(
                    X_train_final_joint, X_test_final_joint, y_train, y_test,
                    judge_name, judge_model, d_name, "3. STGP+EF"
                )
                if res_joint: all_results.append(res_joint)

        except Exception as e:
            print(f"Hata (Dataset ID {set_id_val}): {e}")
            import traceback
            traceback.print_exc()

    if not all_results: return None

    # Pivot Table
    df_results = pd.DataFrame(all_results)
    pivot_df = pd.pivot_table(df_results, index='Dataset', columns=['Algorithm', 'Method'], values='Score')
    pivot_df = pivot_df.fillna('-')

    try:
        desired_order = ["1. Base", "2. EF", "3. STGP+EF"]
        pivot_df = pivot_df.reindex(desired_order, axis=1, level=1)
    except:
        pass

    return pivot_df