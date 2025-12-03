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


# --- ANA MANTIK FONKSİYONLARI ---

def evaluate_models(X_train, X_test, y_train, y_test, dataset_name, method_name):
    """Verilen veri seti üzerinde RF ve XGBoost modellerini eğitip test eder."""
    regressors = {
        'RF': RandomForestRegressor(n_estimators=200, n_jobs=-1, random_state=42),
        'XGBoost': XGBRegressor(n_jobs=1, n_estimators=200, verbosity=0, random_state=42),
    }
    results = []
    for name, model in regressors.items():
        try:
            m = clone(model)
            m.fit(X_train, y_train)
            score = r2_score(y_test, m.predict(X_test))
            results.append({'Algorithm': name, 'Dataset': dataset_name, 'Method': method_name, 'Score': score})
        except Exception as e:
            print(f"Hata ({name}-{dataset_name}): {e}")
            results.append({'Algorithm': name, 'Dataset': dataset_name, 'Method': method_name, 'Score': np.nan})
    return results


def select_candidates_logic(candidate_list, use_threshold, threshold, top_k=10):
    """
    Tekrarlayan seçim mantığını (Threshold vs Top-K) yöneten yardımcı fonksiyon.
    İşlevsellik değişmedi, sadece kod tekrarı önlendi.
    """
    # Puana göre sırala
    candidate_list.sort(key=lambda x: x['score'], reverse=True)

    selected_items = []
    seen_names = set()

    if use_threshold:
        # Threshold Modu
        for item in candidate_list:
            if item['score'] >= threshold and item['name'] not in seen_names:
                selected_items.append(item)
                seen_names.add(item['name'])

        # Fallback (Eğer hiçbiri geçemezse en iyi 5)
        if not selected_items:
            for item in candidate_list[:5]:
                if item['name'] not in seen_names:
                    selected_items.append(item)
                    seen_names.add(item['name'])
    else:
        # Top-K Modu
        count = 0
        for item in candidate_list:
            if count >= top_k: break
            if item['name'] not in seen_names:
                selected_items.append(item)
                seen_names.add(item['name'])
                count += 1

    return selected_items


def run_comparative_analysis_rf_selection(sets_id, use_threshold=True, min_importance=0.05):
    """
    Optimize edilmiş akış:
    1. Base Model
    2. EF Üretimi -> RF ile Seçim -> EF Değerlendirmesi
    3. STGP Üretimi -> (Seçilmiş EF + STGP) Havuzu -> RF ile Seçim -> Hybrid Değerlendirmesi
    """
    all_results = []

    for set_id_val in sets_id:
        print(f"\n{'#' * 60}\nProcessing: Dataset ID {set_id_val}\n{'#' * 60}")

        # --- 1. VERİ YÜKLEME VE ÖN İŞLEME ---
        try:
            X = None
            for attempt in range(3):
                try:
                    dataset = openml.datasets.get_dataset(set_id_val, download_data=True)
                    X, y, _, _ = dataset.get_data(dataset_format="dataframe", target=dataset.default_target_attribute)
                    d_name = dataset.name
                    break
                except Exception as e:
                    if attempt < 2: time.sleep(5)

            if X is None: continue

            X = categorize_to_numeric(X)
            X_vals = np.nan_to_num(X.values.astype(np.float64))
            y_vals = y.values.astype(np.float64) if isinstance(y, pd.Series) else y.astype(np.float64)

            X_train, X_test, y_train, y_test = train_test_split(X_vals, y_vals, test_size=0.2, random_state=42)

            scaler_x, scaler_y = StandardScaler(), StandardScaler()
            X_train = scaler_x.fit_transform(X_train)
            X_test = scaler_x.transform(X_test)
            y_train = scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()
            y_test = scaler_y.transform(y_test.reshape(-1, 1)).ravel()

            print(f'Dataset: {d_name} | Raw Columns: {X_train.shape[1]}')

            # Base Model
            all_results.extend(evaluate_models(X_train, X_test, y_train, y_test, d_name, "1. Base"))

            # --- 2. EVOLUTIONARY FOREST (EF) AŞAMASI ---
            print("\n> [Step 1] Running Evolutionary Forest Generation & Selection...")
            ef = EvolutionaryForestRegressor(random_state=42)
            ef.fit(X_train, y_train)

            X_train_ef_raw = ef.transform(X_train)
            X_test_ef_raw = ef.transform(X_test)

            # İsimlendirme
            try:
                fi_dict = get_feature_importance(ef)
                ef_names = [str(k) for k in fi_dict.keys()]
                # Boyut düzeltme
                if len(ef_names) < X_train_ef_raw.shape[1]:
                    ef_names += [f"EF_{i}" for i in range(len(ef_names), X_train_ef_raw.shape[1])]
                elif len(ef_names) > X_train_ef_raw.shape[1]:
                    ef_names = ef_names[:X_train_ef_raw.shape[1]]
            except:
                ef_names = [f"EF_{i}" for i in range(X_train_ef_raw.shape[1])]

            # RF ile Önem Derecesi (EF Adayları İçin)
            rf_ef = RandomForestRegressor(n_estimators=100, n_jobs=-1, random_state=42)
            rf_ef.fit(X_train_ef_raw, y_train)

            # Aday Listesi Hazırla
            ef_candidates = [
                {'name': ef_names[i], 'score': rf_ef.feature_importances_[i], 'idx': i, 'type': 'EF'}
                for i in range(len(ef_names))
            ]

            # Seçim Fonksiyonunu Çağır
            selected_ef_items = select_candidates_logic(ef_candidates, use_threshold, min_importance)
            selected_ef_indices = [item['idx'] for item in selected_ef_items]
            selected_ef_names_final = [item['name'] for item in selected_ef_items]

            print_feature_table(selected_ef_items, f"Step 1: EF Selection (Threshold >= {min_importance})")

            # Seçilen EF Özelliklerini Filtrele
            X_train_ef = X_train_ef_raw[:, selected_ef_indices]
            X_test_ef = X_test_ef_raw[:, selected_ef_indices]

            # EF Modeli Değerlendir
            all_results.extend(evaluate_models(
                np.hstack((X_train, X_train_ef)),
                np.hstack((X_test, X_test_ef)),
                y_train, y_test, d_name, "2. EF"
            ))

            # --- 3. STGP & ORTAK HAVUZ (JOINT) AŞAMASI ---

            print("\n> [Step 2] Running STGP & Joint Selection (EF Survivors + STGP)...")
            stgp = SymbolicTransformer(random_state=42)
            stgp.fit(X_train, y_train)

            # STGP Çıktılarını Hazırla
            if stgp._best_programs:

                X_train_st = np.column_stack([p.execute(X_train) for p in stgp._best_programs])
                X_test_st = np.column_stack([p.execute(X_test) for p in stgp._best_programs])
                st_names = [str(p) for p in stgp._best_programs]
            else:
                X_train_st, X_test_st = np.zeros((X_train.shape[0], 0)), np.zeros((X_test.shape[0], 0))
                st_names = []

            # Ortak Havuz (Selected EF + All STGP)
            X_pool_train = np.hstack((X_train_ef, X_train_st))
            num_ef_sel = X_train_ef.shape[1]
            num_st = X_train_st.shape[1]

            print(f"  Joint Pool: {num_ef_sel} Selected EF + {num_st} STGP candidates.")

            # RF ile Önem Derecesi (Ortak Havuz İçin)
            rf_joint = RandomForestRegressor(n_estimators=100, n_jobs=-1, random_state=42)
            rf_joint.fit(X_pool_train, y_train)
            joint_imps = rf_joint.feature_importances_

            # Joint Aday Listesi
            joint_candidates = []
            # EF Adayları (Havuzun ilk kısmı)
            for i in range(num_ef_sel):
                joint_candidates.append({
                    'type': 'EF', 'idx': i, 'score': joint_imps[i], 'name': selected_ef_names_final[i]
                })
            # STGP Adayları (Havuzun ikinci kısmı, indeksler offsetli değil, kendi listesine göre)
            for i in range(num_st):
                joint_candidates.append({
                    'type': 'STGP', 'idx': i, 'score': joint_imps[num_ef_sel + i], 'name': st_names[i]
                })

            # Seçim Fonksiyonunu Çağır
            selected_joint_items = select_candidates_logic(joint_candidates, use_threshold, min_importance)

            print_feature_table(selected_joint_items, f"Step 2: Joint Selection (Threshold >= {min_importance})")

            # Final Matrisleri Oluştur (Seçilenleri ayıkla)
            # EF İçin Seçilenler (Joint havuzdaki 'EF' tipli olanlar)
            final_ef_idx = sorted([item['idx'] for item in selected_joint_items if item['type'] == 'EF'])
            # STGP İçin Seçilenler (Joint havuzdaki 'STGP' tipli olanlar)
            final_st_idx = sorted([item['idx'] for item in selected_joint_items if item['type'] == 'STGP'])

            print(f"Result: {len(final_ef_idx)} from EF, {len(final_st_idx)} from STGP kept.")

            X_train_final = np.hstack((
                X_train,
                X_train_ef[:, final_ef_idx] if final_ef_idx else np.zeros((X_train.shape[0], 0)),
                X_train_st[:, final_st_idx] if final_st_idx else np.zeros((X_train.shape[0], 0))
            ))
            X_test_final = np.hstack((
                X_test,
                X_test_ef[:, final_ef_idx] if final_ef_idx else np.zeros((X_test.shape[0], 0)),
                X_test_st[:, final_st_idx] if final_st_idx else np.zeros((X_test.shape[0], 0))
            ))

            # Hybrid Model Değerlendir
            all_results.extend(evaluate_models(X_train_final, X_test_final, y_train, y_test, d_name, "3. STGP+EF"))

        except Exception as e:
            print(f"Critical Error (Dataset {set_id_val}): {e}")
            import traceback;
            traceback.print_exc()

    if not all_results: return None

    # Pivot Table
    pivot_df = pd.pivot_table(pd.DataFrame(all_results), index='Dataset', columns=['Algorithm', 'Method'],
                              values='Score').fillna('-')
    try:
        pivot_df = pivot_df.reindex(["1. Base", "2. EF", "3. STGP+EF"], axis=1, level=1)
    except:
        pass

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


def run_hybrid_comparative_analysis(sets_id, use_threshold=True, min_importance=0.05):
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