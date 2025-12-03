import warnings
import openml
import pandas as pd
import numpy as np
import time
from catboost import CatBoostRegressor
from evolutionary_forest.utils import get_feature_importance
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, AdaBoostRegressor, GradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.base import clone
from sklearn.preprocessing import StandardScaler
from gplearn.genetic import SymbolicTransformer
from evolutionary_forest.forest import EvolutionaryForestRegressor
from Functions.Comparison import categorize_to_numeric

warnings.filterwarnings("ignore")


# --- GÖRSELLEŞTİRME YARDIMCILARI (TABLO İÇİN) ---

def truncate_string(s, length=85):
    """Uzun formülleri tabloyu bozmaması için kısaltır."""
    s = str(s)
    if len(s) > length:
        return s[:length - 3] + "..."
    return s


def print_feature_table(candidates, title, score_col_name="Importance/Score"):
    """Seçilen özellikleri hizalı ve çerçeveli bir tablo halinde basar."""
    if not candidates:
        print(f"\n   {title}")
        print("   " + "-" * 50 + "\n   (Seçim yapılmadı veya özellik bulunamadı)\n" + "   " + "-" * 50)
        return

    w_rank, w_score, w_expr = 6, 18, 90
    header = f"| {'Rank':^{w_rank}} | {score_col_name:^{w_score}} | {'Feature Formula / Name':<{w_expr}} |"
    divider = "+" + "-" * (w_rank + 2) + "+" + "-" * (w_score + 2) + "+" + "-" * (w_expr + 2) + "+"

    print(f"\n   {title}\n   {divider}\n   {header}\n   {divider}")

    for rank, item in enumerate(candidates, 1):
        c_expr = truncate_string(item['name'], length=w_expr)
        # Score bazen None olabilir (örn: comparative_analysis_five)
        if item['score'] is not None:
            s_val = f"{item['score']:.6f}"
        else:
            s_val = "N/A"

        print(f"   | {rank:^{w_rank}} | {s_val:^{w_score}} | {c_expr:<{w_expr}} |")

    print("   " + divider)


# --- DEĞERLENDİRME FONKSİYONLARI ---

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
            results.append({'Algorithm': regr_name, 'Dataset': dataset_name, 'Method': method_name, 'Score': e_score})
        except Exception as e:
            print(f"Hata ({regr_name} - {dataset_name}): {e}")
            results.append({'Algorithm': regr_name, 'Dataset': dataset_name, 'Method': method_name, 'Score': np.nan})
    return results


def evaluate_models_all_algorithims(X_train_new, X_test_new, y_train, y_test, dataset_name, method_name):
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
    for regr_name, model_instance in regressor_dict.items():
        try:
            enh_model = clone(model_instance)
            enh_model.fit(X_train_new, y_train)
            e_score = r2_score(y_test, enh_model.predict(X_test_new))
            results.append({'Algorithm': regr_name, 'Dataset': dataset_name, 'Method': method_name, 'Score': e_score})
        except Exception as e:
            print(f"Hata ({regr_name} - {dataset_name}): {e}")
            results.append({'Algorithm': regr_name, 'Dataset': dataset_name, 'Method': method_name, 'Score': np.nan})
    return results


# --- ANALİZ FONKSİYONLARI ---

def run_comparative_analysis_five(sets_id, use_threshold=True):
    all_results = []
    for set_id_val in sets_id:
        try:
            print(f"\n{'=' * 60}\nProcessing: Dataset ID {set_id_val}\n{'=' * 60}")
            dataset = openml.datasets.get_dataset(set_id_val)
            d_name = f"{dataset.name}"
            X, y, _, _ = dataset.get_data(dataset_format="dataframe", target=dataset.default_target_attribute)
            X = categorize_to_numeric(X)
            X_vals = np.nan_to_num(X.values.astype(np.float64))
            y_vals = y.values.astype(np.float64) if isinstance(y, pd.Series) else y.astype(np.float64)

            X_train, X_test, y_train, y_test = train_test_split(X_vals, y_vals, test_size=0.2, random_state=42)
            scaler_x, scaler_y = StandardScaler(), StandardScaler()
            X_train = scaler_x.fit_transform(X_train)
            X_test = scaler_x.transform(X_test)
            y_train = scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()
            y_test = scaler_y.transform(y_test.reshape(-1, 1)).ravel()

            print(f'Raw {d_name} columns: {X_train.shape[1]}')
            all_results.extend(evaluate_models(X_train, X_test, y_train, y_test, d_name, "Base"))

            # --- EF ---
            ef = EvolutionaryForestRegressor(random_state=42)
            ef.fit(X_train, y_train)
            X_train_ef = ef.transform(X_train)
            X_test_ef = ef.transform(X_test)

            limit = min(5, X_train_ef.shape[1]) if use_threshold else min(10, X_train_ef.shape[1])

            # Görselleştirme için isimleri al
            try:
                fi_dict = get_feature_importance(ef)
                ef_names = list(fi_dict.keys())[:limit]
            except:
                ef_names = [f"EF_Feat_{i}" for i in range(limit)]

            # Tablo verisi hazırla
            ef_display = [{'name': name, 'score': None} for name in ef_names]
            print_feature_table(ef_display, f"Selected Top-{limit} EF Features (Fixed Limit)",
                                score_col_name="Index Only")

            X_train_ef = X_train_ef[:, :limit]
            X_test_ef = X_test_ef[:, :limit]

            new_train_ef = np.hstack((X_train, X_train_ef))
            new_test_ef = np.hstack((X_test, X_test_ef))
            print(f'EF {d_name} columns: {new_train_ef.shape[1]}')
            all_results.extend(evaluate_models(new_train_ef, new_test_ef, y_train, y_test, d_name, "EF"))

            # --- STGP ---

            stgp = SymbolicTransformer(random_state=42)
            stgp.fit(X_train, y_train)
            X_train_st = stgp.transform(X_train)
            X_test_st = stgp.transform(X_test)

            if use_threshold:
                # Top 5 seçimi
                X_train_st = X_train_st[:, :5]
                X_test_st = X_test_st[:, :5]
                # İsimleri görselleştir
                st_names = [str(p) for p in stgp._best_programs[:5]] if stgp._best_programs else []
                st_display = [{'name': name, 'score': None} for name in st_names]
                print_feature_table(st_display, "Selected Top-5 STGP Features (Fixed Limit)",
                                    score_col_name="Index Only")

            new_train_stgp_ef = np.hstack((X_train, X_train_st, X_train_ef))
            new_test_stgp_ef = np.hstack((X_test, X_test_st, X_test_ef))
            print(f'STGP+EF {d_name} columns: {new_train_stgp_ef.shape[1]}')
            all_results.extend(evaluate_models(new_train_stgp_ef, new_test_stgp_ef, y_train, y_test, d_name, "STGP+EF"))

        except Exception as e:
            print(f"General Error (ID: {set_id_val}): {e}")
            continue

    if not all_results: return None
    return pd.pivot_table(pd.DataFrame(all_results), index='Algorithm', columns=['Dataset', 'Method'], values='Score')


def run_comparative_analysis_threshold_pearson(sets_id, use_threshold=True, min_importance_feature_ef=0.05,
                                               min_importance_feature_stgp=0.80):
    all_results = []
    for set_id_val in sets_id:
        try:
            print(f"\n{'=' * 60}\nProcessing: Dataset ID {set_id_val}\n{'=' * 60}")
            dataset = openml.datasets.get_dataset(set_id_val)
            d_name = f"{dataset.name}"
            X, y, _, _ = dataset.get_data(dataset_format="dataframe", target=dataset.default_target_attribute)
            X = categorize_to_numeric(X)
            X_vals = np.nan_to_num(X.values.astype(np.float64))
            y_vals = y.values.astype(np.float64) if isinstance(y, pd.Series) else y.astype(np.float64)
            X_train, X_test, y_train, y_test = train_test_split(X_vals, y_vals, test_size=0.2, random_state=42)
            scaler_x, scaler_y = StandardScaler(), StandardScaler()
            X_train = scaler_x.fit_transform(X_train)
            X_test = scaler_x.transform(X_test)
            y_train = scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()
            y_test = scaler_y.transform(y_test.reshape(-1, 1)).ravel()

            print(f'Raw {d_name} columns: {X_train.shape[1]}')
            all_results.extend(evaluate_models(X_train, X_test, y_train, y_test, d_name, "Base"))

            # --- EF Selection ---
            ef = EvolutionaryForestRegressor(random_state=42)
            ef.fit(X_train, y_train)
            X_train_ef = ef.transform(X_train)
            X_test_ef = ef.transform(X_test)

            feature_importance_dict = get_feature_importance(ef)

            if use_threshold:
                sorted_features = sorted(feature_importance_dict.items(), key=lambda x: x[1], reverse=True)
                selected = [(name, imp) for name, imp in sorted_features if imp >= min_importance_feature_ef]
                if not selected:
                    limit = min(5, len(sorted_features))
                    selected = sorted_features[:limit]
                    print(f"Fallback: Top {limit} EF features automatically selected.")

                # Tablo Yazdır
                ef_display = [{'name': name, 'score': imp} for name, imp in selected]
                print_feature_table(ef_display, f"EF Selection (Imp >= {min_importance_feature_ef})", "Importance")

                all_feature_names = list(feature_importance_dict.keys())
                raw_indices = [all_feature_names.index(name) for name, _ in selected]
                max_valid = X_train_ef.shape[1] - 1
                valid_indices = [idx for idx in raw_indices if idx <= max_valid]
                if not valid_indices:
                    valid_indices = list(range(min(5, X_train_ef.shape[1])))
                X_train_ef = X_train_ef[:, valid_indices]
                X_test_ef = X_test_ef[:, valid_indices]
            else:
                limit = min(10, X_train_ef.shape[1])
                X_train_ef = X_train_ef[:, :limit]
                X_test_ef = X_test_ef[:, :limit]

            new_train_ef = np.hstack((X_train, X_train_ef))
            new_test_ef = np.hstack((X_test, X_test_ef))
            print(f'EF {d_name} columns: {new_train_ef.shape[1]}')
            all_results.extend(evaluate_models(new_train_ef, new_test_ef, y_train, y_test, d_name, "EF"))

            # --- STGP Selection (Pearson) ---
            stgp = SymbolicTransformer(random_state=42)
            stgp.fit(X_train, y_train)

            if use_threshold:
                best_programs = stgp._best_programs
                if not best_programs:
                    X_train_st, X_test_st = np.zeros((X_train.shape[0], 0)), np.zeros((X_test.shape[0], 0))
                else:
                    sorted_programs = sorted([(p, p.raw_fitness_) for p in best_programs], key=lambda x: x[1],
                                             reverse=True)
                    selected_programs = [p for p, fit in sorted_programs if fit >= min_importance_feature_stgp]
                    if not selected_programs:
                        selected_programs = [p for p, _ in sorted_programs[:min(5, len(sorted_programs))]]
                        print("Fallback: Top STGP features selected.")

                    # Tablo Yazdır
                    st_display = [{'name': str(p), 'score': p.raw_fitness_} for p in selected_programs]
                    print_feature_table(st_display, f"STGP Selection (Pearson >= {min_importance_feature_stgp})",
                                        "Fitness")

                    output_train = [p.execute(X_train) for p in selected_programs]
                    output_test = [p.execute(X_test) for p in selected_programs]
                    X_train_st = np.column_stack(output_train) if output_train else np.zeros((X_train.shape[0], 0))
                    X_test_st = np.column_stack(output_test) if output_test else np.zeros((X_test.shape[0], 0))
            else:
                X_train_st = stgp.transform(X_train)
                X_test_st = stgp.transform(X_test)

            new_train_stgp_ef = np.hstack((X_train, X_train_st, X_train_ef))
            new_test_stgp_ef = np.hstack((X_test, X_test_st, X_test_ef))
            print(f'STGP+EF {d_name} columns: {new_train_stgp_ef.shape[1]}')
            all_results.extend(evaluate_models(new_train_stgp_ef, new_test_stgp_ef, y_train, y_test, d_name, "STGP+EF"))

        except Exception as e:
            print(f"General Error (ID: {set_id_val}): {e}")
            continue

    if not all_results: return None
    return pd.pivot_table(pd.DataFrame(all_results), index='Algorithm', columns=['Dataset', 'Method'], values='Score')


def run_comparative_analysis_threshold_permutation_imp(sets_id, use_threshold=True, min_importance_feature_ef=0.05,
                                                       min_mse_threshold=0.00001):
    # Ridge Judge
    return _run_permutation_analysis_generic(sets_id, use_threshold, min_importance_feature_ef, min_mse_threshold,
                                             judge_type='Ridge')


def run_comparative_analysis_threshold_permutation_imp_rf(sets_id, use_threshold=True, min_importance_feature_ef=0.05,
                                                          min_mse_threshold=0.00001):
    # Random Forest Judge
    return _run_permutation_analysis_generic(sets_id, use_threshold, min_importance_feature_ef, min_mse_threshold,
                                             judge_type='RF')


def run_comparative_analysis_threshold_permutation_imp_xgb(sets_id, use_threshold=True, min_importance_feature_ef=0.05,
                                                           min_mse_threshold=0.00001):
    # XGBoost Judge
    return _run_permutation_analysis_generic(sets_id, use_threshold, min_importance_feature_ef, min_mse_threshold,
                                             judge_type='XGB')


def run_comparative_analysis_threshold_permutation_imp_all_algorithims(sets_id, use_threshold=True,
                                                                       min_importance_feature_ef=0.05,
                                                                       min_mse_threshold=0.00001):
    # Same logic but uses evaluate_models_all_algorithims
    return _run_permutation_analysis_generic(sets_id, use_threshold, min_importance_feature_ef, min_mse_threshold,
                                             judge_type='Ridge', use_all_algos=True)


def _run_permutation_analysis_generic(sets_id, use_threshold, min_importance_feature_ef, min_mse_threshold,
                                      judge_type='Ridge', use_all_algos=False):
    """
    Genelleştirilmiş Permütasyon Önemi Analiz Fonksiyonu (Kod tekrarını önlemek için).
    judge_type: 'Ridge', 'RF', 'XGB'
    """
    all_results = []

    # Değerlendirme fonksiyonunu seç
    eval_func = evaluate_models_all_algorithims if use_all_algos else evaluate_models

    for set_id_val in sets_id:
        try:
            print(f"\n{'=' * 60}\nProcessing: Dataset ID {set_id_val}\n{'=' * 60}")
            dataset = openml.datasets.get_dataset(set_id_val)
            d_name = f"{dataset.name}"
            X, y, _, _ = dataset.get_data(dataset_format="dataframe", target=dataset.default_target_attribute)
            X = categorize_to_numeric(X)
            X_vals = np.nan_to_num(X.values.astype(np.float64))
            y_vals = y.values.astype(np.float64) if isinstance(y, pd.Series) else y.astype(np.float64)
            X_train, X_test, y_train, y_test = train_test_split(X_vals, y_vals, test_size=0.2, random_state=42)
            scaler_x, scaler_y = StandardScaler(), StandardScaler()
            X_train = scaler_x.fit_transform(X_train)
            X_test = scaler_x.transform(X_test)
            y_train = scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()
            y_test = scaler_y.transform(y_test.reshape(-1, 1)).ravel()

            print(f'Raw {d_name} columns: {X_train.shape[1]}')
            all_results.extend(eval_func(X_train, X_test, y_train, y_test, d_name, "Base"))

            # --- EF Selection ---
            ef = EvolutionaryForestRegressor(random_state=42)
            ef.fit(X_train, y_train)
            X_train_ef = ef.transform(X_train)
            X_test_ef = ef.transform(X_test)

            if use_threshold:
                fi_dict = get_feature_importance(ef)
                sorted_ef = sorted(fi_dict.items(), key=lambda x: x[1], reverse=True)
                selected_ef = [(n, s) for n, s in sorted_ef if s >= min_importance_feature_ef]

                if not selected_ef:
                    selected_ef = sorted_ef[:min(5, len(sorted_ef))]
                    print("Fallback: Top EF features selected.")

                # Tablo Yazdır
                ef_display = [{'name': n, 'score': s} for n, s in selected_ef]
                print_feature_table(ef_display, f"EF Selection (Imp >= {min_importance_feature_ef})", "Importance")

                all_names = list(fi_dict.keys())
                raw_idx = [all_names.index(n) for n, _ in selected_ef]
                valid_idx = [i for i in raw_idx if i < X_train_ef.shape[1]]
                if not valid_idx: valid_idx = list(range(min(5, X_train_ef.shape[1])))
                X_train_ef = X_train_ef[:, valid_idx]
                X_test_ef = X_test_ef[:, valid_idx]
            else:
                limit = min(10, X_train_ef.shape[1])
                X_train_ef = X_train_ef[:, :limit]
                X_test_ef = X_test_ef[:, :limit]

            new_train_ef = np.hstack((X_train, X_train_ef))
            new_test_ef = np.hstack((X_test, X_test_ef))
            print(f'EF {d_name} columns: {new_train_ef.shape[1]}')
            all_results.extend(eval_func(new_train_ef, new_test_ef, y_train, y_test, d_name, "EF"))

            # --- STGP Selection (Permutation) ---
            stgp = SymbolicTransformer(random_state=42)
            stgp.fit(X_train, y_train)

            X_train_st, X_test_st = np.zeros((X_train.shape[0], 0)), np.zeros((X_test.shape[0], 0))

            if use_threshold:
                best_programs = stgp._best_programs
                if best_programs:
                    cand_out = [p.execute(X_train) for p in best_programs]
                    X_cand = np.column_stack(cand_out)

                    # Hakem Model Seçimi
                    if judge_type == 'Ridge':
                        judge = Ridge(alpha=1.0, random_state=42)
                    elif judge_type == 'RF':
                        judge = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
                    elif judge_type == 'XGB':
                        judge = XGBRegressor(n_jobs=1, n_estimators=200, verbosity=0, random_state=42)

                    judge.fit(X_cand, y_train)

                    # [Image of permutation importance concept]

                    perm = permutation_importance(judge, X_cand, y_train, n_repeats=5, random_state=42,
                                                  scoring='neg_mean_squared_error')

                    pairs = sorted(zip(best_programs, perm.importances_mean), key=lambda x: x[1], reverse=True)

                    selected_progs = []
                    seen = set()
                    for p, s in pairs:
                        if s >= min_mse_threshold:
                            p_str = str(p)
                            if p_str not in seen:
                                selected_progs.append((p, s))
                                seen.add(p_str)

                    if not selected_progs:
                        print("Fallback: Selecting top 5 unique STGP features.")
                        seen = set()
                        for p, s in pairs:
                            if str(p) not in seen:
                                selected_progs.append((p, s))
                                seen.add(str(p))
                                if len(selected_progs) >= 5: break

                    # Tablo Yazdır
                    st_display = [{'name': str(p), 'score': s} for p, s in selected_progs]
                    print_feature_table(st_display, f"STGP Selection ({judge_type} MSE >= {min_mse_threshold})",
                                        "MSE Contribution")

                    final_p = [x[0] for x in selected_progs]
                    out_tr = [p.execute(X_train) for p in final_p]
                    out_te = [p.execute(X_test) for p in final_p]
                    if out_tr:
                        X_train_st = np.column_stack(out_tr)
                        X_test_st = np.column_stack(out_te)
                else:
                    X_train_st = stgp.transform(X_train)
                    X_test_st = stgp.transform(X_test)

                new_train_stgp_ef = np.hstack((X_train, X_train_st, X_train_ef))
                new_test_stgp_ef = np.hstack((X_test, X_test_st, X_test_ef))
                print(f'STGP+EF {d_name} columns: {new_train_stgp_ef.shape[1]}')
                all_results.extend(eval_func(new_train_stgp_ef, new_test_stgp_ef, y_train, y_test, d_name, "STGP+EF"))


        except Exception as e:

            print(f"General Error (ID: {set_id_val}): {e}")

            import traceback

            traceback.print_exc()

            continue

    if not all_results: return None
    return pd.pivot_table(pd.DataFrame(all_results), index='Algorithm', columns=['Dataset', 'Method'],
                          values='Score')
