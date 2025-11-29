import warnings
import openml
import pandas as pd
import numpy as np
from evolutionary_forest.utils import get_feature_importance
from sklearn.ensemble import RandomForestRegressor
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


def run_comparative_analysis_five(sets_id, use_threshold=True):
    all_results = []

    for set_id_val in sets_id:
        try:
            print(f"Processing: Dataset ID {set_id_val}...")
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

            all_results.extend(evaluate_models(X_train, X_test,
                                               y_train, y_test, d_name, "Base"))

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

            stgp = SymbolicTransformer(random_state=42)
            stgp.fit(X_train, y_train)

            X_train_st = stgp.transform(X_train)
            X_test_st = stgp.transform(X_test)

            if use_threshold:
                X_train_st = X_train_st[:, :5]
                X_test_st = X_test_st[:, :5]

            new_train_stgp_ef = np.hstack((X_train, X_train_st, X_train_ef))
            new_test_stgp_ef = np.hstack((X_test, X_test_st, X_test_ef))

            print(f'STGP+EF {d_name} columns: {new_train_stgp_ef.shape[1]}')

            all_results.extend(evaluate_models(new_train_stgp_ef, new_test_stgp_ef,
                                               y_train, y_test, d_name, "STGP+EF"))

        except Exception as e:
            print(f"General Error (ID: {set_id_val}): {e}")
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


def run_comparative_analysis_threshold_pearson(sets_id, use_threshold=True, min_importance_feature_ef=0.05, min_importance_feature_stgp=0.80):
    all_results = []

    for set_id_val in sets_id:
        try:
            print(f"Processing: Dataset ID {set_id_val}...")
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

            all_results.extend(evaluate_models(X_train, X_test,
                                               y_train, y_test, d_name, "Base"))

            # --- Evolutionary Forest (EF) ---
            ef = EvolutionaryForestRegressor(random_state=42)
            ef.fit(X_train, y_train)

            X_train_ef = ef.transform(X_train)
            X_test_ef = ef.transform(X_test)

            feature_importance_dict = get_feature_importance(ef)

            if use_threshold:
                sorted_features = sorted(feature_importance_dict.items(), key=lambda x: x[1], reverse=True)
                selected = [(name, imp) for name, imp in sorted_features if imp >= min_importance_feature_ef]

                if not selected:
                    print(f"Warning (EF): No features found with importance >= {min_importance_feature_ef}.")
                    limit = min(5, len(sorted_features))
                    selected = sorted_features[:limit]
                    print(f"Fallback: Top {limit} features were automatically selected.")

                print("\n" + "=" * 80)
                print(f"Selected EF Features (Total: {len(selected)})")
                print(f"{'Rank':<4} | {'Importance':<15} | {'Formula/Feature'}")
                print("-" * 80)

                for i, (name, imp) in enumerate(selected):
                    display_name = (name[:57] + '...') if len(name) > 60 else name
                    print(f"{i + 1:<4} | {imp:.5f}          | {display_name}")

                print("=" * 80 + "\n")

                all_feature_names = list(feature_importance_dict.keys())

                raw_indices = [all_feature_names.index(name) for name, _ in selected]

                max_valid_index = X_train_ef.shape[1] - 1

                valid_indices = [idx for idx in raw_indices if idx <= max_valid_index]

                if len(valid_indices) < len(raw_indices):
                    print(f"Info: Dropped {len(raw_indices) - len(valid_indices)} features that were out of bounds.")

                if not valid_indices:
                    print("Error: All selected features were out of bounds! Reverting to top available columns.")
                    limit = min(5, X_train_ef.shape[1])
                    valid_indices = list(range(limit))

                X_train_ef = X_train_ef[:, valid_indices]
                X_test_ef = X_test_ef[:, valid_indices]
            else:
                limit = min(10, X_train_ef.shape[1])
                X_train_ef = X_train_ef[:, :limit]
                X_test_ef = X_test_ef[:, :limit]

            new_train_ef = np.hstack((X_train, X_train_ef))
            new_test_ef = np.hstack((X_test, X_test_ef))

            print(f'EF {d_name} columns: {new_train_ef.shape[1]}')

            all_results.extend(evaluate_models(new_train_ef, new_test_ef,
                                               y_train, y_test, d_name, "EF"))

            # --- Symbolic Transformer (STGP) ---
            stgp = SymbolicTransformer(random_state=42)
            stgp.fit(X_train, y_train)

            X_train_st = stgp.transform(X_train)
            X_test_st = stgp.transform(X_test)

            if use_threshold:
                best_programs = stgp._best_programs

                if not best_programs:
                    print("Warning: No programs found in STGP model. Skipping STGP feature selection.")
                    return np.zeros((X_train.shape[0], 0)), np.zeros((X_test.shape[0], 0))

                program_fitness_pairs = [(prog, prog.raw_fitness_) for prog in best_programs]

                sorted_programs = sorted(program_fitness_pairs, key=lambda x: x[1], reverse=True)

                selected_programs = [prog for prog, fit in sorted_programs if fit >= min_importance_feature_stgp]

                if not selected_programs:
                    print(f"Warning: No features passed the fitness threshold ({min_importance_feature_stgp}).")
                    limit = min(5, len(sorted_programs))
                    selected_programs = [prog for prog, _ in sorted_programs[:limit]]
                    print(f"Fallback: Top {limit} features were automatically selected.")

                print("\n" + "=" * 60)
                print(f"Selected STGP Features (Total: {len(selected_programs)})")
                print(f"{'Rank':<4} | {'Fitness (Pearson)':<18} | {'Formula'}")
                print("-" * 60)

                for i, prog in enumerate(selected_programs):
                    print(f"{i + 1:<4} | {prog.raw_fitness_:.5f}            | {str(prog)}")

                print("=" * 60 + "\n")

                output_train = []
                output_test = []

                for prog in selected_programs:
                    output_train.append(prog.execute(X_train))
                    output_test.append(prog.execute(X_test))

                if not output_train:
                    # Same note as above regarding 'return' in a loop
                    print("Error: STGP feature execution failed. Returning empty arrays.")
                    return np.zeros((X_train.shape[0], 0)), np.zeros((X_test.shape[0], 0))

                X_train_st = np.column_stack(output_train)
                X_test_st = np.column_stack(output_test)

            new_train_stgp_ef = np.hstack((X_train, X_train_st, X_train_ef))
            new_test_stgp_ef = np.hstack((X_test, X_test_st, X_test_ef))

            print(f'STGP+EF {d_name} columns: {new_train_stgp_ef.shape[1]}')

            all_results.extend(evaluate_models(new_train_stgp_ef, new_test_stgp_ef,
                                               y_train, y_test, d_name, "STGP+EF"))

        except Exception as e:
            print(f"General Error (ID: {set_id_val}): {e}")
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


def run_comparative_analysis_threshold_permutation_imp(sets_id, use_threshold=True, min_importance_feature_ef=0.05, min_mse_threshold = 0.00001):
    """
    STGP için Ridge + Permutation Importance (MSE Katkısı) yöntemini kullanır.
    EF için MDI (Mean Decrease Impurity) yöntemini kullanır.
    """
    all_results = []

    for set_id_val in sets_id:
        try:
            print(f"Processing: Dataset ID {set_id_val}...")
            dataset = openml.datasets.get_dataset(set_id_val)
            d_name = f"{dataset.name}"

            # Veri Yükleme ve Ön İşleme
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

            # --- Base Model Evaluation ---
            all_results.extend(evaluate_models(X_train, X_test, y_train, y_test, d_name, "Base"))

            # ==========================================
            # 1. Evolutionary Forest (EF) - Feature Selection
            # ==========================================
            ef = EvolutionaryForestRegressor(random_state=42)
            ef.fit(X_train, y_train)

            X_train_ef = ef.transform(X_train)
            X_test_ef = ef.transform(X_test)

            if use_threshold:
                feature_importance_dict = get_feature_importance(ef)
                sorted_features = sorted(feature_importance_dict.items(), key=lambda x: x[1], reverse=True)

                # Threshold'a göre seç
                selected_ef = [(name, imp) for name, imp in sorted_features if imp >= min_importance_feature_ef]

                if not selected_ef:
                    print(f"Warning (EF): No features found with importance >= {min_importance_feature_ef}.")
                    limit = min(5, len(sorted_features))
                    selected_ef = sorted_features[:limit]
                    print(f"Fallback: Top {limit} features were automatically selected.")

                # EF Seçilenleri Listele
                print("\n" + "=" * 80)
                print(f"Selected EF Features (Total: {len(selected_ef)})")
                print(f"{'Rank':<4} | {'Importance':<15} | {'Formula/Feature'}")
                print("-" * 80)
                for i, (name, imp) in enumerate(selected_ef):
                    display_name = (name[:57] + '...') if len(name) > 60 else name
                    print(f"{i + 1:<4} | {imp:.5f}          | {display_name}")
                print("=" * 80 + "\n")

                # İndeksleri Bul ve Filtrele
                all_feature_names = list(feature_importance_dict.keys())
                raw_indices = [all_feature_names.index(name) for name, _ in selected_ef]
                max_valid_index = X_train_ef.shape[1] - 1
                valid_indices = [idx for idx in raw_indices if idx <= max_valid_index]

                if not valid_indices:
                    limit = min(5, X_train_ef.shape[1])
                    valid_indices = list(range(limit))

                X_train_ef = X_train_ef[:, valid_indices]
                X_test_ef = X_test_ef[:, valid_indices]
            else:
                # Threshold yoksa ilk 10'u al (örnek kısıtlama)
                limit = min(10, X_train_ef.shape[1])
                X_train_ef = X_train_ef[:, :limit]
                X_test_ef = X_test_ef[:, :limit]

            # EF Sonuçlarını Değerlendir
            new_train_ef = np.hstack((X_train, X_train_ef))
            new_test_ef = np.hstack((X_test, X_test_ef))
            print(f'EF {d_name} columns: {new_train_ef.shape[1]}')
            all_results.extend(evaluate_models(new_train_ef, new_test_ef, y_train, y_test, d_name, "EF"))

            # ==========================================
            # 2. Symbolic Transformer (STGP) - Ridge MSE Selection
            # ==========================================
            stgp = SymbolicTransformer(random_state=42)
            stgp.fit(X_train, y_train)

            # Başlangıçta boş array atayalım
            X_train_st = np.zeros((X_train.shape[0], 0))
            X_test_st = np.zeros((X_test.shape[0], 0))

            # Eşik Değeri (Örn: 0.0001 gibi çok küçük de olsa pozitif bir katkı veya daha agresif bir değer)


            if use_threshold:
                best_programs = stgp._best_programs

                if not best_programs:
                    print("Warning: No programs found in STGP model. Skipping STGP feature selection.")
                else:
                    # Aday programları çalıştır (Geçici Matris)
                    candidate_outputs = [prog.execute(X_train) for prog in best_programs]
                    X_train_candidates = np.column_stack(candidate_outputs)

                    # Ridge Modelini Eğit (Hakem)
                    ridge_judge = Ridge(alpha=1.0, random_state=42)
                    ridge_judge.fit(X_train_candidates, y_train)

                    # Permütasyon Önemini Hesapla (MSE Katkısı)
                    print("Calculating Ridge MSE Contribution for STGP features...")
                    perm_result = permutation_importance(
                        ridge_judge,
                        X_train_candidates,
                        y_train,
                        n_repeats=5,
                        random_state=42,
                        scoring='neg_mean_squared_error'
                    )

                    # (Program, Score) çiftlerini oluştur
                    program_score_pairs = []
                    for prog, score in zip(best_programs, perm_result.importances_mean):
                        program_score_pairs.append((prog, score))

                    # Puana göre sırala (En yüksek katkıdan en düşüğe)
                    sorted_stgp = sorted(program_score_pairs, key=lambda x: x[1], reverse=True)

                    # --- SEÇİM MANTIĞI GÜNCELLEMESİ ---
                    selected_stgp_pairs = []
                    seen_formulas = set()  # Yapısal benzerliği kontrol etmek için (String formüller)

                    # 1. Aşama: Eşik üstü ve Benzersiz olanları seç
                    for prog, score in sorted_stgp:
                        if score >= min_mse_threshold:
                            prog_str = str(prog)
                            if prog_str not in seen_formulas:
                                selected_stgp_pairs.append((prog, score))
                                seen_formulas.add(prog_str)

                    # 2. Aşama: Fallback (Eğer hiç seçim yapılamadıysa)
                    if not selected_stgp_pairs:
                        print(f"Warning: No unique STGP features passed the MSE threshold ({min_mse_threshold}).")
                        print("Fallback: Selecting top 5 unique features based on rank.")

                        seen_formulas = set()  # Kümeyi sıfırla
                        limit = 5

                        for prog, score in sorted_stgp:
                            prog_str = str(prog)
                            if prog_str not in seen_formulas:
                                selected_stgp_pairs.append((prog, score))
                                seen_formulas.add(prog_str)

                                if len(selected_stgp_pairs) >= limit:
                                    break

                    # Seçilenleri Listele
                    print("\n" + "=" * 80)
                    print(f"Selected Unique STGP Features (Total: {len(selected_stgp_pairs)})")
                    print(f"{'Rank':<4} | {'MSE Contribution':<18} | {'Formula'}")
                    print("-" * 80)
                    for i, (prog, score) in enumerate(selected_stgp_pairs):
                        display_str = str(prog)
                        if len(display_str) > 50: display_str = display_str[:47] + "..."
                        print(f"{i + 1:<4} | {score:.6f}            | {display_str}")
                    print("=" * 80 + "\n")

                    # Final Dönüşüm (Execute)
                    final_progs = [p[0] for p in selected_stgp_pairs]

                    output_train = [prog.execute(X_train) for prog in final_progs]
                    output_test = [prog.execute(X_test) for prog in final_progs]

                    if output_train:
                        X_train_st = np.column_stack(output_train)
                        X_test_st = np.column_stack(output_test)
            else:
                # Threshold yoksa gplearn standart transform kullan
                X_train_st = stgp.transform(X_train)
                X_test_st = stgp.transform(X_test)

            # ==========================================
            # 3. Final Birleştirme
            # ==========================================
            new_train_stgp_ef = np.hstack((X_train, X_train_st, X_train_ef))
            new_test_stgp_ef = np.hstack((X_test, X_test_st, X_test_ef))

            print(f'STGP(Ridge-Selected)+EF {d_name} columns: {new_train_stgp_ef.shape[1]}')

            all_results.extend(evaluate_models(new_train_stgp_ef, new_test_stgp_ef,
                                               y_train, y_test, d_name, "STGP+EF"))

        except Exception as e:
            print(f"General Error (ID: {set_id_val}): {e}")
            import traceback
            traceback.print_exc()  # Hata detayını görmek için
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