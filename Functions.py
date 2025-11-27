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
    # 1. Özellikleri önem sırasına göre diz (Büyükten küçüğe)
    sorted_features = sorted(feature_importance_dict.items(), key=lambda x: x[1], reverse=True)

    # 2. Threshold'a göre filtrele
    selected = [(name, imp) for name, imp in sorted_features if imp >= min_importance_threshold]

    # --- FALLBACK MEKANİZMASI ---
    # Eğer threshold'u geçen özellik sayısı 5'ten azsa, otomatik olarak en iyi 5 taneyi al.
    min_feature_count = 5

    if len(selected) < min_feature_count:
        print(f"Uyarı (EF): Threshold ({min_importance_threshold}) geçen özellik sayısı az ({len(selected)}).")
        # Mevcut sayı 5'ten azsa hepsini, çoksa en iyi 5'ini al
        limit = min(min_feature_count, len(sorted_features))
        selected = sorted_features[:limit]
        print(f"Top {limit} features were automatically selected.")

    # --- RAPORLAMA (YENİ KISIM) ---
    # Seçilen formülleri tablo halinde gösterir
    print("\n" + "=" * 80)
    print(f"Seçilen EF Özellikleri (Toplam: {len(selected)})")
    print(f"{'No':<4} | {'Önem (Imp)':<15} | {'Formül / Özellik'}")
    print("-" * 80)

    for i, (name, imp) in enumerate(selected):
        # Formül çok uzunsa tablo bozulmasın diye ilk 60 karakteri gösterip ... koyabiliriz
        display_name = (name[:57] + '...') if len(name) > 60 else name
        print(f"{i + 1:<4} | {imp:.5f}          | {display_name}")

    print("=" * 80 + "\n")
    # -----------------------------

    # 3. İndeksleri belirle
    all_feature_names = list(feature_importance_dict.keys())

    # Feature ismi (key) ile listedeki sırasını eşleştiriyoruz
    raw_indices = [all_feature_names.index(name) for name, _ in selected]

    # --- KRİTİK DÜZELTME (SAFETY CHECK) ---
    # Matris boyutunu kontrol et
    max_valid_index = X_train_transformed.shape[1] - 1

    # Sadece matris boyut sınırları içinde kalan indeksleri al
    valid_indices = [idx for idx in raw_indices if idx <= max_valid_index]

    # Güvenlik kontrolü sonrası eldeki indeks sayısı azaldıysa bilgi ver
    if len(valid_indices) < len(raw_indices):
        print(f"Dropped {len(raw_indices) - len(valid_indices)} features that were out of bounds.")

    # Eğer tüm indeksler sınır dışı kaldıysa (çok nadir), en baştaki geçerli sütunları al
    if not valid_indices:
        print("All selected features were out of bounds! Reverting to top available columns.")
        # Mevcut matrisin ilk 5 (veya daha az) sütununu al
        limit = min(min_feature_count, X_train_transformed.shape[1])
        valid_indices = list(range(limit))

    # 4. Seçimi Yap
    X_train_selected = X_train_transformed[:, valid_indices]
    X_test_selected = X_test_transformed[:, valid_indices]

    print(f"Selected {len(valid_indices)} out of {len(all_feature_names)} EF features (Safe Mode).")

    return X_train_selected, X_test_selected


def select_st_features_by_threshold(stgp_model, X_train: np.ndarray, X_test: np.ndarray,
                                    min_threshold: float = 0.6) -> Tuple[np.ndarray, np.ndarray]:
    best_programs = stgp_model._best_programs

    if not best_programs:
        print("Uyarı: STGP modelinde kayıtlı program bulunamadı. Boş dizi dönülüyor.")
        return np.zeros((X_train.shape[0], 0)), np.zeros((X_test.shape[0], 0))

    # 2. Programları fitness (başarı) değerine göre eşleştir
    program_fitness_pairs = [(prog, prog.raw_fitness_) for prog in best_programs]

    # 3. Büyükten küçüğe sırala
    sorted_programs = sorted(program_fitness_pairs, key=lambda x: x[1], reverse=True)

    # 4. Eşik değerine (threshold) göre filtrele
    selected_programs = [prog for prog, fit in sorted_programs if fit >= min_threshold]

    # --- GÜVENLİK VE FALLBACK MEKANİZMASI ---
    min_feature_count = 5

    # Not: User snippet'ındaki 'if not selected_programs' yerine daha güvenli olan
    # 'sayı yetersizse' kontrolünü (önceki konuşmamızdaki gibi) kullanmak daha iyidir.
    if len(selected_programs) < min_feature_count:
        print(f"Uyarı: Threshold'u geçen özellik sayısı yetersiz veya yok ({len(selected_programs)}).")
        # Mevcut program sayısı 5'ten azsa hepsini, çoksa en iyi 5'ini al
        limit = min(min_feature_count, len(sorted_programs))
        selected_programs = [prog for prog, _ in sorted_programs[:limit]]
        print(f"Top {limit} features were automatically selected.")

    # --- RAPORLAMA: Formülleri ve Puanları Göster (YENİ KISIM) ---
    print("\n" + "=" * 60)
    print(f"Seçilen STGP Özellikleri (Toplam: {len(selected_programs)})")
    print(f"{'No':<4} | {'Fitness (Pearson)':<18} | {'Formül'}")
    print("-" * 60)

    for i, prog in enumerate(selected_programs):
        # str(prog) -> Formülü verir (örn: add(X0, X1))
        # prog.raw_fitness_ -> Puanı verir
        print(f"{i + 1:<4} | {prog.raw_fitness_:.5f}            | {str(prog)}")

    print("=" * 60 + "\n")
    # -------------------------------------------------------------

    output_train = []
    output_test = []

    for prog in selected_programs:
        output_train.append(prog.execute(X_train))
        output_test.append(prog.execute(X_test))

    if not output_train:
        return np.zeros((X_train.shape[0], 0)), np.zeros((X_test.shape[0], 0))

    # 6. Listeleri numpy array (sütun bazlı) formatına çevir
    X_train_st_selected = np.column_stack(output_train)
    X_test_st_selected = np.column_stack(output_test)

    return X_train_st_selected, X_test_st_selected

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





