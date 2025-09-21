import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from gplearn.genetic import SymbolicRegressor


# CSV dosyasını oku
df = pd.read_csv("car_sales_data.csv")

# Sütun isimlerini listele
print("Sütunlar:", df.columns.tolist())

# Veri tipleri ve boş değer durumlarını öğren
print("\nVeri Bilgisi:")
print(df.info())

# İlk 5 satırı göster
print("\nİlk 5 Satır:")
print(df.head())

# Son 5 satırı görmek için
print("\nSon 5 Satır:")
print(df.tail())

# Sayısal sütunların istatistiksel özetini görmek için
print("\nİstatistiksel Özet:")
print(df.describe())

df_encoded = pd.get_dummies(df, columns=["Fuel type", "Manufacturer"], drop_first=True)

# Özellikler ve hedefi ayır
X = df_encoded.drop(columns=["Price", "Model"])  # Model çok fazla kategoriye sahip olabilir
y = df_encoded["Price"]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42 )

est_gp = SymbolicRegressor(
    population_size=3000,    # aday formüller sayısı
    generations=20,          # nesil sayısı
    stopping_criteria=0.01,  # hata hedefi
    p_crossover=0.7,         # çaprazlama
    p_subtree_mutation=0.1,  # mutasyon
    p_hoist_mutation=0.05,
    p_point_mutation=0.1,
    max_samples=0.9,
    verbose=1,
    parsimony_coefficient=0.01,
    random_state=0
)

est_gp.fit(X_train, y_train)

print(est_gp._program)