#################################################################
# İSTİNYE ÜNİVERSİTESİ - MAKİNE ÖĞRENMESİ PROJESİ
# KONU: KONUT FİYAT TAHMİNİ (REGRESYON VE SINIFLANDIRMA)
# KULLANILAN KONSEPTLER: Week 2 (Linear Regression) & Week 3 (Classification)
#################################################################

# --- ADIM 0: GEREKLİ KÜTÜPHANELERİ İÇE AKTARMA ---
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Scikit-learn (sklearn) kütüphanesinden gerekli modüller
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

# Uyarıları bastırmak için (opsiyonel)
import warnings
warnings.filterwarnings('ignore')

print("Tüm kütüphaneler başarıyla yüklendi.")


# --- ADIM 1: VERİ SETİNİ YÜKLEME VE İLK BAKIŞ ---
try:
    df = pd.read_csv('train.csv')
    print("Veri seti 'train.csv' başarıyla yüklendi.")
    print("Veri setinin ilk 5 satırı:")
    print(df.head())
except FileNotFoundError:
    print("HATA: 'train.csv' dosyası bu klasörde bulunamadı.")
    print("Lütfen dosyayı kodun bulunduğu dizine kopyalayın.")
    exit() # Dosya yoksa programı durdur


#################################################################
# BÖLÜM 1: REGRESYON (WEEK 2 KONSEPTLERİ)
# HEDEF: Sürekli bir değer olan 'SalePrice' (Satış Fiyatı) tahmin etmek.
#################################################################
print("\n" + "="*50)
print(" BÖLÜM 1: REGRESYON GÖREVİ (WEEK 2)")
print("="*50)

# --- 1.A: BASİT DOĞRUSAL REGRESYON (Simple Linear Regression) ---
# Sadece bir özellik (X) kullanarak hedefi (y) tahmin etme.
print("\n--- 1.A: Basit Doğrusal Regresyon ---")
print("Hedef (y): SalePrice, Özellik (X): GrLivArea")

# Eksik veri olmayan satırları al
df_simple = df[['GrLivArea', 'SalePrice']].dropna()

X_simple = df_simple[['GrLivArea']]
y_simple = df_simple['SalePrice']

# Veriyi Eğitim ve Test olarak ayırma
X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(X_simple, y_simple, test_size=0.2, random_state=42)

# Modeli oluşturma ve eğitme
simple_model = LinearRegression()
simple_model.fit(X_train_s, y_train_s)
print("Basit Doğrusal Regresyon modeli eğitildi.")

# Tahmin yapma
y_pred_s = simple_model.predict(X_test_s)

# Modeli Değerlendirme (Week 2 - Cost Function: MSE)
mse_simple = mean_squared_error(y_test_s, y_pred_s)
r2_simple = r2_score(y_test_s, y_pred_s) # R-Kare: Modelin varyansı açıklama oranı

print(f"  Modelin Katsayısı (Eğim, β1): {simple_model.coef_[0]:.2f}")
print(f"  Modelin Sabiti (Kesişim, β0): {simple_model.intercept_:.2f}")
print(f"  Ortalama Kare Hata (MSE): {mse_simple:.2f}")
print(f"  R-Kare Değeri: {r2_simple:.2f}")


# --- 1.B: ÇOKLU DOĞRUSAL REGRESYON (Multiple Linear Regression) ---
# Birden fazla özellik (X1, X2, ...) kullanarak hedefi (y) tahmin etme.
print("\n--- 1.B: Çoklu Doğrusal Regresyon ---")
features = ['GrLivArea', 'BedroomAbvGr', 'FullBath']
target = 'SalePrice'
print(f"Hedef (y): {target}, Özellikler (X): {features}")

# Eksik veri olan satırları atla
df_multi = df[features + [target]].dropna()

X_multi = df_multi[features]
y_multi = df_multi[target]

# Veriyi Eğitim ve Test olarak ayırma
X_train_m, X_test_m, y_train_m, y_test_m = train_test_split(X_multi, y_multi, test_size=0.2, random_state=42)

# Modeli oluşturma ve eğitme
multi_model = LinearRegression()
multi_model.fit(X_train_m, y_train_m)
print("Çoklu Doğrusal Regresyon modeli eğitildi.")

# Tahmin yapma
y_pred_m = multi_model.predict(X_test_m)

# Modeli Değerlendirme
mse_multi = mean_squared_error(y_test_m, y_pred_m)
r2_multi = r2_score(y_test_m, y_pred_m)

print(f"  Modelin Katsayıları (b1, b2, b3): {multi_model.coef_}")
print(f"  Modelin Sabiti (b0): {multi_model.intercept_:.2f}")
print(f"  Ortalama Kare Hata (MSE): {mse_multi:.2f}")
print(f"  R-Kare Değeri: {r2_multi:.2f}")

# Karşılaştırma
print(f"\n  Karşılaştırma: Çoklu Regresyon (R2: {r2_multi:.2f})")
print(f"  Basit Regresyondan (R2: {r2_simple:.2f}) daha iyi bir açıklama sağladı.")


#################################################################
# BÖLÜM 2: SINIFLANDIRMA (WEEK 3 KONSEPTLERİ)
# HEDEF: Kategorik bir değer olan 'Is_Expensive' (Pahalı mı?) tahmin etmek.
#################################################################
print("\n" + "="*50)
print(" BÖLÜM 2: SINIFLANDIRMA GÖREVİ (WEEK 3)")
print("="*50)

# --- 2.A: SINIFLANDIRMA İÇİN VERİ HAZIRLAMA ---
print("\n--- 2.A: Sınıflandırma için Veri Hazırlama ---")

# 1. Kategorik Hedef Değişkeni Oluşturma
# 'SalePrice'ı kategorik hale getiriyoruz.
median_price = df['SalePrice'].median()
print(f"Ortanca ev fiyatı (Median Price): ${median_price:.2f}")

# Fiyatı ortancadan büyükse 1 (Pahalı), değilse 0 (Pahalı Değil)
df['Is_Expensive'] = (df['SalePrice'] > median_price).astype(int)
print("Yeni hedef sütun 'Is_Expensive' (0 veya 1) oluşturuldu.")

# 2. Sınıflandırma için X ve y'yi tanımla
# Regresyonda kullandığımız aynı özellikleri kullanalım
class_features = ['GrLivArea', 'BedroomAbvGr', 'FullBath']
class_target = 'Is_Expensive'
print(f"Hedef (y): {class_target}, Özellikler (X): {class_features}")

# 3. Eksik Veri Yönetimi
df_class = df[class_features + [class_target]].dropna()

X_class = df_class[class_features]
y_class = df_class[class_target]

# 4. Veri Ayırma
X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X_class, y_class, test_size=0.2, random_state=42)

# 5. ÖNEMLİ: Veri Ölçeklendirme (Feature Scaling)
# KNN (mesafe bazlı) ve Lojistik Regresyon ölçeklendirmeden faydalanır.
scaler = StandardScaler()
X_train_c_scaled = scaler.fit_transform(X_train_c)
X_test_c_scaled = scaler.transform(X_test_c)
print("Özellikler (X_train, X_test) StandardScaler ile ölçeklendi.")


# --- 2.B: MODEL 1 - LOJİSTİK REGRESYON ---
print("\n--- 2.B: Model 1 - Lojistik Regresyon ---")
# Week 3'te öğrenilen Lojistik Regresyon, ikili sınıflandırma için kullanılır.
# Tahmini olasılığa dönüştürmek için Sigmoid Fonksiyonu kullanır.
log_model = LogisticRegression(random_state=42)
log_model.fit(X_train_c_scaled, y_train_c)
print("Lojistik Regresyon modeli eğitildi.")

# Tahmin yapma
y_pred_log = log_model.predict(X_test_c_scaled)


# --- 2.C: MODEL 2 - K-EN YAKIN KOMŞU (KNN) ---
print("\n--- 2.C: Model 2 - K-En Yakın Komşu (KNN) ---")
# Week 3'te öğrenilen KNN, bir "tembel öğrenici"dir (lazy learner).
# Tahmin için en yakın K komşunun oyunu kullanır.
# K=5 seçiyoruz (genellikle tek sayı olması önerilir).
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train_c_scaled, y_train_c)
print("KNN modeli (k=5) eğitildi.")

# Tahmin yapma
y_pred_knn = knn_model.predict(X_test_c_scaled)


#################################################################
# BÖLÜM 3: SINIFLANDIRMA DEĞERLENDİRME (WEEK 3 METRİKLERİ)
#################################################################
print("\n" + "="*50)
print(" BÖLÜM 3: SINIFLANDIRMA DEĞERLENDİRMESİ")
print("="*50)

# Bu metrikler Week 3, Sayfa 31-35 arasında anlatılmıştır.

# --- Lojistik Regresyon Değerlendirmesi ---
print("\n--- Lojistik Regresyon Sonuçları ---")
cm_log = confusion_matrix(y_test_c, y_pred_log)
acc_log = accuracy_score(y_test_c, y_pred_log)
pre_log = precision_score(y_test_c, y_pred_log)
rec_log = recall_score(y_test_c, y_pred_log)
f1_log = f1_score(y_test_c, y_pred_log)

print(f"  Confusion Matrix:\n {cm_log}")
print(f"  Accuracy (Doğruluk): {acc_log:.3f}")
print(f"  Precision (Kesinlik): {pre_log:.3f}")
print(f"  Recall (Duyarlılık): {rec_log:.3f}")
print(f"  F1-Score: {f1_log:.3f}")


# --- KNN Değerlendirmesi ---
print("\n--- KNN (k=5) Sonuçları ---")
cm_knn = confusion_matrix(y_test_c, y_pred_knn)
acc_knn = accuracy_score(y_test_c, y_pred_knn)
pre_knn = precision_score(y_test_c, y_pred_knn)
rec_knn = recall_score(y_test_c, y_pred_knn)
f1_knn = f1_score(y_test_c, y_pred_knn)

print(f"  Confusion Matrix:\n {cm_knn}")
print(f"  Accuracy (Doğruluk): {acc_knn:.3f}")
print(f"  Precision (Kesinlik): {pre_knn:.3f}")
print(f"  Recall (Duyarlılık): {rec_knn:.3f}")
print(f"  F1-Score: {f1_knn:.3f}")


print("\n" + "="*50)
print("PROJE TAMAMLANDI.")
print(f"KARŞILAŞTIRMA: Accuracy değerlerine göre KNN ({acc_knn:.3f})")
print(f"bu görev için Lojistik Regresyondan ({acc_log:.3f}) biraz daha iyi performans gösterdi.")
print("="*50)


# --- BÖLÜM 4: GÖRSELLEŞTİRME (Opsiyonel ama önerilir) ---
print("\n--- Bölüm 4: Görselleştirmeler Oluşturuluyor... ---")

# 1. Regresyon Çizgisi (Basit Model)
plt.figure(figsize=(10, 6))
sns.scatterplot(x=X_test_s['GrLivArea'], y=y_test_s, color='blue', label='Gerçek Fiyatlar', alpha=0.6)
sns.lineplot(x=X_test_s['GrLivArea'], y=y_pred_s, color='red', linewidth=2, label='Tahmin Çizgisi (Regresyon)')
plt.title('Bölüm 1: Basit Doğrusal Regresyon Sonucu')
plt.xlabel('Yaşam Alanı (GrLivArea)')
plt.ylabel('Satış Fiyatı (SalePrice)')
plt.legend()
plt.savefig('1_simple_regression_plot.png')
print("Grafik '1_simple_regression_plot.png' olarak kaydedildi.")

# 2. Sınıflandırma Sınırları (2 özellik kullanarak basitleştirelim)
# Sadece GrLivArea ve BedroomAbvGr kullanarak modeli tekrar eğitelim (görselleştirme için)
X_vis = df_class[['GrLivArea', 'BedroomAbvGr']]
y_vis = df_class[class_target]

X_train_v, X_test_v, y_train_v, y_test_v = train_test_split(X_vis, y_vis, test_size=0.2, random_state=42)

scaler_vis = StandardScaler()
X_train_v_scaled = scaler_vis.fit_transform(X_train_v)
X_test_v_scaled = scaler_vis.transform(X_test_v)

knn_vis = KNeighborsClassifier(n_neighbors=5)
knn_vis.fit(X_train_v_scaled, y_train_v)
print("Görselleştirme için 2 özellikli KNN modeli eğitildi.")

# Karar Sınırlarını Çizme (Meshgrid)
x_min, x_max = X_train_v_scaled[:, 0].min() - 1, X_train_v_scaled[:, 0].max() + 1
y_min, y_max = X_train_v_scaled[:, 1].min() - 1, X_train_v_scaled[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                     np.arange(y_min, y_max, 0.1))

Z = knn_vis.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.figure(figsize=(10, 6))
plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.coolwarm)
sns.scatterplot(x=X_train_v_scaled[:, 0], y=X_train_v_scaled[:, 1], c=y_train_v,
                cmap=plt.cm.coolwarm, edgecolor='k', label='Veri Noktaları')
plt.title('Bölüm 2: KNN (k=5) Karar Sınırları (2 Özellik)')
plt.xlabel('Yaşam Alanı (GrLivArea) - Ölçeklenmiş')
plt.ylabel('Oda Sayısı (BedroomAbvGr) - Ölçeklenmiş')
plt.legend(['0: Ucuz', '1: Pahalı'])
plt.savefig('2_knn_decision_boundary.png')
print("Grafik '2_knn_decision_boundary.png' olarak kaydedildi.")
print("\nTüm işlemler tamamlandı.")
