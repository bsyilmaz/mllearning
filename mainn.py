# Gerekli kütüphaneleri içe aktarıyoruz.
import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer, load_iris # Dersimiz için iki farklı veri seti
from sklearn.model_selection import train_test_split # Veriyi eğitim ve test olarak ayırmak için
from sklearn.preprocessing import StandardScaler # Veri ölçeklendirme için (özellikle KNN için önemli)
from sklearn.linear_model import LogisticRegression # Logistic Regression modeli
from sklearn.neighbors import KNeighborsClassifier # K-Nearest Neighbors (KNN) modeli
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report # Değerlendirme metrikleri

# ---
# KONU 1: LOGISTIC REGRESSION (İkili Sınıflandırma)
# ---
# Sunumunuzda (Sayfa 7) belirtildiği gibi, Logistic Regression bir denetimli öğrenme tekniğidir [cite: 161]
# ve genellikle ikili (binary) sonuçları tahmin etmek için kullanılır (0 veya 1)[cite: 163, 164].
# Biz de "Breast Cancer" (Meme Kanseri) veri setini kullanacağız.
# Amaç: Tümörün iyi huylu (0) mu yoksa kötü huylu (1) mu olduğunu tahmin etmek.
# Bu, sunumdaki "Binomial" sınıflandırma türüne bir örnektir[cite: 261].
print("="*30)
print("KONU 1: LOGISTIC REGRESSION BAŞLIYOR")
print("="*30)

# 1. Veri Setini Yükle
cancer_data = load_breast_cancer()
X_cancer = pd.DataFrame(cancer_data.data, columns=cancer_data.feature_names)
y_cancer = cancer_data.target # 0 = Kötü Huylu, 1 = İyi Huylu

print(f"Logistic Regression Veri Seti: {X_cancer.shape[0]} örnek, {X_cancer.shape[1]} öznitelik\n")
print("Örnek Veri (İlk 5 satır):")
print(X_cancer.head())
print("\nTarget (Hedef) Sınıflar:", np.unique(y_cancer), "(0=Kötü Huylu, 1=İyi Huylu)\n")

# 2. Veriyi Eğitim ve Test Olarak Ayır
# Modeli eğitmek ve test etmek için veriyi ayırıyoruz.
X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X_cancer, y_cancer, test_size=0.3, random_state=42)

# 3. Veriyi Ölçeklendir
# Logistic Regression ve özellikle KNN, özniteliklerin (features) ölçeğine duyarlıdır.
# En iyi performansı almak için veriyi standartlaştırıyoruz.
scaler = StandardScaler()
X_train_c = scaler.fit_transform(X_train_c)
X_test_c = scaler.transform(X_test_c)

# 4. Modeli Oluştur ve Eğit
# Sunumda (Sayfa 5) belirtildiği gibi, Logistic Regression "Linear" bir modeldir[cite: 127, 128].
# (Sayfa 14-15) "Eager learner" kategorisine girer; yani veriyi alır almaz bir model oluşturur[cite: 150].
log_reg_model = LogisticRegression(random_state=42)
log_reg_model.fit(X_train_c, y_train_c)
print("Logistic Regression modeli başarıyla eğitildi.\n")

# 5. Tahmin Yap ve Modeli Değerlendir
y_pred_c = log_reg_model.predict(X_test_c)

# Sunumunuzdaki (Sayfa 31-35) Değerlendirme Metriklerini (Evaluation Metrics) kullanıyoruz:
# A) Accuracy (Doğruluk) [cite: 502, 503]
accuracy_c = accuracy_score(y_test_c, y_pred_c)
print(f"1. Doğruluk (Accuracy): {accuracy_c:.4f}")

# B) Confusion Matrix (Karmaşıklık Matrisi) [cite: 506, 507]
#           Tahmin: 0  Tahmin: 1
# Gerçek: 0    [TN]       [FP]
# Gerçek: 1    [FN]       [TP]
cm_c = confusion_matrix(y_test_c, y_pred_c)
print("\n2. Karmaşıklık Matrisi (Confusion Matrix):")
print(cm_c)

# C) Precision, Recall, F1-Score [cite: 518, 522, 527]
report_c = classification_report(y_test_c, y_pred_c, target_names=['Kötü Huylu (0)', 'İyi Huylu (1)'])
print("\n3. Sınıflandırma Raporu (Precision, Recall, F1-Score):")
print(report_c)
print("KONU 1: LOGISTIC REGRESSION TAMAMLANDI\n\n")


# ---
# KONU 2: K-NEAREST NEIGHBORS (KNN) (Çok Sınıflı Sınıflandırma)
# ---
# Sunumunuzda (Sayfa 15) belirtildiği gibi, KNN denetimli bir öğrenme algoritmasıdır [cite: 282]
# ve hem sınıflandırma hem de regresyon için kullanılabilir[cite: 284].
# KNN, "non-parametric" [cite: 287] ve "lazy learner" [cite: 143, 287, 355] bir algoritmadır.
# "Lazy" olması, eğitim aşamasında sadece veriyi depoladığı anlamına gelir[cite: 143, 355].
# Sunumunuzun 4. sayfasında gösterilen Iris veri setini kullanacağız [cite: 73-121].
# Amaç: Çiçeğin türünü (3 farklı sınıf) tahmin etmek.
# Bu, sunumdaki "Multiclass" sınıflandırma türüne bir örnektir[cite: 67, 69].
print("="*30)
print("KONU 2: K-NEAREST NEIGHBORS (KNN) BAŞLIYOR")
print("="*30)

# 1. Veri Setini Yükle
iris_data = load_iris()
X_iris = pd.DataFrame(iris_data.data, columns=iris_data.feature_names)
y_iris = iris_data.target # 0=Setosa, 1=Versicolor, 2=Virginica

print(f"KNN Veri Seti: {X_iris.shape[0]} örnek, {X_iris.shape[1]} öznitelik\n")
print("Örnek Veri (İlk 5 satır):")
print(X_iris.head())
print("\nTarget (Hedef) Sınıflar:", np.unique(y_iris), "(0=Setosa, 1=Versicolor, 2=Virginica)\n")

# 2. Veriyi Eğitim ve Test Olarak Ayır
X_train_i, X_test_i, y_train_i, y_test_i = train_test_split(X_iris, y_iris, test_size=0.3, random_state=42)

# 3. Veriyi Ölçeklendir
# KNN, mesafeye dayalı bir algoritmadır (Euclidean, Manhattan vb.)[cite: 413, 422].
# Bu yüzden özniteliklerin aynı ölçekte olması ÇOK önemlidir.
scaler_i = StandardScaler()
X_train_i = scaler_i.fit_transform(X_train_i)
X_test_i = scaler_i.transform(X_test_i)

# 4. Modeli Oluştur ve Eğit
# Sunumda (Sayfa 27) "K" değerinin nasıl seçileceği anlatılıyor[cite: 450].
# K değeri genellikle tek bir sayı olarak seçilir (örn: 3, 5, 7)[cite: 452].
# Biz K=5 olarak belirleyelim.
# 'n_neighbors=5' ifadesi, K=5 demektir.
knn_model = KNeighborsClassifier(n_neighbors=5) 
knn_model.fit(X_train_i, y_train_i) # "Lazy learner" burada sadece veriyi 'ezberler' [cite: 355]
print("KNN modeli K=5 için oluşturuldu (veri depolandı).\n")

# 5. Tahmin Yap ve Modeli Değerlendir
# Tahmin aşaması, KNN için 'pahalı' olan kısımdır[cite: 356].
# Yeni nokta (test verisi) için en yakın K komşuyu bulur[cite: 364].
y_pred_i = knn_model.predict(X_test_i)

# Sunumunuzdaki (Sayfa 31-35) Değerlendirme Metriklerini (Evaluation Metrics) kullanıyoruz:
# A) Accuracy (Doğruluk) [cite: 502, 503]
accuracy_i = accuracy_score(y_test_i, y_pred_i)
print(f"1. Doğruluk (Accuracy): {accuracy_i:.4f}")

# B) Confusion Matrix (Karmaşıklık Matrisi) [cite: 506, 507]
cm_i = confusion_matrix(y_test_i, y_pred_i)
print("\n2. Karmaşıklık Matrisi (Confusion Matrix):")
print(cm_i)

# C) Precision, Recall, F1-Score [cite: 518, 522, 527]
report_i = classification_report(y_test_i, y_pred_i, target_names=iris_data.target_names)
print("\n3. Sınıflandırma Raporu (Precision, Recall, F1-Score):")
print(report_i)
print("KONU 2: K-NEAREST NEIGHBORS (KNN) TAMAMLANDI")
