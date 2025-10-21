# ---
# GEREKLİ KÜTÜPHANELER
# ---
import pandas as pd
import numpy as np

# Veri setlerini yüklemek için (scikit-learn içinden hazır)
from sklearn.datasets import load_breast_cancer, load_iris

# Veriyi ayırmak ve ölçeklendirmek için (ÇOK ÖNEMLİ ADIMLAR)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Modeller (Dersin ana konuları)
from sklearn.linear_model import LogisticRegression # Konu 1
from sklearn.neighbors import KNeighborsClassifier # Konu 2

# Değerlendirme Metrikleri (Modeller ne kadar başarılı?)
# Ders notları Sayfa 31-35 arası [cite: 500-531]
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# GÖRSELLEŞTİRME KÜTÜPHANELERİ (YENİ EKLENDİ)
import matplotlib.pyplot as plt # Çizim yapmak ve kaydetmek için
import seaborn as sns # Daha güzel görünen grafikler, özellikle ısı haritası için

# ---
# KONU 1: LOGISTIC REGRESSION (İkili Sınıflandırma)
# ---
# AÇIKLAMA: Logistic Regression, adında "Regression" geçse de bir *sınıflandırma* algoritmasıdır[cite: 162].
# Genellikle sonucu 0 veya 1 olan (ikili/binary) problemleri çözmek için kullanılır[cite: 261].
# Sigmoid fonksiyonu kullanarak bir olayın olma olasılığını hesaplar[cite: 188, 196].
# Ders notlarına göre (Sayfa 6), Logistic Regression bir "Eager Learner"dır.
# Yani, eğitim verisini alır almaz bir matematiksel model (formül) oluşturur.
print("="*30)
print("KONU 1: LOGISTIC REGRESSION BAŞLIYOR")
print("="*30)

# 1. Veri Setini Yükle (Breast Cancer)
# Amaç: 30 farklı özniteliğe (tümör yarıçapı, dokusu vb.) bakarak
# bir tümörün Kötü Huylu (0) mu, İyi Huylu (1) mi olduğunu tahmin etmek.
cancer_data = load_breast_cancer()
X_cancer = pd.DataFrame(cancer_data.data, columns=cancer_data.feature_names)
y_cancer = cancer_data.target

print(f"Logistic Regression Veri Seti: {X_cancer.shape[0]} örnek, {X_cancer.shape[1]} öznitelik")
print("Target (Hedef) Sınıflar:", np.unique(y_cancer), "(0=Kötü Huylu, 1=İyi Huylu)\n")

# 2. Veriyi Eğitim ve Test Olarak Ayır
# Modelimizi 'X_train_c' ve 'y_train_c' ile eğiteceğiz.
# Sonra, modelin daha önce hiç görmediği 'X_test_c' verisini verip
# tahminlerinin ('y_pred_c') gerçek sonuçlarla ('y_test_c') ne kadar
# uyuştuğuna bakacağız.
X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X_cancer, y_cancer, test_size=0.3, random_state=42)

# 3. Veriyi Ölçeklendir (StandardScaler)
# AÇIKLAMA: Bu adım, bazı özniteliklerin (mesela 'mean area' 1000'lerdeyken)
# diğerlerini (mesela 'mean smoothness' 0.1'lerde) domine etmesini engeller.
# Tüm öznitelikleri ortalaması 0, standart sapması 1 olan bir ölçeğe getirir.
# Logistic Regression ve özellikle KNN için bu adım performansı ciddi şekilde artırır.
scaler = StandardScaler()
X_train_c = scaler.fit_transform(X_train_c)
X_test_c = scaler.transform(X_test_c)

# 4. Modeli Oluştur ve Eğit
# Modeli 'random_state=42' ile başlatıyoruz ki her çalıştırdığımızda
# aynı sonucu alalım (bilimsel tekrarlanabilirlik için).
log_reg_model = LogisticRegression(random_state=42)
log_reg_model.fit(X_train_c, y_train_c) # 'fit' komutu Eager Learning'in gerçekleştiği yerdir.
print("Logistic Regression modeli başarıyla eğitildi.\n")

# 5. Tahmin Yap ve Modeli Değerlendir
y_pred_c = log_reg_model.predict(X_test_c)

# METRİK 1: Doğruluk (Accuracy) [cite: 502-504]
# Toplam tahminlerin yüzde kaçı doğru?
accuracy_c = accuracy_score(y_test_c, y_pred_c)
print(f"1. Doğruluk (Accuracy): {accuracy_c:.4f} (Yani test verisinin %{accuracy_c*100:.2f}'si doğru tahmin edildi)\n")

# METRİK 2: Sınıflandırma Raporu (Precision, Recall, F1-Score) [cite: 518-531]
# Bu, doğruluktan daha detaylı bilgi verir.
print("2. Sınıflandırma Raporu (Detaylı):")
report_c = classification_report(y_test_c, y_pred_c, target_names=['Kötü Huylu (0)', 'İyi Huylu (1)'])
print(report_c)
print("""
--- Rapor Nasıl Okunur? ---
- Precision (Kötü Huylu): Modelin 'Kötü Huylu' dediği vakaların % kaçı gerçekten 'Kötü Huylu'ydu?
- Recall (Kötü Huylu): Gerçekteki 'Kötü Huylu' vakaların % kaçını model yakalayabildi?
- f1-score: Precision ve Recall'un harmonik ortalamasıdır, genel bir başarı ölçütüdür.
---------------------------\n
""")

# METRİK 3: Karmaşıklık Matrisi (Confusion Matrix) - GÖRSEL (PNG) [cite: 506-516]
print("3. Karmaşıklık Matrisi (PNG dosyası oluşturuluyor...)")
cm_c = confusion_matrix(y_test_c, y_pred_c)

# Sınıf etiketlerini belirliyoruz (0 ve 1 için)
class_names_c = ['Kötü Huylu (0)', 'İyi Huylu (1)']

# Seaborn kullanarak heatmap çizdirme
plt.figure(figsize=(8, 6)) # Resim boyutunu ayarla
sns.heatmap(cm_c, annot=True, fmt='d', cmap='Blues', # annot=True: sayıları göster, fmt='d': tam sayı
            xticklabels=class_names_c,
            yticklabels=class_names_c)

plt.title('Logistic Regression Karmaşıklık Matrisi')
plt.ylabel('Gerçek (Actual) Sınıf')
plt.xlabel('Tahmin Edilen (Predicted) Sınıf')
plt.tight_layout() # Sıkışmayı önlemek için

# PNG olarak kaydetme
plt.savefig('logistic_regression_cm.png')
print("PNG dosyası 'logistic_regression_cm.png' olarak başarıyla kaydedildi.")
plt.clf() # Bir sonraki çizim için mevcut grafiği temizle

print("KONU 1: LOGISTIC REGRESSION TAMAMLANDI\n\n")


# ---
# KONU 2: K-NEAREST NEIGHBORS (KNN) (Çok Sınıflı Sınıflandırma)
# ---
# AÇIKLAMA: KNN, en basit makine öğrenimi algoritmalarından biridir[cite: 282].
# "Non-parametric" (yani verinin dağılımı hakkında bir varsayımda bulunmaz)[cite: 287].
# En önemlisi, bir "Lazy Learner"dır (Tembel Öğrenici).
# Yani, 'fit' aşamasında bir model/formül oluşturmaz; sadece tüm eğitim verisini 'ezberler'.
# Tahmin aşamasında, yeni noktaya en yakın 'K' adet komşuyu bulur ve
# onların çoğunluğunun oyuna göre sınıfı belirler [cite: 293, 389-392].
print("="*30)
print("KONU 2: K-NEAREST NEIGHBORS (KNN) BAŞLIYOR")
print("="*30)

# 1. Veri Setini Yükle (Iris)
# Amaç: 4 özniteliğe (çanak yaprağı ve taç yaprağı en/boy) bakarak
# çiçeğin 3 türünden (0=Setosa, 1=Versicolor, 2=Virginica) hangisi olduğunu tahmin etmek.
# Bu, "Multiclass" (çok sınıflı) bir sınıflandırma problemidir [cite: 67-69].
iris_data = load_iris()
X_iris = pd.DataFrame(iris_data.data, columns=iris_data.feature_names)
y_iris = iris_data.target

print(f"KNN Veri Seti: {X_iris.shape[0]} örnek, {X_iris.shape[1]} öznitelik")
print("Target (Hedef) Sınıflar:", np.unique(y_iris), "(0=Setosa, 1=Versicolor, 2=Virginica)\n")

# 2. Veriyi Eğitim ve Test Olarak Ayır
X_train_i, X_test_i, y_train_i, y_test_i = train_test_split(X_iris, y_iris, test_size=0.3, random_state=42)

# 3. Veriyi Ölçeklendir (StandardScaler)
# AÇIKLAMA: BU ADIM KNN İÇİN HAYATİ ÖNEM TAŞIR!
# KNN, 'Euclidean Distance' gibi  mesafe metriklerine dayanır.
# Eğer ölçeklendirme yapmazsanız, 'sepal length' (cm) gibi büyük sayılar,
# 'petal width' (mm) gibi küçük sayıları ezer ve model yanlış komşuları bulur.
scaler_i = StandardScaler()
X_train_i = scaler_i.fit_transform(X_train_i)
X_test_i = scaler_i.transform(X_test_i)

# 4. Modeli Oluştur ve Eğit
# 'n_neighbors=5' ifadesi, K=5 demektir. Ders notlarında (Sayfa 27) [cite: 450] 'K' değerinin
# seçimi anlatılır. Genellikle tek sayı (beraberliği önlemek için) ve
# çok düşük (Overfitting) [cite: 454-458] veya çok yüksek (Underfitting) [cite: 460] olmayan bir değer seçilir.
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train_i, y_train_i) # 'fit' komutu burada sadece veriyi 'ezberler' [cite: 355]
print("KNN modeli K=5 için oluşturuldu (veri depolandı).\n")

# 5. Tahmin Yap ve Modeli Değerlendir
# Tahmin aşaması ('predict'), KNN'de tüm hesaplamanın yapıldığı yerdir.
# Model, her bir test noktası için tüm eğitim verisine olan mesafeyi hesaplar.
y_pred_i = knn_model.predict(X_test_i)

# METRİK 1: Doğruluk (Accuracy)
accuracy_i = accuracy_score(y_test_i, y_pred_i)
print(f"1. Doğruluk (Accuracy): {accuracy_i:.4f} (Yani test verisinin %{accuracy_i*100:.2f}'si doğru tahmin edildi)\n")

# METRİK 2: Sınıflandırma Raporu
print("2. Sınıflandırma Raporu (Detaylı):")
report_i = classification_report(y_test_i, y_pred_i, target_names=iris_data.target_names)
print(report_i)

# METRİK 3: Karmaşıklık Matrisi (Confusion Matrix) - GÖRSEL (PNG)
print("3. Karmaşıklık Matrisi (PNG dosyası oluşturuluyor...)")
cm_i = confusion_matrix(y_test_i, y_pred_i)

# Sınıf etiketlerini veri setinden alıyoruz
class_names_i = iris_data.target_names

# Seaborn kullanarak heatmap çizdirme
plt.figure(figsize=(8, 6))
sns.heatmap(cm_i, annot=True, fmt='d', cmap='Greens', # Bu sefer farklı bir renk haritası
            xticklabels=class_names_i,
            yticklabels=class_names_i)

plt.title('KNN (K=5) Karmaşıklık Matrisi')
plt.ylabel('Gerçek (Actual) Sınıf')
plt.xlabel('Tahmin Edilen (Predicted) Sınıf')
plt.tight_layout()

# PNG olarak kaydetme
plt.savefig('knn_cm.png')
print("PNG dosyası 'knn_cm.png' olarak başarıyla kaydedildi.")
plt.clf() # Grafiği temizle

print("KONU 2: K-NEAREST NEIGHBORS (KNN) TAMAMLANDI")
