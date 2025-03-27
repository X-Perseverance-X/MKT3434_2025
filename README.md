# Machine Learning Course GUI

Bu proje, makine öğrenimi modellerini ve derin öğrenme yapılarını görselleştirmek, eğitim süreçlerini yönetmek ve veri ön işleme adımlarını uygulamak için kapsamlı bir GUI (Grafiksel Kullanıcı Arayüzü) sağlar. Uygulama, klasik makine öğrenimi algoritmalarından derin öğrenmeye kadar geniş bir yelpazede model eğitimi ve değerlendirmesi yapmanızı kolaylaştırır.

---

## İçindekiler

- [Özellikler](#özellikler)
- [Kurulum](#kurulum)
- [Kullanım](#kullanım)
  - [Veri Yönetimi](#veri-yönetimi)
  - [Görselleştirme](#görselleştirme)
  - [Model Eğitimi](#model-eğitimi)
  - [Derin Öğrenme Sekmesi](#derin-öğrenme-sekmesi)
  - [Ek Özellikler](#ek-özellikler)
- [Bağımlılıklar](#bağımlılıklar)
- [Proje Yapısı](#proje-yapısı)
- [Sorun Giderme](#sorun-giderme)
- [Katkıda Bulunanlar](#katkıda-bulunanlar)
- [Lisans](#lisans)

---

## Özellikler

- **Veri Yönetimi:**
  - Öntanımlı veri setleri: Iris, Boston Housing, Breast Cancer.
  - Kullanıcının CSV formatında kendi veri setini yüklemesi.
  - Ölçekleme seçenekleri: Standard Scaling, Min-Max Scaling, Robust Scaling.
  - Veri setini eğitim ve test olarak ayırma oranı belirleme.

- **Görselleştirme:**
  - Ham verinin 3D scatter plot ile görselleştirilmesi.
  - Histogramlar ile verinin dağılımının incelenmesi.
  - X, Y ve Z eksenleri için öznitelik seçimi.
  - Model tahminlerinin 3D grafik üzerinde gerçek değerlerle karşılaştırılması.
  - Model performans metriklerinin (hata, doğruluk, karışıklık matrisi vb.) metin kutusunda gösterimi.

- **Model Eğitimi:**
  - Klasik makine öğrenimi algoritmaları: Linear Regression, Logistic Regression, Naive Bayes, Support Vector Machine, Decision Tree, Random Forest, K-Nearest Neighbors.
  - Her algoritma için parametre ayarlarının kullanıcı tarafından belirlenebilmesi.
  - Kayıp fonksiyonu ayarları: Sınıflandırma (Cross Entropy, Binary Cross Entropy, Hinge Loss) ve regresyon (MSE, MAE, Huber Loss) seçenekleri.
  - Eğitim tamamlandığında, modelin tahminleri, görselleştirmeleri ve performans metriklerinin güncellenmesi.

- **Derin Öğrenme Sekmesi:**
  - Çok katmanlı algılayıcı (MLP), Konvolüsyonel Sinir Ağı (CNN) ve Tekrarlayan Sinir Ağı (RNN) için yapılandırma seçenekleri.
  - Dinamik katman ekleme: Dense, Conv2D, MaxPooling2D, Flatten, Dropout katmanları.
  - Eğitim parametreleri: Batch size, epochs, öğrenme oranı.
  - Eğitim sürecinde ilerleme çubuğu ve eğitim geçmişi görselleştirmesi.

- **Ek Özellikler:**
  - Sekmeli yapı: Klasik ML, Derin Öğrenme, Boyut İndirgeme ve Pekiştirmeli Öğrenme konularına özel sekmeler.
  - Hata ve uyarı mesajlarının gösterimi.
  - Kullanıcı dostu, kompakt ve düzenli arayüz.

---

## Kurulum

### Gereksinimler
- numpy>=1.21.0
- pandas>=1.3.0
- PyQt6>=6.4.0
- matplotlib>=3.4.0
- scikit-learn>=1.0.0
- tensorflow>=2.8.0
- scipy>=1.7.0


- **Python 3.7+**
- Aşağıdaki Python paketlerine ihtiyaç vardır:
  - `numpy`
  - `pandas`
  - `scikit-learn`
  - `matplotlib`
  - `PyQt6`
  - `tensorflow` (ve `keras` modülü, TensorFlow 2.x ile birlikte gelir)

### Kurulum Adımları

1. Python ve pip’in sisteminizde kurulu olduğundan emin olun.
2. Gerekli paketleri yüklemek için aşağıdaki komutu çalıştırın:
   ```bash
   pip install numpy pandas scikit-learn matplotlib PyQt6 tensorflow

