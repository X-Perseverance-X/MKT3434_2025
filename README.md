# Machine Learning Course GUI

Bu proje, makine öğrenimi ve derin öğrenme modellerini **kolayca yapılandırmanızı**, **eğitmenizi**, **değerlendirmenizi** ve **görselleştirmenizi** sağlayan kapsamlı bir PyQt6 tabanlı grafiksel kullanıcı arayüzüdür. Hem klasik makine öğrenimi algoritmalarını hem de derin öğrenme yapılarını destekler; ayrıca boyut indirgeme, özellik çıkarımı ve pekiştirmeli öğrenme için ayrılmış sekmeler sunar.

---

## Özellikler

### 1. Veri Yönetimi
- **Öntanımlı Veri Setleri**: Iris, Boston Housing, Breast Cancer.
- **Özel CSV Yükleme**: Kullanıcının kendi CSV dosyasını seçip hedef değişkeni diyaloğu ile belirleyebilme.
- **Eksik Veri İşleme**:
  - _No Action_ (Hiçbir işlem yapılmaz)
  - _Mean Imputation_ (Ortalama ile doldurma)
- **Öznitelik Ölçekleme**:
  - No Scaling
  - Standard Scaling
  - Min-Max Scaling
  - Robust Scaling
- **Veri Bölme**:
  - Split oranları: 80-20 (Train-Test), 70-15-15, 60-20-20
  - K-Fold çapraz doğrulama (2–20 kat)

### 2. Görselleştirme
- **Ham Veri**: 3B scatter plot + ek histogram
- **Model Tahminleri**: 3B scatter plot ile tahmin vs. gerçek değerler
- **Metrikler**: Hata, doğruluk, karışıklık matrisi vb.
- Dinamik eksen seçimi: X, Y, Z (veya hedef değişken)

### 3. Model Eğitimi: Klasik ML
- Algoritmalar:
  - **Regresyon**: Linear Regression, Logistic Regression
  - **Sınıflandırma**: Naive Bayes (GaussianNB), SVM, Decision Tree, Random Forest, KNN
  - **Kümeleme**: K-Means
- Parametre ayarları: her algoritma için ilgili widget’lar (SpinBox, ComboBox, Checkbox)
- **K-Fold** sonuçları: Accuracy ve RMSE çıktısı bildirimi

### 4. Kayıp Fonksiyonu Ayarları
- **Sınıflandırma**:
  - Cross Entropy (kategorik)
  - Binary Cross Entropy
  - Hinge Loss
  - Sınıf ağırlıkları: None, Balanced, Custom
- **Regresyon**:
  - Mean Squared Error (MSE)
  - Mean Absolute Error (MAE)
  - Huber Loss (δ parametresi)

### 5. Derin Öğrenme
- **Dinamik Katman Yapısı**: Dense, Conv2D, MaxPooling2D, Flatten, Dropout
- **Eğitim Parametreleri**: Batch Size, Epochs, Learning Rate
- **Eğitim İlerleme**: Progress bar & grafik çıktıları (loss/accuracy)

### 6. Boyut İndirgeme Sekmesi
- **PCA**: Açıklanan varyans grafiği
- **Truncated SVD**: Açıklanan varyans grafiği
- **t‑SNE**: 2 veya 3 boyutlu projeksiyon
- **LDA**: Tek boyutlu histogram veya çok boyutlu scatter plot

### 7. Gelişmiş Boyut İndirgeme
- **Elbow Method** ve **Silhouette Score** ile K-Means kümeleme
- Özelleştirilebilir t‑SNE Perplexity

### 8. Özellik Çıkarımı
- Ayrı bir sekmede PCA, SVD, t‑SNE ve LDA işlemleri: bileşen sayısı seçimi & sonuç görselleştirme

### 9. Pekiştirmeli Öğrenme (RL)
- **Ortamlar**: CartPole-v1, MountainCar-v0, Acrobot-v1
- **Algoritmalar**: Q-Learning, SARSA, DQN (geliştirme aşamasında)

---

## Kurulum

1. **Python 3.7+** yüklü olduğundan emin olun.
2. Gerekli paketleri yükleyin:
   ```bash
   pip install numpy pandas matplotlib PyQt6 scikit-learn tensorflow
   ```
3. Proje dosyalarını indirin/clonelayın.
4. Ana dizinde terminalde:
   ```bash
   python v19.py
   ```

---

## Kullanım
1. Uygulamayı çalıştırdığınızda üst kısımda **Data Management** bölümünden veri setinizi seçin veya CSV yükleyin.
2. Eksik veri, ölçekleme ve split ayarlarını yapılandırın.
3. **Apply Loss Settings** ile kayıp fonksiyonunu ve (varsa) sınıf ağırlıklarını belirleyin.
4. Alt sekmelerden ilgilendiğiniz ML yöntemini seçip parametreleri girin.
5. **Train** butonuna basın, sonuçlar ve grafikler otomatik güncellenecektir.

---

## Proje Yapısı
```
├── README.md          # Proje dokümantasyonu
├── 21067004.py        # Ana uygulama kodu
└── requirements.txt   # (isteğe bağlı) paket listesi
```

---

## Sorun Giderme
- **GUI açılmıyor**: PyQt6 sürümünü kontrol edin.
- **TensorFlow Hataları**: GPU/CPU sürüm uyumluluğu.
- **Eksik paket**: `pip install <package_name>` ile yükleyin.

---

## Katkıda Bulunanlar
- YTÜ.

---

## Lisans
MIT Lisansı