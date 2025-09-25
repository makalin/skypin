# 🧭 SkyPin  
**Tek bir fotoğraftan Celestial-GPS: Güneş, gölgeler ve zaman damgası kullanarak Dünya'da nerede çekildiğini belirle.**

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)](https://streamlit.io)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.8+-orange.svg)](https://opencv.org)

---

## 30 saniyelik tanıtım  
Akıllı telefon fotoğrafını (veya herhangi bir JPG) SkyPin'e yükle.  
Gökyüzü görünürse, Güneş'in azimut ve yükseklik açısını tahmin edeceğiz, astronomik denklemleri tersine çevireceğiz ve size **olası enlem/boylam ısı haritası** vereceğiz—genellikle birkaç kilometre yarıçapında.  
GPS gerekmez, sokak adresi vaat edilmez: sadece **"göksel üçgenleme"**.

---

## ⚙️ Hızlı Başlangıç

### Kurulum
```bash
git clone https://github.com/makalin/skypin.git
cd skypin
python setup.py
```

### SkyPin'i Çalıştırma
```bash
# Web Arayüzü
python run.py
# http://localhost:8501 adresini açar

# Komut Satırı Arayüzü
python cli.py analyze image.jpg --output results.json

# REST API Sunucusu
python cli.py api --port 5000
```

### Test Etme
```bash
python test_skypin.py
```

---

## 🌟 Yeni Gelişmiş Özellikler

### 🌤️ **Hava Durumu ve Bulut Analizi**
- Hava durumu tahmini ile **çok spektral bulut tespiti**
- **Gökyüzü bölgesi analizi** ve bulut kaplama hesaplama
- Gelişmiş doğruluk için **hava durumu API entegrasyonu**

### 🌙 **Gece Saatleri Konumlandırma**
- Faz analizi ve dairesel tespit ile **ay tespiti**
- Göksel kutup hesaplama için **yıldız izi analizi**
- **Takımyıldız tespiti** ve desen tanıma

### 🖼️ **Gelişmiş Görüntü İşleme**
- **CLAHE geliştirme**, histogram eşitleme, gürültü azaltma
- **Parlaklık düzeltme**, renk dengesi, lens distorsiyonu giderme
- Kapsamlı doğrulama ile **kalite değerlendirmesi**

### 📊 **Toplu İşleme ve Veritabanı**
- Büyük veri setleri için **çok iş parçacıklı toplu işleme**
- Kapsamlı şema ile **SQLite veritabanı**
- Geri çağırma tabanlı güncellemeler ile **ilerleme takibi**

### 🌐 **Çoklu Arayüzler**
- Sürükle-bırak ile **güzel Streamlit uygulaması**
- CORS desteği ile **tam Flask tabanlı API**
- Tüm işlevsellik ile **kapsamlı CLI**

### 📤 **Dışa Aktarma ve Görselleştirme**
- **Çoklu formatlar**: JSON, CSV, XML, KML, GeoJSON, HTML, PDF
- Güven ısı haritaları ile **etkileşimli haritalar**
- Analiz sonuçları için **görselleştirme araçları**

### 🔍 **Doğrulama ve Kalite Kontrolü**
- Kalite ve tutarlılık kontrolü ile **sonuç doğrulama**
- **Görüntü kalitesi değerlendirmesi** (parlaklık, kontrast, keskinlik)
- Doğruluk metrikleri ile **gerçek veri karşılaştırması**

### 📈 **Performans ve İzleme**
- CPU, bellek, disk kullanımının **gerçek zamanlı izlenmesi**
- Yürütme süresi ve bellek kullanımı için **fonksiyon profilleme**
- Otomatik öneriler ile **optimizasyon araçları**

---

## 🔍 Motorun altında neler oluyor  
1. **EXIF** – zaman damgası, yönelim, odak uzaklığını çıkar.  
2. **Güneş tespitçisi** – YOLOv8 modeli sınırlayıcı kutu + lens parlaması merkezi döndürür.  
3. **Gölge vektörü** – Canny + Hough en uzun dikey kenarı bulur; kaybolan nokta matematiği → azimut.  
4. **Bulut tespiti** – Hava durumu koşulları için çok spektral analiz.
5. **Ay tespiti** – Gece saatleri için faz analizi ile dairesel tespit.
6. **Yıldız izleri** – Yıldız desenlerinden göksel kutup hesaplama.
7. **Astronomi tersine çevirme** – [Skyfield](https://github.com/skyfielders/python-skyfield) ile kaba kuvvet 1°×1° dünya ızgarası; (gözlemlenen − tahmin edilen) Güneş (az, el) minimize et.  
8. **Güven yüzeyi** – en iyi %1 eşleşmelerin çekirdek yoğunluğu → GeoJSON ısı haritası.  
9. **Rapor** – merkez koordinat, 1-σ elips, km belirsizlik, sahtecilik skoru.

---

## 📸 Giriş gereksinimleri  
| Olması gereken | Olması iyi olan |
|-----------|--------------|
| Gökyüzü veya keskin gölge görünür | Orijinal JPG/HEIC (yeniden kaydetme yok) |
| Kabaca zaman damgası (±1 saat) | Kamera yüksekliği ve yönelimi |
| — | Bilinen saat dilimi |

---

## 📊 Doğruluk (kıyaslama)  
| Veri seti | Medyan hata | 95-percentil |
|---------|--------------|---------------|
| Topluluk kaynaklı 1000 açık hava fotoğrafı | 180 km | 420 km |
| "Altın saat" alt kümesi (düşük Güneş) | 90 km | 220 km |
| Ay/yıldızlarla gece saatleri | 250 km | 500 km |

---

## 🏗️ Gelişmiş Proje Yapısı

```
skypin/
├── 📱 Ana Uygulamalar
│   ├── app.py                 # Streamlit web uygulaması
│   ├── api_server.py          # REST API sunucusu
│   ├── cli.py                 # Komut satırı arayüzü
│   └── run.py                 # Kolay çalıştırma scripti
│
├── ⚙️ Yapılandırma ve Kurulum
│   ├── config.py              # Kapsamlı yapılandırma
│   ├── setup.py               # Otomatik kurulum scripti
│   ├── requirements.txt       # Tüm bağımlılıklar
│   └── .gitignore            # Git ignore kuralları
│
├── 🧪 Test ve Örnekler
│   ├── test_skypin.py         # Kapsamlı test paketi
│   └── example.py             # Kullanım örnekleri
│
├── 📚 Dokümantasyon
│   ├── README.md              # Tam dokümantasyon
│   └── LICENSE                # MIT Lisansı
│
└── 📦 Ana Modüller (17 modül)
    ├── exif_extractor.py      # EXIF veri çıkarma
    ├── sun_detector.py        # YOLOv8 ile güneş tespiti
    ├── shadow_analyzer.py     # Gölge analizi
    ├── astronomy_calculator.py # Astronomik hesaplamalar
    ├── confidence_mapper.py   # Güven haritalama
    ├── tamper_detector.py     # Sahtecilik tespiti
    ├── cloud_detector.py      # Bulut tespiti ve hava durumu
    ├── moon_detector.py       # Gece saatleri için ay tespiti
    ├── star_tracker.py        # Yıldız izi analizi
    ├── image_enhancer.py      # Görüntü geliştirme araçları
    ├── batch_processor.py     # Toplu işleme
    ├── database_manager.py    # Veritabanı yönetimi
    ├── export_tools.py        # Çeşitli formatlara dışa aktarma
    ├── validation_tools.py    # Kalite değerlendirmesi
    ├── performance_tools.py   # Performans izleme
    └── utils.py               # Yardımcı fonksiyonlar
```

---

## 🚀 Kullanım Örnekleri

### Web Arayüzü
```bash
python run.py
# Sürükle-bırak arayüzü ile http://localhost:8501'i açar
```

### Komut Satırı Arayüzü
```bash
# Tek görüntü analizi
python cli.py analyze image.jpg --output results.json

# Dizin toplu işleme
python cli.py batch /path/to/images --output batch_results.json

# Sonuçları doğrula
python cli.py validate results.json

# Farklı formatlara dışa aktar
python cli.py export results.json --format csv --output results.csv

# API sunucusunu başlat
python cli.py api --port 5000

# Performans izleme
python cli.py performance --start
python cli.py performance --summary
```

### REST API
```bash
# API sunucusunu başlat
python api_server.py

# API ile görüntü analizi
curl -X POST http://localhost:5000/analyze \
  -H "Content-Type: application/json" \
  -d '{"image": "base64_encoded_image_data"}'

# API ile toplu işleme
curl -X POST http://localhost:5000/batch/process \
  -H "Content-Type: application/json" \
  -d '{"images": ["base64_image1", "base64_image2"]}'
```

### Programatik Kullanım
```python
from modules.sun_detector import detect_sun_position
from modules.cloud_detector import detect_clouds
from modules.moon_detector import detect_moon
from modules.batch_processor import process_directory
from modules.export_tools import export_to_kml

# Çeşitli göksel nesneleri tespit et
sun_result = detect_sun_position(image)
cloud_result = detect_clouds(image)
moon_result = detect_moon(image)

# Dizin toplu işleme
results = process_directory("/path/to/images")

# Google Earth için KML'e dışa aktar
export_to_kml(results, "locations.kml")
```

---

## 🛠️ Gelişmiş Özellikler

### 🌤️ Hava Durumu Entegrasyonu
- Bulut kaplama analizi
- Hava durumu koşulu tahmini
- Harici hava durumu API desteği

### 🌙 Gece Saatleri Yetenekleri
- Ay fazı tespiti
- Yıldız izi analizi
- Göksel kutup hesaplama

### 📊 Toplu İşleme
- Çok iş parçacıklı işleme
- İlerleme geri çağırımları
- İstatistiksel analiz

### 💾 Veritabanı Yönetimi
- SQLite depolama
- Sonuç önbellekleme
- Geçmiş analiz

### 📈 Performans İzleme
- Gerçek zamanlı metrikler
- Fonksiyon profilleme
- Bellek takibi
- Optimizasyon önerileri

### 🔍 Kalite Kontrolü
- Görüntü kalitesi değerlendirmesi
- Sonuç doğrulama
- Tutarlılık kontrolü
- Gerçek veri karşılaştırması

---

## 🧪 Test ve Doğrulama

```bash
# Kapsamlı test paketini çalıştır
python test_skypin.py

# Belirli modülleri test et
python -c "from modules.cloud_detector import detect_clouds; print('Bulut tespiti OK')"

# Performans testi
python cli.py performance --start
python test_skypin.py
python cli.py performance --summary
```

---

## 📊 Performans Kıyaslamaları

| İşlem | Süre | Bellek |
|-----------|------|--------|
| Tek görüntü analizi | 2-5s | 200-500MB |
| Toplu işleme (100 görüntü) | 5-10dk | 1-2GB |
| Veritabanı işlemleri | <100ms | 50-100MB |
| API yanıt süresi | <1s | 100-200MB |

---

## 🔧 Yapılandırma

SkyPin, `config.py` aracılığıyla kapsamlı yapılandırma destekler:

```python
# Güneş tespiti ayarları
CONFIG['sun_detection']['confidence_threshold'] = 0.3

# Astronomi hesaplama ayarları
CONFIG['astronomy']['grid_resolution'] = 1.0

# Performans ayarları
CONFIG['performance']['max_workers'] = 4
```

---

## 🤝 Katkıda Bulunma  
PR'lar hoş geldi—`good-first-issue` etiketlerine bakın:  
- Eğik drone görüntülerinde gölge tespitçisi eğitimi  
- İstemci tarafı gizlilik için çekirdek tersine çevirmeyi WebAssembly'ye taşıma  
- Sağdan sola diller için yerelleştirme (i18n)
- Gezegen nesneleri desteği ekleme (Venüs, Mars)
- Makine öğrenmesi tabanlı bulut sınıflandırması uygulama

---

## 📄 Lisans ve atıf  
MIT © 2025 Mehmet T. AKALIN  
SkyPin'i araştırmada kullanırsanız, lütfen şu şekilde atıf yapın:

```bibtex
@software{skypin2025,
  title = {SkyPin: Open-source celestial geolocation toolkit},
  url  = {https://github.com/makalin/skypin},
  year = {2025}
}
```

---

## 🙋‍♂️ SSS  
**S: GPS'i geçebilir mi?**  
C: Hayır—GPS metreler; SkyPin *"hangi bölge?"* doğruluğu.  
**S: Kapalı mekanlarda çalışır mı?**  
C: Sadece pencere gökyüzü/gölge gösteriyorsa.  
**S: Sahte fotoğraflar?**  
C: Konum adımından önce sahtecilik tespiti (Error-Level & JPEG ghost analizi) çalıştır.  
**S: Gece saatleri fotoğrafları?**  
C: Evet! Gece saatleri konumlandırma için ay tespiti veya yıldız izi analizi kullan.  
**S: Toplu işleme?**  
C: Evet! Çok iş parçacıklı toplu işlemcimizle yüzlerce görüntüyü işle.  
**S: API erişimi?**  
C: Evet! JSON yanıtları ve CORS desteği ile tam REST API.

---

## 🎯 Yol Haritası

- [ ] **Makine Öğrenmesi**: Daha iyi tespit için özel modeller eğitme
- [ ] **WebAssembly**: Gizlilik için istemci tarafı işleme
- [ ] **Mobil Uygulama**: Yerel mobil uygulama
- [ ] **Bulut Servisi**: Barındırılan API servisi
- [ ] **Gerçek Zamanlı**: Canlı kamera akışı analizi
- [ ] **Uydu Entegrasyonu**: Uydu görüntüleri ile birleştirme

---

*"Pikseler gezegenlerle buluştuğunda."*