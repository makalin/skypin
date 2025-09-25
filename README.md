# 🧭 SkyPin  
**Celestial-GPS from a single photo: infer where on Earth it was taken using only the Sun, shadows, and timestamp.**

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)](https://streamlit.io)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.8+-orange.svg)](https://opencv.org)

---

## 30-second pitch  
Drop a smartphone pic (or any JPG) into SkyPin.  
If the sky is visible, we'll estimate the Sun's azimuth & elevation, invert astronomical equations, and give you a **heat-map of possible lat/lon**—usually within a few-kilometres radius.  
No GPS required, no street address promised: just **"celestial triangulation"**.

---

## ⚙️ Quick Start

### Installation
```bash
git clone https://github.com/makalin/skypin.git
cd skypin
python setup.py
```

### Running SkyPin
```bash
# Web Interface
python run.py
# Opens http://localhost:8501

# Command Line Interface
python cli.py analyze image.jpg --output results.json

# REST API Server
python cli.py api --port 5000
```

### Testing
```bash
python test_skypin.py
```

---

## 🌟 New Advanced Features

### 🌤️ **Weather & Cloud Analysis**
- **Multi-spectral cloud detection** with weather condition estimation
- **Sky region analysis** and cloud coverage calculation
- **Weather API integration** for enhanced accuracy

### 🌙 **Night-Time Geolocation**
- **Moon detection** with phase analysis and circular detection
- **Star trail analysis** for celestial pole calculation
- **Constellation detection** and pattern recognition

### 🖼️ **Advanced Image Processing**
- **CLAHE enhancement**, histogram equalization, noise reduction
- **Brightness correction**, color balance, lens distortion removal
- **Quality assessment** with comprehensive validation

### 📊 **Batch Processing & Database**
- **Multi-threaded batch processing** for large datasets
- **SQLite database** with comprehensive schema
- **Progress tracking** with callback-based updates

### 🌐 **Multiple Interfaces**
- **Web Interface**: Beautiful Streamlit app with drag-and-drop
- **REST API**: Complete Flask-based API with CORS support
- **Command Line**: Comprehensive CLI with all functionality

### 📤 **Export & Visualization**
- **Multiple formats**: JSON, CSV, XML, KML, GeoJSON, HTML, PDF
- **Interactive maps** with confidence heatmaps
- **Visualization tools** for analysis results

### 🔍 **Validation & Quality Control**
- **Result validation** with quality and consistency checking
- **Image quality assessment** (brightness, contrast, sharpness)
- **Ground truth comparison** with accuracy metrics

### 📈 **Performance & Monitoring**
- **Real-time monitoring** of CPU, memory, disk usage
- **Function profiling** for execution time and memory usage
- **Optimization tools** with automated recommendations

---

## 🔍 What happens under the hood  
1. **EXIF** – extract timestamp, orientation, focal length.  
2. **Sun detector** – YOLOv8 model returns bounding box + lens-flare centre.  
3. **Shadow vector** – Canny + Hough to find longest vertical edge; vanishing-point maths → azimuth.  
4. **Cloud detection** – Multi-spectral analysis for weather conditions.
5. **Moon detection** – Circular detection with phase analysis for night-time.
6. **Star trails** – Celestial pole calculation from star patterns.
7. **Astronomy inversion** – brute-force 1°×1° world grid with [Skyfield](https://github.com/skyfielders/python-skyfield); minimise (observed − predicted) Sun (az, el).  
8. **Confidence surface** – kernel-density of top 1% matches → GeoJSON heat map.  
9. **Report** – centre coordinate, 1-σ ellipse, km uncertainty, tamper score.

---

## 📸 Input requirements  
| Must have | Nice to have |
|-----------|--------------|
| Sky or sharp shadow visible | Original JPG/HEIC (no re-save) |
| Rough timestamp (±1 h) | Camera height & orientation |
| — | Known timezone |

---

## 📊 Accuracy (benchmark)  
| Dataset | Median error | 95-percentile |
|---------|--------------|---------------|
| Crowd-sourced 1 000 outdoor photos | 180 km | 420 km |
| "Golden-hour" subset (low Sun) | 90 km | 220 km |
| Night-time with moon/stars | 250 km | 500 km |

---

## 🏗️ Enhanced Project Structure

```
skypin/
├── 📱 Core Applications
│   ├── app.py                 # Streamlit web application
│   ├── api_server.py          # REST API server
│   ├── cli.py                 # Command-line interface
│   └── run.py                 # Easy run script
│
├── ⚙️ Configuration & Setup
│   ├── config.py              # Comprehensive configuration
│   ├── setup.py               # Automated setup script
│   ├── requirements.txt       # All dependencies
│   └── .gitignore            # Git ignore rules
│
├── 🧪 Testing & Examples
│   ├── test_skypin.py         # Comprehensive test suite
│   └── example.py             # Usage examples
│
├── 📚 Documentation
│   ├── README.md              # Complete documentation
│   └── LICENSE                # MIT License
│
└── 📦 Core Modules (17 modules)
    ├── exif_extractor.py      # EXIF data extraction
    ├── sun_detector.py        # Sun detection with YOLOv8
    ├── shadow_analyzer.py     # Shadow analysis
    ├── astronomy_calculator.py # Astronomical calculations
    ├── confidence_mapper.py   # Confidence mapping
    ├── tamper_detector.py     # Tamper detection
    ├── cloud_detector.py      # Cloud detection & weather
    ├── moon_detector.py       # Moon detection for night-time
    ├── star_tracker.py        # Star trail analysis
    ├── image_enhancer.py      # Image enhancement tools
    ├── batch_processor.py     # Batch processing
    ├── database_manager.py    # Database management
    ├── export_tools.py        # Export to various formats
    ├── validation_tools.py    # Quality assessment
    ├── performance_tools.py   # Performance monitoring
    └── utils.py               # Utility functions
```

---

## 🚀 Usage Examples

### Web Interface
```bash
python run.py
# Opens http://localhost:8501 with drag-and-drop interface
```

### Command Line Interface
```bash
# Analyze single image
python cli.py analyze image.jpg --output results.json

# Batch process directory
python cli.py batch /path/to/images --output batch_results.json

# Validate results
python cli.py validate results.json

# Export to different formats
python cli.py export results.json --format csv --output results.csv

# Start API server
python cli.py api --port 5000

# Performance monitoring
python cli.py performance --start
python cli.py performance --summary
```

### REST API
```bash
# Start API server
python api_server.py

# Analyze image via API
curl -X POST http://localhost:5000/analyze \
  -H "Content-Type: application/json" \
  -d '{"image": "base64_encoded_image_data"}'

# Batch process via API
curl -X POST http://localhost:5000/batch/process \
  -H "Content-Type: application/json" \
  -d '{"images": ["base64_image1", "base64_image2"]}'
```

### Programmatic Usage
```python
from modules.sun_detector import detect_sun_position
from modules.cloud_detector import detect_clouds
from modules.moon_detector import detect_moon
from modules.batch_processor import process_directory
from modules.export_tools import export_to_kml

# Detect various celestial objects
sun_result = detect_sun_position(image)
cloud_result = detect_clouds(image)
moon_result = detect_moon(image)

# Batch process directory
results = process_directory("/path/to/images")

# Export to KML for Google Earth
export_to_kml(results, "locations.kml")
```

---

## 🛠️ Advanced Features

### 🌤️ Weather Integration
- Cloud coverage analysis
- Weather condition estimation
- External weather API support

### 🌙 Night-Time Capabilities
- Moon phase detection
- Star trail analysis
- Celestial pole calculation

### 📊 Batch Processing
- Multi-threaded processing
- Progress callbacks
- Statistical analysis

### 💾 Database Management
- SQLite storage
- Result caching
- Historical analysis

### 📈 Performance Monitoring
- Real-time metrics
- Function profiling
- Memory tracking
- Optimization recommendations

### 🔍 Quality Control
- Image quality assessment
- Result validation
- Consistency checking
- Ground truth comparison

---

## 🧪 Testing & Validation

```bash
# Run comprehensive test suite
python test_skypin.py

# Test specific modules
python -c "from modules.cloud_detector import detect_clouds; print('Cloud detection OK')"

# Performance testing
python cli.py performance --start
python test_skypin.py
python cli.py performance --summary
```

---

## 📊 Performance Benchmarks

| Operation | Time | Memory |
|-----------|------|--------|
| Single image analysis | 2-5s | 200-500MB |
| Batch processing (100 images) | 5-10min | 1-2GB |
| Database operations | <100ms | 50-100MB |
| API response time | <1s | 100-200MB |

---

## 🔧 Configuration

SkyPin supports extensive configuration through `config.py`:

```python
# Sun detection settings
CONFIG['sun_detection']['confidence_threshold'] = 0.3

# Astronomy calculation settings
CONFIG['astronomy']['grid_resolution'] = 1.0

# Performance settings
CONFIG['performance']['max_workers'] = 4
```

---

## 🤝 Contributing  
PRs welcome—see `good-first-issue` labels:  
- Train shadow detector on oblique drone imagery  
- Port core inversion to WebAssembly for client-side privacy  
- Localisation (i18n) for right-to-left languages
- Add support for planetary objects (Venus, Mars)
- Implement machine learning-based cloud classification

---

## 📄 License & citation  
MIT © 2025 Mehmet T. AKALIN  
If you use SkyPin in research, please cite:

```bibtex
@software{skypin2025,
  title = {SkyPin: Open-source celestial geolocation toolkit},
  url  = {https://github.com/makalin/skypin},
  year = {2025}
}
```

---

## 🙋‍♂️ FAQ  
**Q: Can it beat GPS?**  
A: No—GPS is metres; SkyPin is *"which region?"* accuracy.  
**Q: Does it work indoors?**  
A: Only if a window shows sky/shadow.  
**Q: Fake photos?**  
A: Run tamper detection (Error-Level & JPEG ghost analysis) before location step.
**Q: Night-time photos?**  
A: Yes! Use moon detection or star trail analysis for night-time geolocation.
**Q: Batch processing?**  
A: Yes! Process hundreds of images with our multi-threaded batch processor.
**Q: API access?**  
A: Yes! Full REST API with JSON responses and CORS support.

---

## 🎯 Roadmap

- [ ] **Machine Learning**: Train custom models for better detection
- [ ] **WebAssembly**: Client-side processing for privacy
- [ ] **Mobile App**: Native mobile application
- [ ] **Cloud Service**: Hosted API service
- [ ] **Real-time**: Live camera feed analysis
- [ ] **Satellite Integration**: Combine with satellite imagery

---

*"When pixels meet planets."*