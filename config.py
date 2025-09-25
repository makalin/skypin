"""
Configuration file for SkyPin
Contains all configuration parameters and settings
"""

import os
from pathlib import Path
from typing import Dict, Any

# Base directory
BASE_DIR = Path(__file__).parent

# Model paths
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)

# Data paths
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

# Output paths
OUTPUT_DIR = BASE_DIR / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

# Configuration settings
CONFIG = {
    # Sun detection settings
    'sun_detection': {
        'model_path': str(MODELS_DIR / "sun_detector.pt"),
        'confidence_threshold': 0.3,
        'min_sun_area': 0.001,  # 0.1% of image
        'max_sun_area': 0.1,    # 10% of image
        'brightness_threshold': 200,
        'saturation_threshold': 30
    },
    
    # Shadow analysis settings
    'shadow_analysis': {
        'min_shadow_length': 50,
        'max_shadow_length': 1000,
        'edge_threshold_low': 50,
        'edge_threshold_high': 150,
        'hough_threshold': 50,
        'min_line_length': 50,
        'max_line_gap': 10
    },
    
    # Astronomy calculation settings
    'astronomy': {
        'grid_resolution': 1.0,  # degrees
        'time_range_hours': 1,   # ±1 hour around timestamp
        'time_step_minutes': 15, # 15-minute intervals
        'lat_range': (-90, 90),
        'lon_range': (-180, 180),
        'azimuth_weight': 0.7,
        'elevation_weight': 0.3,
        'max_error_threshold': 90.0
    },
    
    # Confidence mapping settings
    'confidence_mapping': {
        'kernel_size': 5,
        'confidence_threshold': 0.1,
        'sigma': 0.5,
        'grid_size': 50,
        'contour_levels': [0.1, 0.3, 0.5, 0.7, 0.9]
    },
    
    # Tamper detection settings
    'tamper_detection': {
        'error_level_threshold': 0.1,
        'jpeg_ghost_threshold': 0.05,
        'noise_threshold': 0.02,
        'compression_threshold': 0.1,
        'weights': {
            'error_level': 0.4,
            'jpeg_ghost': 0.3,
            'noise_analysis': 0.2,
            'compression_analysis': 0.1
        }
    },
    
    # Image processing settings
    'image_processing': {
        'max_image_size': (1024, 1024),
        'resize_method': 'area',
        'normalize_range': (0, 1),
        'gaussian_blur_kernel': (5, 5),
        'morphology_kernel_size': (3, 3)
    },
    
    # Streamlit settings
    'streamlit': {
        'page_title': "SkyPin - Celestial GPS",
        'page_icon': "🧭",
        'layout': "wide",
        'initial_sidebar_state': "expanded",
        'max_upload_size': 200 * 1024 * 1024,  # 200MB
        'allowed_file_types': ['jpg', 'jpeg', 'png', 'heic']
    },
    
    # Logging settings
    'logging': {
        'level': 'INFO',
        'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        'file': str(BASE_DIR / "skypin.log"),
        'max_file_size': 10 * 1024 * 1024,  # 10MB
        'backup_count': 5
    },
    
    # Performance settings
    'performance': {
        'max_workers': 4,
        'chunk_size': 1000,
        'cache_size': 100,
        'timeout': 30
    }
}

def get_config(section: str = None) -> Dict[str, Any]:
    """
    Get configuration settings.
    
    Args:
        section: Configuration section name. If None, returns all config.
        
    Returns:
        Configuration dictionary
    """
    if section is None:
        return CONFIG
    return CONFIG.get(section, {})

def update_config(section: str, key: str, value: Any) -> None:
    """
    Update configuration setting.
    
    Args:
        section: Configuration section name
        key: Configuration key
        value: New value
    """
    if section in CONFIG:
        CONFIG[section][key] = value

def get_model_path(model_name: str) -> str:
    """
    Get model file path.
    
    Args:
        model_name: Model name
        
    Returns:
        Model file path
    """
    return str(MODELS_DIR / model_name)

def get_data_path(filename: str) -> str:
    """
    Get data file path.
    
    Args:
        filename: Data filename
        
    Returns:
        Data file path
    """
    return str(DATA_DIR / filename)

def get_output_path(filename: str) -> str:
    """
    Get output file path.
    
    Args:
        filename: Output filename
        
    Returns:
        Output file path
    """
    return str(OUTPUT_DIR / filename)

def create_directories() -> None:
    """Create necessary directories."""
    MODELS_DIR.mkdir(exist_ok=True)
    DATA_DIR.mkdir(exist_ok=True)
    OUTPUT_DIR.mkdir(exist_ok=True)

def validate_config() -> bool:
    """
    Validate configuration settings.
    
    Returns:
        True if configuration is valid, False otherwise
    """
    try:
        # Check required sections
        required_sections = [
            'sun_detection', 'shadow_analysis', 'astronomy',
            'confidence_mapping', 'tamper_detection', 'image_processing'
        ]
        
        for section in required_sections:
            if section not in CONFIG:
                print(f"Missing configuration section: {section}")
                return False
        
        # Check required keys in each section
        required_keys = {
            'sun_detection': ['model_path', 'confidence_threshold'],
            'shadow_analysis': ['min_shadow_length', 'max_shadow_length'],
            'astronomy': ['grid_resolution', 'lat_range', 'lon_range'],
            'confidence_mapping': ['kernel_size', 'confidence_threshold'],
            'tamper_detection': ['error_level_threshold'],
            'image_processing': ['max_image_size']
        }
        
        for section, keys in required_keys.items():
            for key in keys:
                if key not in CONFIG[section]:
                    print(f"Missing configuration key: {section}.{key}")
                    return False
        
        return True
        
    except Exception as e:
        print(f"Configuration validation failed: {e}")
        return False

# Environment-specific configurations
def get_environment_config() -> Dict[str, Any]:
    """Get environment-specific configuration."""
    env = os.getenv('SKYPIN_ENV', 'development')
    
    if env == 'production':
        return {
            'debug': False,
            'log_level': 'WARNING',
            'max_upload_size': 100 * 1024 * 1024,  # 100MB
            'cache_enabled': True,
            'performance_mode': True
        }
    elif env == 'testing':
        return {
            'debug': True,
            'log_level': 'DEBUG',
            'max_upload_size': 10 * 1024 * 1024,   # 10MB
            'cache_enabled': False,
            'performance_mode': False
        }
    else:  # development
        return {
            'debug': True,
            'log_level': 'INFO',
            'max_upload_size': 200 * 1024 * 1024,  # 200MB
            'cache_enabled': True,
            'performance_mode': False
        }

# Initialize configuration
create_directories()

# Validate configuration
if not validate_config():
    print("Warning: Configuration validation failed")