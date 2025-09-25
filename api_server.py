"""
REST API server for SkyPin
Provides programmatic access to SkyPin functionality
"""

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import numpy as np
from PIL import Image
import io
import base64
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
import os
import tempfile
from typing import Dict, Any

# Import SkyPin modules
from modules.exif_extractor import extract_exif_data
from modules.sun_detector import detect_sun_position
from modules.shadow_analyzer import analyze_shadows
from modules.astronomy_calculator import calculate_location
from modules.confidence_mapper import create_confidence_map
from modules.tamper_detector import detect_tampering
from modules.cloud_detector import detect_clouds
from modules.moon_detector import detect_moon
from modules.star_tracker import analyze_star_trails
from modules.image_enhancer import enhance_for_analysis
from modules.batch_processor import process_files, get_processing_statistics
from modules.utils import validate_image, format_coordinates
from config import get_config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Configuration
config = get_config()
MAX_FILE_SIZE = config.get('streamlit', {}).get('max_upload_size', 200 * 1024 * 1024)

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'version': '1.0.0'
    })

@app.route('/analyze', methods=['POST'])
def analyze_image():
    """
    Analyze a single image for location.
    
    Expected JSON payload:
    {
        "image": "base64_encoded_image_data",
        "options": {
            "confidence_threshold": 0.7,
            "grid_resolution": 1.0,
            "enable_tamper_detection": true,
            "timezone_override": "UTC"
        }
    }
    """
    try:
        # Parse request
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({'error': 'No image data provided'}), 400
        
        # Decode image
        try:
            image_data = base64.b64decode(data['image'])
            image = Image.open(io.BytesIO(image_data))
            image_array = np.array(image)
        except Exception as e:
            return jsonify({'error': f'Invalid image data: {str(e)}'}), 400
        
        # Validate image
        if not validate_image(image_array):
            return jsonify({'error': 'Invalid image format'}), 400
        
        # Get options
        options = data.get('options', {})
        confidence_threshold = options.get('confidence_threshold', 0.7)
        grid_resolution = options.get('grid_resolution', 1.0)
        enable_tamper_detection = options.get('enable_tamper_detection', True)
        timezone_override = options.get('timezone_override')
        
        # Extract EXIF data
        exif_data = extract_exif_data(image_data)
        
        # Detect sun position
        sun_result = detect_sun_position(image_array)
        
        # Analyze shadows
        shadow_result = analyze_shadows(image_array)
        
        # Detect clouds
        cloud_result = detect_clouds(image_array)
        
        # Detect moon
        moon_result = detect_moon(image_array)
        
        # Analyze star trails
        star_result = analyze_star_trails(image_array)
        
        # Detect tampering
        tamper_result = None
        if enable_tamper_detection:
            tamper_result = detect_tampering(image_array)
        
        # Calculate location
        location_result = calculate_location(
            sun_result, shadow_result, exif_data, 
            grid_resolution, timezone_override
        )
        
        # Create confidence map
        confidence_map = None
        if location_result.get('best_location'):
            confidence_map = create_confidence_map(location_result, confidence_threshold)
        
        # Format response
        response = {
            'success': True,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'results': {
                'exif_data': exif_data,
                'sun_detection': sun_result,
                'shadow_analysis': shadow_result,
                'cloud_detection': cloud_result,
                'moon_detection': moon_result,
                'star_analysis': star_result,
                'tamper_detection': tamper_result,
                'location_result': location_result,
                'confidence_map': confidence_map
            }
        }
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Analysis error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/detect/sun', methods=['POST'])
def detect_sun():
    """Detect sun position in image."""
    try:
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({'error': 'No image data provided'}), 400
        
        # Decode and process image
        image_data = base64.b64decode(data['image'])
        image = Image.open(io.BytesIO(image_data))
        image_array = np.array(image)
        
        if not validate_image(image_array):
            return jsonify({'error': 'Invalid image format'}), 400
        
        # Detect sun
        result = detect_sun_position(image_array)
        
        return jsonify({
            'success': True,
            'result': result,
            'timestamp': datetime.now(timezone.utc).isoformat()
        })
        
    except Exception as e:
        logger.error(f"Sun detection error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/detect/shadows', methods=['POST'])
def detect_shadows():
    """Analyze shadows in image."""
    try:
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({'error': 'No image data provided'}), 400
        
        # Decode and process image
        image_data = base64.b64decode(data['image'])
        image = Image.open(io.BytesIO(image_data))
        image_array = np.array(image)
        
        if not validate_image(image_array):
            return jsonify({'error': 'Invalid image format'}), 400
        
        # Analyze shadows
        result = analyze_shadows(image_array)
        
        return jsonify({
            'success': True,
            'result': result,
            'timestamp': datetime.now(timezone.utc).isoformat()
        })
        
    except Exception as e:
        logger.error(f"Shadow analysis error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/detect/clouds', methods=['POST'])
def detect_clouds_endpoint():
    """Detect clouds in image."""
    try:
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({'error': 'No image data provided'}), 400
        
        # Decode and process image
        image_data = base64.b64decode(data['image'])
        image = Image.open(io.BytesIO(image_data))
        image_array = np.array(image)
        
        if not validate_image(image_array):
            return jsonify({'error': 'Invalid image format'}), 400
        
        # Detect clouds
        result = detect_clouds(image_array)
        
        return jsonify({
            'success': True,
            'result': result,
            'timestamp': datetime.now(timezone.utc).isoformat()
        })
        
    except Exception as e:
        logger.error(f"Cloud detection error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/detect/moon', methods=['POST'])
def detect_moon_endpoint():
    """Detect moon in image."""
    try:
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({'error': 'No image data provided'}), 400
        
        # Decode and process image
        image_data = base64.b64decode(data['image'])
        image = Image.open(io.BytesIO(image_data))
        image_array = np.array(image)
        
        if not validate_image(image_array):
            return jsonify({'error': 'Invalid image format'}), 400
        
        # Detect moon
        result = detect_moon(image_array)
        
        return jsonify({
            'success': True,
            'result': result,
            'timestamp': datetime.now(timezone.utc).isoformat()
        })
        
    except Exception as e:
        logger.error(f"Moon detection error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/detect/tampering', methods=['POST'])
def detect_tampering_endpoint():
    """Detect image tampering."""
    try:
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({'error': 'No image data provided'}), 400
        
        # Decode and process image
        image_data = base64.b64decode(data['image'])
        image = Image.open(io.BytesIO(image_data))
        image_array = np.array(image)
        
        if not validate_image(image_array):
            return jsonify({'error': 'Invalid image format'}), 400
        
        # Detect tampering
        result = detect_tampering(image_array)
        
        return jsonify({
            'success': True,
            'result': result,
            'timestamp': datetime.now(timezone.utc).isoformat()
        })
        
    except Exception as e:
        logger.error(f"Tamper detection error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/enhance', methods=['POST'])
def enhance_image():
    """Enhance image for analysis."""
    try:
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({'error': 'No image data provided'}), 400
        
        enhancement_type = data.get('enhancement_type', 'general')
        
        # Decode and process image
        image_data = base64.b64decode(data['image'])
        image = Image.open(io.BytesIO(image_data))
        image_array = np.array(image)
        
        if not validate_image(image_array):
            return jsonify({'error': 'Invalid image format'}), 400
        
        # Enhance image
        enhanced = enhance_for_analysis(image_array, enhancement_type)
        
        # Convert back to base64
        enhanced_image = Image.fromarray(enhanced)
        buffer = io.BytesIO()
        enhanced_image.save(buffer, format='PNG')
        enhanced_data = base64.b64encode(buffer.getvalue()).decode()
        
        return jsonify({
            'success': True,
            'enhanced_image': enhanced_data,
            'enhancement_type': enhancement_type,
            'timestamp': datetime.now(timezone.utc).isoformat()
        })
        
    except Exception as e:
        logger.error(f"Image enhancement error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/batch/process', methods=['POST'])
def batch_process():
    """Process multiple images in batch."""
    try:
        data = request.get_json()
        if not data or 'images' not in data:
            return jsonify({'error': 'No images provided'}), 400
        
        images = data['images']
        if not isinstance(images, list):
            return jsonify({'error': 'Images must be a list'}), 400
        
        if len(images) > 100:  # Limit batch size
            return jsonify({'error': 'Batch size too large (max 100 images)'}), 400
        
        # Process images
        file_paths = []
        temp_files = []
        
        try:
            # Save images to temporary files
            for i, image_data in enumerate(images):
                temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
                temp_files.append(temp_file.name)
                
                # Decode and save image
                image_bytes = base64.b64decode(image_data)
                with open(temp_file.name, 'wb') as f:
                    f.write(image_bytes)
                
                file_paths.append(temp_file.name)
            
            # Process files
            results = process_files(file_paths)
            
            return jsonify({
                'success': True,
                'results': results,
                'timestamp': datetime.now(timezone.utc).isoformat()
            })
            
        finally:
            # Clean up temporary files
            for temp_file in temp_files:
                try:
                    os.unlink(temp_file)
                except:
                    pass
        
    except Exception as e:
        logger.error(f"Batch processing error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/batch/statistics', methods=['POST'])
def batch_statistics():
    """Get statistics from batch processing results."""
    try:
        data = request.get_json()
        if not data or 'results' not in data:
            return jsonify({'error': 'No results provided'}), 400
        
        results = data['results']
        stats = get_processing_statistics(results)
        
        return jsonify({
            'success': True,
            'statistics': stats,
            'timestamp': datetime.now(timezone.utc).isoformat()
        })
        
    except Exception as e:
        logger.error(f"Statistics calculation error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/config', methods=['GET'])
def get_configuration():
    """Get current configuration."""
    try:
        return jsonify({
            'success': True,
            'config': config,
            'timestamp': datetime.now(timezone.utc).isoformat()
        })
        
    except Exception as e:
        logger.error(f"Configuration error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/models', methods=['GET'])
def list_models():
    """List available models."""
    try:
        models_dir = Path('models')
        models = []
        
        if models_dir.exists():
            for model_file in models_dir.glob('*.pt'):
                models.append({
                    'name': model_file.name,
                    'path': str(model_file),
                    'size': model_file.stat().st_size,
                    'modified': datetime.fromtimestamp(model_file.stat().st_mtime).isoformat()
                })
        
        return jsonify({
            'success': True,
            'models': models,
            'timestamp': datetime.now(timezone.utc).isoformat()
        })
        
    except Exception as e:
        logger.error(f"Model listing error: {e}")
        return jsonify({'error': str(e)}), 500

@app.errorhandler(413)
def too_large(e):
    """Handle file too large error."""
    return jsonify({'error': 'File too large'}), 413

@app.errorhandler(400)
def bad_request(e):
    """Handle bad request error."""
    return jsonify({'error': 'Bad request'}), 400

@app.errorhandler(500)
def internal_error(e):
    """Handle internal server error."""
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    # Run the API server
    app.run(host='0.0.0.0', port=5000, debug=True)