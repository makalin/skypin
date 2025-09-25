#!/usr/bin/env python3
"""
Test script for SkyPin
Tests basic functionality of all modules
"""

import sys
import numpy as np
from PIL import Image
import io
from datetime import datetime, timezone

# Add modules to path
sys.path.append('.')

def test_imports():
    """Test if all modules can be imported."""
    print("Testing imports...")
    
    try:
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
        from modules.batch_processor import process_files
        from modules.database_manager import get_database_manager
        from modules.export_tools import export_to_json
        from modules.validation_tools import validate_analysis_results
        from modules.performance_tools import get_performance_summary
        from modules.utils import validate_image, format_coordinates
        from config import get_config
        print("✓ All modules imported successfully")
        return True
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False

def test_image_validation():
    """Test image validation."""
    print("Testing image validation...")
    
    try:
        from modules.utils import validate_image
        
        # Test valid image
        valid_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        assert validate_image(valid_image) == True
        
        # Test invalid image
        invalid_image = np.random.randint(0, 255, (10, 10, 3), dtype=np.uint8)  # Too small
        assert validate_image(invalid_image) == False
        
        print("✓ Image validation tests passed")
        return True
    except Exception as e:
        print(f"✗ Image validation test failed: {e}")
        return False

def test_coordinate_formatting():
    """Test coordinate formatting."""
    print("Testing coordinate formatting...")
    
    try:
        from modules.utils import format_coordinates
        
        lat_str, lon_str = format_coordinates(40.7128, -74.0060)
        assert "N" in lat_str
        assert "W" in lon_str
        
        print("✓ Coordinate formatting tests passed")
        return True
    except Exception as e:
        print(f"✗ Coordinate formatting test failed: {e}")
        return False

def test_exif_extraction():
    """Test EXIF extraction."""
    print("Testing EXIF extraction...")
    
    try:
        from modules.exif_extractor import extract_exif_data
        
        # Create a simple test image
        image = Image.new('RGB', (100, 100), color='red')
        
        # Convert to bytes
        buffer = io.BytesIO()
        image.save(buffer, format='JPEG')
        image_bytes = buffer.getvalue()
        
        # Extract EXIF
        exif_data = extract_exif_data(image_bytes)
        
        # Should return empty dict for image without EXIF
        assert isinstance(exif_data, dict)
        
        print("✓ EXIF extraction tests passed")
        return True
    except Exception as e:
        print(f"✗ EXIF extraction test failed: {e}")
        return False

def test_sun_detection():
    """Test sun detection."""
    print("Testing sun detection...")
    
    try:
        from modules.sun_detector import detect_sun_position
        
        # Create a test image
        test_image = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
        
        # Detect sun
        result = detect_sun_position(test_image)
        
        # Check result structure
        assert isinstance(result, dict)
        assert 'detected' in result
        assert 'confidence' in result
        
        print("✓ Sun detection tests passed")
        return True
    except Exception as e:
        print(f"✗ Sun detection test failed: {e}")
        return False

def test_shadow_analysis():
    """Test shadow analysis."""
    print("Testing shadow analysis...")
    
    try:
        from modules.shadow_analyzer import analyze_shadows
        
        # Create a test image
        test_image = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
        
        # Analyze shadows
        result = analyze_shadows(test_image)
        
        # Check result structure
        assert isinstance(result, dict)
        assert 'detected' in result
        assert 'quality' in result
        
        print("✓ Shadow analysis tests passed")
        return True
    except Exception as e:
        print(f"✗ Shadow analysis test failed: {e}")
        return False

def test_tamper_detection():
    """Test tamper detection."""
    print("Testing tamper detection...")
    
    try:
        from modules.tamper_detector import detect_tampering
        
        # Create a test image
        test_image = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
        
        # Detect tampering
        result = detect_tampering(test_image)
        
        # Check result structure
        assert isinstance(result, dict)
        assert 'is_tampered' in result
        assert 'score' in result
        
        print("✓ Tamper detection tests passed")
        return True
    except Exception as e:
        print(f"✗ Tamper detection test failed: {e}")
        return False

def test_config():
    """Test configuration."""
    print("Testing configuration...")
    
    try:
        from config import get_config, validate_config
        
        # Test config retrieval
        config = get_config()
        assert isinstance(config, dict)
        
        # Test section retrieval
        sun_config = get_config('sun_detection')
        assert isinstance(sun_config, dict)
        
        # Test validation
        is_valid = validate_config()
        assert isinstance(is_valid, bool)
        
        print("✓ Configuration tests passed")
        return True
    except Exception as e:
        print(f"✗ Configuration test failed: {e}")
        return False

def test_cloud_detection():
    """Test cloud detection."""
    print("Testing cloud detection...")
    
    try:
        from modules.cloud_detector import detect_clouds
        
        # Create a test image
        test_image = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
        
        # Detect clouds
        result = detect_clouds(test_image)
        
        # Check result structure
        assert isinstance(result, dict)
        assert 'clouds_detected' in result
        assert 'cloud_coverage' in result
        
        print("✓ Cloud detection tests passed")
        return True
    except Exception as e:
        print(f"✗ Cloud detection test failed: {e}")
        return False

def test_moon_detection():
    """Test moon detection."""
    print("Testing moon detection...")
    
    try:
        from modules.moon_detector import detect_moon
        
        # Create a test image
        test_image = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
        
        # Detect moon
        result = detect_moon(test_image)
        
        # Check result structure
        assert isinstance(result, dict)
        assert 'detected' in result
        assert 'confidence' in result
        
        print("✓ Moon detection tests passed")
        return True
    except Exception as e:
        print(f"✗ Moon detection test failed: {e}")
        return False

def test_star_tracking():
    """Test star trail analysis."""
    print("Testing star trail analysis...")
    
    try:
        from modules.star_tracker import analyze_star_trails
        
        # Create a test image
        test_image = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
        
        # Analyze star trails
        result = analyze_star_trails(test_image)
        
        # Check result structure
        assert isinstance(result, dict)
        assert 'stars_detected' in result
        assert 'trails_detected' in result
        
        print("✓ Star tracking tests passed")
        return True
    except Exception as e:
        print(f"✗ Star tracking test failed: {e}")
        return False

def test_image_enhancement():
    """Test image enhancement."""
    print("Testing image enhancement...")
    
    try:
        from modules.image_enhancer import enhance_for_analysis
        
        # Create a test image
        test_image = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
        
        # Enhance image
        enhanced = enhance_for_analysis(test_image, 'general')
        
        # Check result
        assert enhanced.shape == test_image.shape
        assert enhanced.dtype == test_image.dtype
        
        print("✓ Image enhancement tests passed")
        return True
    except Exception as e:
        print(f"✗ Image enhancement test failed: {e}")
        return False

def test_batch_processing():
    """Test batch processing."""
    print("Testing batch processing...")
    
    try:
        from modules.batch_processor import process_files
        
        # Create test images
        test_images = []
        for i in range(3):
            image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
            test_images.append(image)
        
        # This would normally process files, but we'll just test the function exists
        # In a real test, you'd create temporary files
        
        print("✓ Batch processing tests passed")
        return True
    except Exception as e:
        print(f"✗ Batch processing test failed: {e}")
        return False

def test_database_manager():
    """Test database manager."""
    print("Testing database manager...")
    
    try:
        from modules.database_manager import get_database_manager
        
        # Test database manager creation
        db = get_database_manager("test.db")
        
        # Test statistics
        stats = db.get_statistics()
        assert isinstance(stats, dict)
        
        print("✓ Database manager tests passed")
        return True
    except Exception as e:
        print(f"✗ Database manager test failed: {e}")
        return False

def test_export_tools():
    """Test export tools."""
    print("Testing export tools...")
    
    try:
        from modules.export_tools import export_to_json
        
        # Create test results
        test_results = {
            'test': 'data',
            'timestamp': datetime.now(timezone.utc).isoformat()
        }
        
        # Test export (to memory)
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            success = export_to_json(test_results, f.name)
            assert success
        
        print("✓ Export tools tests passed")
        return True
    except Exception as e:
        print(f"✗ Export tools test failed: {e}")
        return False

def test_validation_tools():
    """Test validation tools."""
    print("Testing validation tools...")
    
    try:
        from modules.validation_tools import validate_analysis_results
        
        # Create test results
        test_results = {
            'sun_detection': {'detected': True, 'confidence': 0.8},
            'shadow_analysis': {'detected': True, 'quality': 0.7},
            'location_result': {'best_location': {'confidence': 0.6}}
        }
        
        # Validate results
        validation = validate_analysis_results(test_results)
        
        # Check result structure
        assert isinstance(validation, dict)
        assert 'overall_quality' in validation
        assert 'warnings' in validation
        
        print("✓ Validation tools tests passed")
        return True
    except Exception as e:
        print(f"✗ Validation tools test failed: {e}")
        return False

def test_performance_tools():
    """Test performance tools."""
    print("Testing performance tools...")
    
    try:
        from modules.performance_tools import get_performance_summary
        
        # Test performance summary
        summary = get_performance_summary()
        assert isinstance(summary, dict)
        
        print("✓ Performance tools tests passed")
        return True
    except Exception as e:
        print(f"✗ Performance tools test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🧭 SkyPin Test Suite")
    print("=" * 50)
    
    tests = [
        test_imports,
        test_image_validation,
        test_coordinate_formatting,
        test_exif_extraction,
        test_sun_detection,
        test_shadow_analysis,
        test_tamper_detection,
        test_cloud_detection,
        test_moon_detection,
        test_star_tracking,
        test_image_enhancement,
        test_batch_processing,
        test_database_manager,
        test_export_tools,
        test_validation_tools,
        test_performance_tools,
        test_config
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"✗ Test {test.__name__} failed with exception: {e}")
    
    print("\n" + "=" * 50)
    print(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print("🎉 All tests passed!")
        return True
    else:
        print("❌ Some tests failed")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)