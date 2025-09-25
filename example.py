#!/usr/bin/env python3
"""
Example script for SkyPin
Demonstrates how to use the SkyPin modules programmatically
"""

import numpy as np
from PIL import Image
import io
from datetime import datetime, timezone

# Import SkyPin modules
from modules.exif_extractor import extract_exif_data
from modules.sun_detector import detect_sun_position
from modules.shadow_analyzer import analyze_shadows
from modules.astronomy_calculator import calculate_location
from modules.confidence_mapper import create_confidence_map
from modules.tamper_detector import detect_tampering
from modules.utils import validate_image, format_coordinates

def create_sample_image():
    """Create a sample image for testing."""
    # Create a simple test image with a bright spot (simulating sun)
    image = np.random.randint(50, 150, (400, 400, 3), dtype=np.uint8)
    
    # Add a bright spot in the center (simulating sun)
    center_y, center_x = 200, 200
    for y in range(center_y-20, center_y+20):
        for x in range(center_x-20, center_x+20):
            if 0 <= y < 400 and 0 <= x < 400:
                image[y, x] = [255, 255, 200]  # Bright yellow-white
    
    # Add a shadow line (simulating shadow)
    for y in range(100, 300):
        for x in range(150, 155):
            if 0 <= y < 400 and 0 <= x < 400:
                image[y, x] = [50, 50, 50]  # Dark shadow
    
    return image

def create_sample_exif_data():
    """Create sample EXIF data."""
    return {
        'timestamp': datetime.now(timezone.utc),
        'make': 'Example Camera',
        'model': 'Test Model',
        'focal_length': '50mm',
        'orientation': 'Normal'
    }

def main():
    """Main example function."""
    print("🧭 SkyPin Example")
    print("=" * 50)
    
    # Create sample image
    print("Creating sample image...")
    image = create_sample_image()
    
    # Validate image
    print("Validating image...")
    if validate_image(image):
        print("✓ Image is valid")
    else:
        print("✗ Image validation failed")
        return
    
    # Create sample EXIF data
    print("Creating sample EXIF data...")
    exif_data = create_sample_exif_data()
    print(f"✓ EXIF data: {exif_data['timestamp']}")
    
    # Detect sun position
    print("Detecting sun position...")
    sun_result = detect_sun_position(image)
    print(f"✓ Sun detection: {sun_result['detected']}")
    if sun_result['detected']:
        print(f"  Azimuth: {sun_result['azimuth']:.1f}°")
        print(f"  Elevation: {sun_result['elevation']:.1f}°")
        print(f"  Confidence: {sun_result['confidence']:.2f}")
    
    # Analyze shadows
    print("Analyzing shadows...")
    shadow_result = analyze_shadows(image)
    print(f"✓ Shadow analysis: {shadow_result['detected']}")
    if shadow_result['detected']:
        print(f"  Shadow azimuth: {shadow_result['azimuth']:.1f}°")
        print(f"  Quality: {shadow_result['quality']:.2f}")
    
    # Detect tampering
    print("Detecting tampering...")
    tamper_result = detect_tampering(image)
    print(f"✓ Tamper detection: {'Tampered' if tamper_result['is_tampered'] else 'Authentic'}")
    print(f"  Score: {tamper_result['score']:.2f}")
    
    # Calculate location (if we have sun data)
    if sun_result['detected']:
        print("Calculating location...")
        location_result = calculate_location(
            sun_result, 
            shadow_result, 
            exif_data, 
            grid_resolution=1.0
        )
        
        if location_result['best_location']:
            best_loc = location_result['best_location']
            lat_str, lon_str = format_coordinates(best_loc['latitude'], best_loc['longitude'])
            print(f"✓ Location found:")
            print(f"  Latitude: {lat_str}")
            print(f"  Longitude: {lon_str}")
            print(f"  Confidence: {best_loc['confidence']:.2f}")
            
            # Create confidence map
            print("Creating confidence map...")
            confidence_map = create_confidence_map(location_result)
            print(f"✓ Confidence map created with {confidence_map['num_points']} points")
        else:
            print("✗ No location found")
    else:
        print("Skipping location calculation (no sun detected)")
    
    print("\n🎉 Example completed successfully!")
    print("\nTo run the full SkyPin application:")
    print("  python run.py")

if __name__ == "__main__":
    main()