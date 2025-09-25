"""
SkyPin - Celestial GPS from a single photo
Main Streamlit application
"""

import streamlit as st
import numpy as np
import cv2
from PIL import Image
import io
import json
import folium
from streamlit_folium import st_folium
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timezone
import pandas as pd

from modules.exif_extractor import extract_exif_data
from modules.sun_detector import detect_sun_position
from modules.shadow_analyzer import analyze_shadows
from modules.astronomy_calculator import calculate_location
from modules.confidence_mapper import create_confidence_map
from modules.tamper_detector import detect_tampering
from modules.utils import validate_image, format_coordinates

# Page configuration
st.set_page_config(
    page_title="SkyPin - Celestial GPS",
    page_icon="🧭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        text-align: center;
        color: #1f77b4;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .confidence-high { color: #28a745; }
    .confidence-medium { color: #ffc107; }
    .confidence-low { color: #dc3545; }
</style>
""", unsafe_allow_html=True)

def main():
    # Header
    st.markdown('<h1 class="main-header">🧭 SkyPin - Celestial GPS</h1>', unsafe_allow_html=True)
    st.markdown("**Infer where on Earth a photo was taken using only the Sun, shadows, and timestamp.**")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # Confidence threshold
        confidence_threshold = st.slider(
            "Confidence Threshold", 
            min_value=0.1, 
            max_value=1.0, 
            value=0.7, 
            step=0.1,
            help="Minimum confidence for location estimates"
        )
        
        # Grid resolution
        grid_resolution = st.selectbox(
            "Grid Resolution",
            options=[1.0, 0.5, 0.25],
            index=0,
            format_func=lambda x: f"{x}° × {x}°",
            help="Resolution for astronomical calculations"
        )
        
        # Enable tamper detection
        enable_tamper_detection = st.checkbox(
            "Enable Tamper Detection",
            value=True,
            help="Analyze image for potential manipulation"
        )
        
        # Manual timezone override
        timezone_override = st.text_input(
            "Timezone Override (optional)",
            placeholder="e.g., UTC, America/New_York",
            help="Override EXIF timezone if needed"
        )
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📸 Upload Photo")
        
        # File uploader
        uploaded_file = st.file_uploader(
            "Choose an image file",
            type=['jpg', 'jpeg', 'png', 'heic'],
            help="Upload a photo with visible sky or shadows"
        )
        
        if uploaded_file is not None:
            # Display uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)
            
            # Process button
            if st.button("🔍 Analyze Location", type="primary"):
                with st.spinner("Analyzing image..."):
                    try:
                        # Process the image
                        results = process_image(
                            uploaded_file, 
                            confidence_threshold,
                            grid_resolution,
                            enable_tamper_detection,
                            timezone_override
                        )
                        
                        # Store results in session state
                        st.session_state.results = results
                        
                    except Exception as e:
                        st.error(f"Error processing image: {str(e)}")
                        st.exception(e)
    
    with col2:
        st.header("📍 Results")
        
        if 'results' in st.session_state:
            results = st.session_state.results
            
            # Display results
            display_results(results)
        else:
            st.info("Upload an image and click 'Analyze Location' to see results.")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p><strong>SkyPin</strong> - When pixels meet planets 🌍</p>
        <p>MIT © 2025 Mehmet T. AKALIN</p>
    </div>
    """, unsafe_allow_html=True)

def process_image(uploaded_file, confidence_threshold, grid_resolution, enable_tamper_detection, timezone_override):
    """Process uploaded image and return location analysis results."""
    
    # Read image
    image_bytes = uploaded_file.read()
    image = Image.open(io.BytesIO(image_bytes))
    image_array = np.array(image)
    
    # Validate image
    if not validate_image(image_array):
        raise ValueError("Invalid image format or corrupted file")
    
    # Extract EXIF data
    exif_data = extract_exif_data(image_bytes)
    
    # Tamper detection (if enabled)
    tamper_score = None
    if enable_tamper_detection:
        tamper_score = detect_tampering(image_array)
    
    # Detect sun position
    sun_position = detect_sun_position(image_array)
    
    # Analyze shadows
    shadow_data = analyze_shadows(image_array)
    
    # Calculate location using astronomy
    location_results = calculate_location(
        sun_position, 
        shadow_data, 
        exif_data, 
        grid_resolution,
        timezone_override
    )
    
    # Create confidence map
    confidence_map = create_confidence_map(location_results, confidence_threshold)
    
    # Compile results
    results = {
        'exif_data': exif_data,
        'sun_position': sun_position,
        'shadow_data': shadow_data,
        'location_results': location_results,
        'confidence_map': confidence_map,
        'tamper_score': tamper_score,
        'processing_time': datetime.now().isoformat()
    }
    
    return results

def display_results(results):
    """Display analysis results in an organized format."""
    
    # Location summary
    st.subheader("🎯 Location Estimate")
    
    if results['location_results']['best_location']:
        best_loc = results['location_results']['best_location']
        confidence = results['location_results']['confidence']
        
        # Format coordinates
        lat_str, lon_str = format_coordinates(best_loc['latitude'], best_loc['longitude'])
        
        # Display coordinates
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Latitude", lat_str)
        with col2:
            st.metric("Longitude", lon_str)
        with col3:
            confidence_class = "confidence-high" if confidence > 0.7 else "confidence-medium" if confidence > 0.4 else "confidence-low"
            st.metric("Confidence", f"{confidence:.1%}", delta=None)
        
        # Uncertainty ellipse
        if 'uncertainty_ellipse' in best_loc:
            ellipse = best_loc['uncertainty_ellipse']
            st.info(f"📏 Uncertainty: {ellipse['semi_major']:.1f} km × {ellipse['semi_minor']:.1f} km")
        
        # Interactive map
        st.subheader("🗺️ Location Map")
        create_interactive_map(best_loc, results['confidence_map'])
        
    else:
        st.warning("❌ Could not determine location from this image.")
        st.info("💡 Try uploading an image with:")
        st.markdown("- Clear sky visible")
        st.markdown("- Sharp shadows")
        st.markdown("- Valid EXIF timestamp")
    
    # Technical details
    with st.expander("🔬 Technical Details"):
        display_technical_details(results)
    
    # Tamper detection results
    if results['tamper_score'] is not None:
        st.subheader("🔍 Tamper Detection")
        tamper_score = results['tamper_score']
        
        if tamper_score['is_tampered']:
            st.error(f"⚠️ Image appears to be tampered (Score: {tamper_score['score']:.2f})")
        else:
            st.success(f"✅ Image appears authentic (Score: {tamper_score['score']:.2f})")

def create_interactive_map(best_location, confidence_map):
    """Create an interactive map showing the location estimate."""
    
    # Create base map
    m = folium.Map(
        location=[best_location['latitude'], best_location['longitude']],
        zoom_start=10,
        tiles='OpenStreetMap'
    )
    
    # Add confidence heatmap if available
    if confidence_map and 'heatmap_data' in confidence_map:
        # Add heatmap layer
        folium.plugins.HeatMap(
            confidence_map['heatmap_data'],
            name='Confidence Heatmap',
            show=True,
            overlay=True,
            control=True
        ).add_to(m)
    
    # Add best location marker
    folium.Marker(
        [best_location['latitude'], best_location['longitude']],
        popup=f"Estimated Location<br>Confidence: {best_location.get('confidence', 0):.1%}",
        tooltip="SkyPin Estimate",
        icon=folium.Icon(color='red', icon='star')
    ).add_to(m)
    
    # Add uncertainty ellipse if available
    if 'uncertainty_ellipse' in best_location:
        ellipse = best_location['uncertainty_ellipse']
        folium.Circle(
            [best_location['latitude'], best_location['longitude']],
            radius=ellipse['semi_major'] * 1000,  # Convert km to meters
            popup=f"Uncertainty: {ellipse['semi_major']:.1f} km",
            color='blue',
            fill=False,
            weight=2
        ).add_to(m)
    
    # Display map
    st_folium(m, width=700, height=500)

def display_technical_details(results):
    """Display technical analysis details."""
    
    # EXIF data
    st.subheader("📷 EXIF Data")
    exif_data = results['exif_data']
    if exif_data:
        col1, col2 = st.columns(2)
        with col1:
            st.text(f"Timestamp: {exif_data.get('timestamp', 'N/A')}")
            st.text(f"Camera: {exif_data.get('camera', 'N/A')}")
        with col2:
            st.text(f"Focal Length: {exif_data.get('focal_length', 'N/A')}")
            st.text(f"Orientation: {exif_data.get('orientation', 'N/A')}")
    else:
        st.warning("No EXIF data found")
    
    # Sun detection
    st.subheader("☀️ Sun Detection")
    sun_pos = results['sun_position']
    if sun_pos['detected']:
        st.text(f"Azimuth: {sun_pos['azimuth']:.1f}°")
        st.text(f"Elevation: {sun_pos['elevation']:.1f}°")
        st.text(f"Confidence: {sun_pos['confidence']:.2f}")
    else:
        st.warning("Sun not detected")
    
    # Shadow analysis
    st.subheader("🌑 Shadow Analysis")
    shadow_data = results['shadow_data']
    if shadow_data['detected']:
        st.text(f"Shadow Azimuth: {shadow_data['azimuth']:.1f}°")
        st.text(f"Shadow Length: {shadow_data['length']:.1f} pixels")
        st.text(f"Quality: {shadow_data['quality']:.2f}")
    else:
        st.warning("Shadows not detected")

if __name__ == "__main__":
    main()