"""
Cloud detection module for SkyPin
Detects clouds and analyzes weather conditions for better location accuracy
"""

import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from scipy import ndimage
from skimage import feature, measure, segmentation
import requests
import json
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

class CloudDetector:
    """Detects clouds and analyzes weather conditions."""
    
    def __init__(self):
        """Initialize cloud detector."""
        self.cloud_threshold = 0.3
        self.sky_threshold = 0.7
        self.weather_api_key = None
        
    def detect_clouds(self, image: np.ndarray) -> Dict:
        """
        Detect clouds in image.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Dictionary containing cloud detection results
        """
        try:
            # Convert to different color spaces
            hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
            
            # Detect sky region
            sky_mask = self._detect_sky_region(image, hsv, lab)
            
            # Detect clouds in sky region
            cloud_mask = self._detect_clouds_in_sky(image, sky_mask)
            
            # Analyze cloud properties
            cloud_properties = self._analyze_cloud_properties(cloud_mask, sky_mask)
            
            # Estimate weather conditions
            weather_conditions = self._estimate_weather_conditions(cloud_properties)
            
            return {
                'clouds_detected': cloud_properties['cloud_coverage'] > self.cloud_threshold,
                'cloud_coverage': cloud_properties['cloud_coverage'],
                'sky_coverage': cloud_properties['sky_coverage'],
                'cloud_types': cloud_properties['cloud_types'],
                'weather_conditions': weather_conditions,
                'confidence': cloud_properties['confidence'],
                'sky_mask': sky_mask,
                'cloud_mask': cloud_mask,
                'method': 'multi_spectral'
            }
            
        except Exception as e:
            logger.error(f"Cloud detection failed: {e}")
            return {
                'clouds_detected': False,
                'cloud_coverage': 0.0,
                'sky_coverage': 0.0,
                'cloud_types': [],
                'weather_conditions': 'unknown',
                'confidence': 0.0,
                'sky_mask': None,
                'cloud_mask': None,
                'method': 'error'
            }
    
    def _detect_sky_region(self, image: np.ndarray, hsv: np.ndarray, lab: np.ndarray) -> np.ndarray:
        """Detect sky region in image."""
        try:
            height, width = image.shape[:2]
            
            # Create sky mask based on color and position
            sky_mask = np.zeros((height, width), dtype=np.uint8)
            
            # Sky is typically blue and in upper portion of image
            # Blue hue range in HSV
            blue_mask = cv2.inRange(hsv, (100, 50, 50), (130, 255, 255))
            
            # Light colors (sky is typically bright)
            light_mask = cv2.inRange(lab[:, :, 0], 150, 255)
            
            # Combine masks
            combined_mask = cv2.bitwise_and(blue_mask, light_mask)
            
            # Focus on upper portion of image (sky is usually at top)
            upper_region = np.zeros_like(combined_mask)
            upper_region[:height//2, :] = combined_mask[:height//2, :]
            
            # Apply morphological operations to clean up
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            sky_mask = cv2.morphologyEx(upper_region, cv2.MORPH_CLOSE, kernel)
            sky_mask = cv2.morphologyEx(sky_mask, cv2.MORPH_OPEN, kernel)
            
            return sky_mask
            
        except Exception as e:
            logger.error(f"Sky region detection failed: {e}")
            return np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
    
    def _detect_clouds_in_sky(self, image: np.ndarray, sky_mask: np.ndarray) -> np.ndarray:
        """Detect clouds within sky region."""
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            
            # Apply sky mask
            masked_gray = cv2.bitwise_and(gray, gray, mask=sky_mask)
            
            # Detect clouds using texture analysis
            cloud_mask = self._detect_clouds_by_texture(masked_gray, sky_mask)
            
            # Detect clouds using brightness
            brightness_mask = self._detect_clouds_by_brightness(masked_gray, sky_mask)
            
            # Combine detection methods
            combined_mask = cv2.bitwise_or(cloud_mask, brightness_mask)
            
            # Apply sky mask to ensure clouds are only in sky region
            final_mask = cv2.bitwise_and(combined_mask, sky_mask)
            
            return final_mask
            
        except Exception as e:
            logger.error(f"Cloud detection in sky failed: {e}")
            return np.zeros_like(sky_mask)
    
    def _detect_clouds_by_texture(self, gray: np.ndarray, sky_mask: np.ndarray) -> np.ndarray:
        """Detect clouds using texture analysis."""
        try:
            # Calculate local binary patterns
            lbp = feature.local_binary_pattern(gray, 8, 1, method='uniform')
            
            # Calculate texture variance
            texture_variance = cv2.Laplacian(gray, cv2.CV_64F)
            texture_variance = np.abs(texture_variance)
            
            # Clouds have different texture than clear sky
            cloud_threshold = np.mean(texture_variance[sky_mask > 0]) * 1.5
            
            cloud_mask = (texture_variance > cloud_threshold).astype(np.uint8) * 255
            
            return cloud_mask
            
        except Exception as e:
            logger.error(f"Texture-based cloud detection failed: {e}")
            return np.zeros_like(sky_mask)
    
    def _detect_clouds_by_brightness(self, gray: np.ndarray, sky_mask: np.ndarray) -> np.ndarray:
        """Detect clouds using brightness analysis."""
        try:
            # Calculate local brightness
            kernel = np.ones((15, 15), np.float32) / 225
            local_brightness = cv2.filter2D(gray.astype(np.float32), -1, kernel)
            
            # Clouds are typically brighter than clear sky
            sky_brightness = np.mean(local_brightness[sky_mask > 0])
            cloud_threshold = sky_brightness * 1.2
            
            cloud_mask = (local_brightness > cloud_threshold).astype(np.uint8) * 255
            
            return cloud_mask
            
        except Exception as e:
            logger.error(f"Brightness-based cloud detection failed: {e}")
            return np.zeros_like(sky_mask)
    
    def _analyze_cloud_properties(self, cloud_mask: np.ndarray, sky_mask: np.ndarray) -> Dict:
        """Analyze properties of detected clouds."""
        try:
            # Calculate coverage
            sky_pixels = np.sum(sky_mask > 0)
            cloud_pixels = np.sum(cloud_mask > 0)
            
            if sky_pixels > 0:
                cloud_coverage = cloud_pixels / sky_pixels
                sky_coverage = 1.0 - cloud_coverage
            else:
                cloud_coverage = 0.0
                sky_coverage = 0.0
            
            # Analyze cloud types
            cloud_types = self._classify_cloud_types(cloud_mask, sky_mask)
            
            # Calculate confidence
            confidence = self._calculate_cloud_confidence(cloud_mask, sky_mask)
            
            return {
                'cloud_coverage': cloud_coverage,
                'sky_coverage': sky_coverage,
                'cloud_types': cloud_types,
                'confidence': confidence,
                'cloud_pixels': cloud_pixels,
                'sky_pixels': sky_pixels
            }
            
        except Exception as e:
            logger.error(f"Cloud property analysis failed: {e}")
            return {
                'cloud_coverage': 0.0,
                'sky_coverage': 0.0,
                'cloud_types': [],
                'confidence': 0.0,
                'cloud_pixels': 0,
                'sky_pixels': 0
            }
    
    def _classify_cloud_types(self, cloud_mask: np.ndarray, sky_mask: np.ndarray) -> List[str]:
        """Classify types of clouds."""
        try:
            cloud_types = []
            
            # Find contours of clouds
            contours, _ = cv2.findContours(cloud_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                area = cv2.contourArea(contour)
                
                if area > 100:  # Minimum cloud size
                    # Analyze cloud shape and size
                    cloud_type = self._classify_single_cloud(contour, area)
                    if cloud_type:
                        cloud_types.append(cloud_type)
            
            return list(set(cloud_types))  # Remove duplicates
            
        except Exception as e:
            logger.error(f"Cloud type classification failed: {e}")
            return []
    
    def _classify_single_cloud(self, contour: np.ndarray, area: float) -> Optional[str]:
        """Classify a single cloud."""
        try:
            # Calculate aspect ratio
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / h if h > 0 else 0
            
            # Calculate circularity
            perimeter = cv2.arcLength(contour, True)
            circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
            
            # Classify based on properties
            if circularity > 0.7:
                return "cumulus"  # Round, puffy clouds
            elif aspect_ratio > 3:
                return "stratus"  # Layered clouds
            elif area > 10000:
                return "cumulonimbus"  # Large storm clouds
            else:
                return "cirrus"  # Wispy high clouds
                
        except Exception as e:
            logger.error(f"Single cloud classification failed: {e}")
            return None
    
    def _calculate_cloud_confidence(self, cloud_mask: np.ndarray, sky_mask: np.ndarray) -> float:
        """Calculate confidence in cloud detection."""
        try:
            # Factors affecting confidence
            sky_coverage = np.sum(sky_mask > 0) / (sky_mask.shape[0] * sky_mask.shape[1])
            cloud_clarity = np.sum(cloud_mask > 0) / np.sum(sky_mask > 0) if np.sum(sky_mask > 0) > 0 else 0
            
            # Higher confidence with more sky visible and clear cloud boundaries
            confidence = min(1.0, sky_coverage * 2 + cloud_clarity)
            
            return confidence
            
        except Exception as e:
            logger.error(f"Cloud confidence calculation failed: {e}")
            return 0.0
    
    def _estimate_weather_conditions(self, cloud_properties: Dict) -> str:
        """Estimate weather conditions from cloud properties."""
        try:
            cloud_coverage = cloud_properties['cloud_coverage']
            cloud_types = cloud_properties['cloud_types']
            
            if cloud_coverage < 0.1:
                return "clear"
            elif cloud_coverage < 0.3:
                return "partly_cloudy"
            elif cloud_coverage < 0.7:
                return "mostly_cloudy"
            elif "cumulonimbus" in cloud_types:
                return "stormy"
            else:
                return "overcast"
                
        except Exception as e:
            logger.error(f"Weather condition estimation failed: {e}")
            return "unknown"
    
    def get_weather_data(self, latitude: float, longitude: float, timestamp: datetime) -> Dict:
        """Get weather data from external API."""
        try:
            if not self.weather_api_key:
                return {'error': 'Weather API key not configured'}
            
            # This would integrate with a weather API like OpenWeatherMap
            # For now, return placeholder data
            return {
                'temperature': 20.0,
                'humidity': 60.0,
                'pressure': 1013.25,
                'visibility': 10.0,
                'cloud_coverage': 0.3,
                'weather_description': 'partly cloudy'
            }
            
        except Exception as e:
            logger.error(f"Weather data retrieval failed: {e}")
            return {'error': str(e)}
    
    def enhance_image_for_analysis(self, image: np.ndarray) -> np.ndarray:
        """Enhance image for better cloud detection."""
        try:
            # Apply CLAHE to improve contrast
            lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            lab[:, :, 0] = clahe.apply(lab[:, :, 0])
            enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
            
            return enhanced
            
        except Exception as e:
            logger.error(f"Image enhancement failed: {e}")
            return image

# Global detector instance
_detector = None

def detect_clouds(image: np.ndarray) -> Dict:
    """
    Detect clouds in image.
    
    Args:
        image: Input image as numpy array
        
    Returns:
        Dictionary containing cloud detection results
    """
    global _detector
    
    if _detector is None:
        _detector = CloudDetector()
    
    return _detector.detect_clouds(image)

def get_weather_data(latitude: float, longitude: float, timestamp: datetime) -> Dict:
    """
    Get weather data for location and time.
    
    Args:
        latitude: Latitude in degrees
        longitude: Longitude in degrees
        timestamp: Timestamp
        
    Returns:
        Dictionary containing weather data
    """
    global _detector
    
    if _detector is None:
        _detector = CloudDetector()
    
    return _detector.get_weather_data(latitude, longitude, timestamp)

def enhance_image_for_analysis(image: np.ndarray) -> np.ndarray:
    """
    Enhance image for better cloud detection.
    
    Args:
        image: Input image as numpy array
        
    Returns:
        Enhanced image
    """
    global _detector
    
    if _detector is None:
        _detector = CloudDetector()
    
    return _detector.enhance_image_for_analysis(image)