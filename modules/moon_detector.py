"""
Moon detection module for SkyPin
Detects moon position for night-time geolocation
"""

import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from scipy import ndimage
from skimage import feature, measure
import math
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

class MoonDetector:
    """Detects moon position and phase for night-time geolocation."""
    
    def __init__(self):
        """Initialize moon detector."""
        self.min_moon_size = 10  # Minimum moon size in pixels
        self.max_moon_size = 200  # Maximum moon size in pixels
        self.brightness_threshold = 150  # Minimum brightness for moon detection
        
    def detect_moon(self, image: np.ndarray) -> Dict:
        """
        Detect moon position in image.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Dictionary containing moon detection results
        """
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            
            # Detect bright circular objects (potential moon)
            moon_candidates = self._detect_moon_candidates(gray)
            
            # Analyze candidates
            moon_results = self._analyze_moon_candidates(moon_candidates, gray)
            
            # Calculate moon phase if detected
            moon_phase = self._calculate_moon_phase(moon_results)
            
            # Calculate moon position angles
            position_angles = self._calculate_moon_angles(moon_results, image.shape)
            
            return {
                'detected': moon_results['detected'],
                'center': moon_results['center'],
                'radius': moon_results['radius'],
                'brightness': moon_results['brightness'],
                'phase': moon_phase,
                'azimuth': position_angles['azimuth'],
                'elevation': position_angles['elevation'],
                'confidence': moon_results['confidence'],
                'method': 'circular_detection'
            }
            
        except Exception as e:
            logger.error(f"Moon detection failed: {e}")
            return {
                'detected': False,
                'center': None,
                'radius': 0,
                'brightness': 0,
                'phase': 'unknown',
                'azimuth': None,
                'elevation': None,
                'confidence': 0.0,
                'method': 'error'
            }
    
    def _detect_moon_candidates(self, gray: np.ndarray) -> List[Dict]:
        """Detect potential moon candidates."""
        try:
            candidates = []
            
            # Apply Gaussian blur to reduce noise
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            
            # Detect bright regions
            bright_mask = cv2.threshold(blurred, self.brightness_threshold, 255, cv2.THRESH_BINARY)[1]
            
            # Apply morphological operations to clean up
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            bright_mask = cv2.morphologyEx(bright_mask, cv2.MORPH_CLOSE, kernel)
            
            # Find contours
            contours, _ = cv2.findContours(bright_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                # Calculate properties
                area = cv2.contourArea(contour)
                perimeter = cv2.arcLength(contour, True)
                
                if area > 50:  # Minimum area
                    # Calculate circularity
                    circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
                    
                    # Calculate equivalent radius
                    radius = math.sqrt(area / np.pi)
                    
                    # Filter by size and circularity
                    if (self.min_moon_size <= radius <= self.max_moon_size and 
                        circularity > 0.7):  # Moon is roughly circular
                        
                        # Get center
                        M = cv2.moments(contour)
                        if M["m00"] != 0:
                            center_x = int(M["m10"] / M["m00"])
                            center_y = int(M["m01"] / M["m00"])
                            
                            # Calculate average brightness
                            mask = np.zeros(gray.shape, dtype=np.uint8)
                            cv2.fillPoly(mask, [contour], 255)
                            brightness = np.mean(gray[mask > 0])
                            
                            candidates.append({
                                'center': (center_x, center_y),
                                'radius': radius,
                                'area': area,
                                'circularity': circularity,
                                'brightness': brightness,
                                'contour': contour
                            })
            
            return candidates
            
        except Exception as e:
            logger.error(f"Moon candidate detection failed: {e}")
            return []
    
    def _analyze_moon_candidates(self, candidates: List[Dict], gray: np.ndarray) -> Dict:
        """Analyze moon candidates to find the best match."""
        try:
            if not candidates:
                return {
                    'detected': False,
                    'center': None,
                    'radius': 0,
                    'brightness': 0,
                    'confidence': 0.0
                }
            
            # Score candidates based on multiple factors
            best_candidate = None
            best_score = 0
            
            for candidate in candidates:
                score = self._score_moon_candidate(candidate, gray)
                if score > best_score:
                    best_score = score
                    best_candidate = candidate
            
            if best_candidate and best_score > 0.5:
                return {
                    'detected': True,
                    'center': best_candidate['center'],
                    'radius': best_candidate['radius'],
                    'brightness': best_candidate['brightness'],
                    'confidence': best_score
                }
            else:
                return {
                    'detected': False,
                    'center': None,
                    'radius': 0,
                    'brightness': 0,
                    'confidence': best_score
                }
                
        except Exception as e:
            logger.error(f"Moon candidate analysis failed: {e}")
            return {
                'detected': False,
                'center': None,
                'radius': 0,
                'brightness': 0,
                'confidence': 0.0
            }
    
    def _score_moon_candidate(self, candidate: Dict, gray: np.ndarray) -> float:
        """Score a moon candidate."""
        try:
            center = candidate['center']
            radius = candidate['radius']
            brightness = candidate['brightness']
            circularity = candidate['circularity']
            
            # Brightness score (moon should be bright)
            brightness_score = min(1.0, brightness / 255.0)
            
            # Circularity score (moon should be circular)
            circularity_score = circularity
            
            # Size score (prefer medium-sized moons)
            size_score = 1.0 - abs(radius - 50) / 50.0  # Optimal around 50 pixels
            size_score = max(0.0, size_score)
            
            # Position score (moon is often in upper portion of image)
            height, width = gray.shape
            position_score = 1.0 - (center[1] / height)  # Higher score for upper positions
            
            # Uniformity score (moon should have relatively uniform brightness)
            uniformity_score = self._calculate_uniformity_score(candidate, gray)
            
            # Combine scores
            total_score = (
                brightness_score * 0.3 +
                circularity_score * 0.25 +
                size_score * 0.2 +
                position_score * 0.15 +
                uniformity_score * 0.1
            )
            
            return total_score
            
        except Exception as e:
            logger.error(f"Moon candidate scoring failed: {e}")
            return 0.0
    
    def _calculate_uniformity_score(self, candidate: Dict, gray: np.ndarray) -> float:
        """Calculate uniformity score for moon candidate."""
        try:
            center = candidate['center']
            radius = candidate['radius']
            
            # Create circular mask
            mask = np.zeros(gray.shape, dtype=np.uint8)
            cv2.circle(mask, center, int(radius), 255, -1)
            
            # Calculate brightness variance within the circle
            masked_gray = cv2.bitwise_and(gray, gray, mask=mask)
            brightness_values = masked_gray[mask > 0]
            
            if len(brightness_values) > 0:
                variance = np.var(brightness_values)
                # Lower variance = higher uniformity score
                uniformity_score = max(0.0, 1.0 - variance / 1000.0)
            else:
                uniformity_score = 0.0
            
            return uniformity_score
            
        except Exception as e:
            logger.error(f"Uniformity score calculation failed: {e}")
            return 0.0
    
    def _calculate_moon_phase(self, moon_results: Dict) -> str:
        """Calculate moon phase from detection results."""
        try:
            if not moon_results['detected']:
                return 'unknown'
            
            # This is a simplified phase calculation
            # In practice, you'd use astronomical calculations based on date/time
            brightness = moon_results['brightness']
            
            if brightness > 200:
                return 'full'
            elif brightness > 150:
                return 'gibbous'
            elif brightness > 100:
                return 'quarter'
            elif brightness > 50:
                return 'crescent'
            else:
                return 'new'
                
        except Exception as e:
            logger.error(f"Moon phase calculation failed: {e}")
            return 'unknown'
    
    def _calculate_moon_angles(self, moon_results: Dict, image_shape: Tuple) -> Dict:
        """Calculate moon azimuth and elevation angles."""
        try:
            if not moon_results['detected']:
                return {'azimuth': None, 'elevation': None}
            
            center = moon_results['center']
            height, width = image_shape[:2]
            
            # Convert pixel coordinates to angles
            # Assuming image represents a field of view of ~60 degrees
            fov = 60  # degrees
            
            # Calculate normalized coordinates
            x_norm = (2 * center[0] / width) - 1
            y_norm = (2 * center[1] / height) - 1
            
            # Calculate azimuth (horizontal angle)
            azimuth = np.arctan2(x_norm, 1) * 180 / np.pi
            
            # Calculate elevation (vertical angle)
            elevation = np.arcsin(y_norm * np.sin(fov * np.pi / 180)) * 180 / np.pi
            
            return {
                'azimuth': azimuth,
                'elevation': elevation
            }
            
        except Exception as e:
            logger.error(f"Moon angle calculation failed: {e}")
            return {'azimuth': None, 'elevation': None}
    
    def detect_moon_phase_from_shape(self, image: np.ndarray, moon_center: Tuple[int, int], 
                                   moon_radius: int) -> str:
        """Detect moon phase from the shape of the moon."""
        try:
            # Extract moon region
            x, y = moon_center
            radius = moon_radius
            
            # Create circular mask
            mask = np.zeros(image.shape[:2], dtype=np.uint8)
            cv2.circle(mask, (x, y), radius, 255, -1)
            
            # Extract moon region
            moon_region = cv2.bitwise_and(image, image, mask=mask)
            
            # Analyze brightness distribution
            gray_moon = cv2.cvtColor(moon_region, cv2.COLOR_RGB2GRAY)
            brightness_values = gray_moon[mask > 0]
            
            if len(brightness_values) == 0:
                return 'unknown'
            
            # Calculate brightness statistics
            mean_brightness = np.mean(brightness_values)
            std_brightness = np.std(brightness_values)
            
            # Analyze brightness distribution to determine phase
            if std_brightness < 20:  # Very uniform
                return 'full'
            elif std_brightness < 40:  # Moderately uniform
                return 'gibbous'
            elif std_brightness < 60:  # Some variation
                return 'quarter'
            else:  # High variation
                return 'crescent'
                
        except Exception as e:
            logger.error(f"Moon phase detection from shape failed: {e}")
            return 'unknown'
    
    def enhance_image_for_moon_detection(self, image: np.ndarray) -> np.ndarray:
        """Enhance image for better moon detection."""
        try:
            # Apply histogram equalization to improve contrast
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            enhanced_gray = cv2.equalizeHist(gray)
            
            # Convert back to RGB
            enhanced = cv2.cvtColor(enhanced_gray, cv2.COLOR_GRAY2RGB)
            
            return enhanced
            
        except Exception as e:
            logger.error(f"Image enhancement for moon detection failed: {e}")
            return image
    
    def calculate_moon_position_astronomical(self, latitude: float, longitude: float, 
                                           timestamp: datetime) -> Dict:
        """Calculate moon position using astronomical calculations."""
        try:
            # This would use astronomical libraries like PyEphem or Skyfield
            # For now, return placeholder data
            return {
                'azimuth': 180.0,
                'elevation': 45.0,
                'distance': 384400,  # km
                'phase': 'full',
                'illumination': 100.0
            }
            
        except Exception as e:
            logger.error(f"Astronomical moon position calculation failed: {e}")
            return {
                'azimuth': None,
                'elevation': None,
                'distance': None,
                'phase': 'unknown',
                'illumination': 0.0
            }

# Global detector instance
_detector = None

def detect_moon(image: np.ndarray) -> Dict:
    """
    Detect moon position in image.
    
    Args:
        image: Input image as numpy array
        
    Returns:
        Dictionary containing moon detection results
    """
    global _detector
    
    if _detector is None:
        _detector = MoonDetector()
    
    return _detector.detect_moon(image)

def enhance_image_for_moon_detection(image: np.ndarray) -> np.ndarray:
    """
    Enhance image for better moon detection.
    
    Args:
        image: Input image as numpy array
        
    Returns:
        Enhanced image
    """
    global _detector
    
    if _detector is None:
        _detector = MoonDetector()
    
    return _detector.enhance_image_for_moon_detection(image)

def calculate_moon_position_astronomical(latitude: float, longitude: float, 
                                        timestamp: datetime) -> Dict:
    """
    Calculate moon position using astronomical calculations.
    
    Args:
        latitude: Latitude in degrees
        longitude: Longitude in degrees
        timestamp: Timestamp
        
    Returns:
        Dictionary containing astronomical moon position data
    """
    global _detector
    
    if _detector is None:
        _detector = MoonDetector()
    
    return _detector.calculate_moon_position_astronomical(latitude, longitude, timestamp)