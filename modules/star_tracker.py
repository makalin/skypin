"""
Star tracker module for SkyPin
Analyzes star trails and patterns for night-time geolocation
"""

import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from scipy import ndimage, optimize
from skimage import feature, measure, filters
import math
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

class StarTracker:
    """Analyzes star trails and patterns for geolocation."""
    
    def __init__(self):
        """Initialize star tracker."""
        self.min_star_brightness = 50
        self.max_star_size = 5
        self.trail_min_length = 20
        self.trail_max_length = 500
        
    def analyze_star_trails(self, image: np.ndarray) -> Dict:
        """
        Analyze star trails in image.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Dictionary containing star trail analysis results
        """
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            
            # Detect stars
            stars = self._detect_stars(gray)
            
            # Analyze star trails
            trails = self._analyze_trails(gray, stars)
            
            # Calculate celestial pole
            celestial_pole = self._calculate_celestial_pole(trails)
            
            # Estimate exposure time
            exposure_time = self._estimate_exposure_time(trails)
            
            # Calculate location from trails
            location_estimate = self._calculate_location_from_trails(trails, celestial_pole)
            
            return {
                'stars_detected': len(stars),
                'trails_detected': len(trails),
                'celestial_pole': celestial_pole,
                'exposure_time': exposure_time,
                'location_estimate': location_estimate,
                'confidence': self._calculate_confidence(trails, celestial_pole),
                'method': 'star_trail_analysis'
            }
            
        except Exception as e:
            logger.error(f"Star trail analysis failed: {e}")
            return {
                'stars_detected': 0,
                'trails_detected': 0,
                'celestial_pole': None,
                'exposure_time': None,
                'location_estimate': None,
                'confidence': 0.0,
                'method': 'error'
            }
    
    def _detect_stars(self, gray: np.ndarray) -> List[Dict]:
        """Detect stars in image."""
        try:
            stars = []
            
            # Apply Gaussian blur to reduce noise
            blurred = cv2.GaussianBlur(gray, (3, 3), 0)
            
            # Detect bright spots
            bright_spots = self._detect_bright_spots(blurred)
            
            # Filter by size and brightness
            for spot in bright_spots:
                if (self.min_star_brightness <= spot['brightness'] <= 255 and
                    spot['size'] <= self.max_star_size):
                    stars.append(spot)
            
            return stars
            
        except Exception as e:
            logger.error(f"Star detection failed: {e}")
            return []
    
    def _detect_bright_spots(self, gray: np.ndarray) -> List[Dict]:
        """Detect bright spots that could be stars."""
        try:
            spots = []
            
            # Use morphological operations to find bright spots
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            bright_spots = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)
            
            # Threshold to get bright regions
            _, thresh = cv2.threshold(bright_spots, 30, 255, cv2.THRESH_BINARY)
            
            # Find contours
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 1:  # Minimum area
                    # Get center
                    M = cv2.moments(contour)
                    if M["m00"] != 0:
                        center_x = int(M["m10"] / M["m00"])
                        center_y = int(M["m01"] / M["m00"])
                        
                        # Calculate brightness
                        brightness = gray[center_y, center_x]
                        
                        # Calculate size (radius)
                        radius = math.sqrt(area / np.pi)
                        
                        spots.append({
                            'center': (center_x, center_y),
                            'brightness': brightness,
                            'size': radius,
                            'area': area
                        })
            
            return spots
            
        except Exception as e:
            logger.error(f"Bright spot detection failed: {e}")
            return []
    
    def _analyze_trails(self, gray: np.ndarray, stars: List[Dict]) -> List[Dict]:
        """Analyze star trails."""
        try:
            trails = []
            
            for star in stars:
                center = star['center']
                
                # Look for linear patterns around the star
                trail = self._detect_trail_pattern(gray, center)
                
                if trail and trail['length'] > self.trail_min_length:
                    trails.append(trail)
            
            return trails
            
        except Exception as e:
            logger.error(f"Trail analysis failed: {e}")
            return []
    
    def _detect_trail_pattern(self, gray: np.ndarray, center: Tuple[int, int]) -> Optional[Dict]:
        """Detect trail pattern around a star."""
        try:
            x, y = center
            
            # Extract region around star
            region_size = 50
            x1 = max(0, x - region_size)
            y1 = max(0, y - region_size)
            x2 = min(gray.shape[1], x + region_size)
            y2 = min(gray.shape[0], y + region_size)
            
            region = gray[y1:y2, x1:x2]
            
            if region.size == 0:
                return None
            
            # Apply edge detection
            edges = cv2.Canny(region, 50, 150)
            
            # Detect lines using Hough transform
            lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=20, 
                                  minLineLength=10, maxLineGap=5)
            
            if lines is not None and len(lines) > 0:
                # Find the longest line (most likely trail)
                longest_line = max(lines, key=lambda line: 
                                 math.sqrt((line[0][2] - line[0][0])**2 + 
                                         (line[0][3] - line[0][1])**2))
                
                # Calculate trail properties
                x1_line, y1_line, x2_line, y2_line = longest_line[0]
                length = math.sqrt((x2_line - x1_line)**2 + (y2_line - y1_line)**2)
                angle = math.atan2(y2_line - y1_line, x2_line - x1_line) * 180 / np.pi
                
                # Adjust coordinates to image coordinates
                trail_x1 = x1_line + x1
                trail_y1 = y1_line + y1
                trail_x2 = x2_line + x1
                trail_y2 = y2_line + y1
                
                return {
                    'start': (trail_x1, trail_y1),
                    'end': (trail_x2, trail_y2),
                    'length': length,
                    'angle': angle,
                    'center': center
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Trail pattern detection failed: {e}")
            return None
    
    def _calculate_celestial_pole(self, trails: List[Dict]) -> Optional[Dict]:
        """Calculate celestial pole from star trails."""
        try:
            if len(trails) < 3:
                return None
            
            # Star trails converge at the celestial pole
            # Use intersection of trail lines to find the pole
            
            # Convert trails to line equations
            lines = []
            for trail in trails:
                x1, y1 = trail['start']
                x2, y2 = trail['end']
                
                # Calculate line equation: ax + by + c = 0
                if x2 != x1:
                    slope = (y2 - y1) / (x2 - x1)
                    a = -slope
                    b = 1
                    c = y1 - slope * x1
                else:
                    a = 1
                    b = 0
                    c = -x1
                
                lines.append((a, b, c))
            
            # Find intersection point (celestial pole)
            pole = self._find_line_intersection(lines)
            
            if pole:
                return {
                    'center': pole,
                    'confidence': self._calculate_pole_confidence(trails, pole)
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Celestial pole calculation failed: {e}")
            return None
    
    def _find_line_intersection(self, lines: List[Tuple[float, float, float]]) -> Optional[Tuple[float, float]]:
        """Find intersection point of multiple lines."""
        try:
            if len(lines) < 2:
                return None
            
            # Use least squares to find best intersection point
            A = []
            b = []
            
            for a, b_coeff, c in lines:
                A.append([a, b_coeff])
                b.append(-c)
            
            A = np.array(A)
            b = np.array(b)
            
            # Solve using least squares
            try:
                x, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
                if len(x) == 2:
                    return (x[0], x[1])
            except np.linalg.LinAlgError:
                pass
            
            # Fallback: use first two lines
            if len(lines) >= 2:
                a1, b1, c1 = lines[0]
                a2, b2, c2 = lines[1]
                
                # Solve system of equations
                det = a1 * b2 - a2 * b1
                if abs(det) > 1e-10:
                    x = (b2 * c1 - b1 * c2) / det
                    y = (a1 * c2 - a2 * c1) / det
                    return (x, y)
            
            return None
            
        except Exception as e:
            logger.error(f"Line intersection calculation failed: {e}")
            return None
    
    def _calculate_pole_confidence(self, trails: List[Dict], pole: Tuple[float, float]) -> float:
        """Calculate confidence in celestial pole detection."""
        try:
            if not pole:
                return 0.0
            
            # Calculate how well trails converge at the pole
            total_error = 0
            for trail in trails:
                # Calculate distance from trail to pole
                trail_center = ((trail['start'][0] + trail['end'][0]) / 2,
                              (trail['start'][1] + trail['end'][1]) / 2)
                
                distance = math.sqrt((trail_center[0] - pole[0])**2 + 
                                   (trail_center[1] - pole[1])**2)
                total_error += distance
            
            # Normalize by number of trails and image size
            avg_error = total_error / len(trails) if trails else 0
            confidence = max(0.0, 1.0 - avg_error / 100.0)  # Assume 100px is max error
            
            return confidence
            
        except Exception as e:
            logger.error(f"Pole confidence calculation failed: {e}")
            return 0.0
    
    def _estimate_exposure_time(self, trails: List[Dict]) -> Optional[float]:
        """Estimate exposure time from trail length."""
        try:
            if not trails:
                return None
            
            # Calculate average trail length
            avg_length = np.mean([trail['length'] for trail in trails])
            
            # Estimate exposure time based on trail length
            # This is a rough approximation - actual calculation would need
            # camera settings and star movement rate
            exposure_time = avg_length / 10.0  # Rough conversion factor
            
            return exposure_time
            
        except Exception as e:
            logger.error(f"Exposure time estimation failed: {e}")
            return None
    
    def _calculate_location_from_trails(self, trails: List[Dict], 
                                       celestial_pole: Optional[Dict]) -> Optional[Dict]:
        """Calculate location from star trails."""
        try:
            if not celestial_pole or not trails:
                return None
            
            # This is a simplified calculation
            # In practice, you'd need more sophisticated astronomical calculations
            
            pole_center = celestial_pole['center']
            
            # Calculate approximate latitude from pole position
            # Pole at image center = high latitude
            # Pole at image edge = low latitude
            image_center_y = 200  # Assume image height of 400
            pole_y = pole_center[1]
            
            # Rough latitude estimation
            latitude = 90 - (pole_y / image_center_y) * 90
            
            # Longitude is more difficult to determine from star trails alone
            # Would need additional information like time and date
            
            return {
                'latitude': latitude,
                'longitude': None,  # Cannot determine from trails alone
                'confidence': celestial_pole['confidence']
            }
            
        except Exception as e:
            logger.error(f"Location calculation from trails failed: {e}")
            return None
    
    def _calculate_confidence(self, trails: List[Dict], celestial_pole: Optional[Dict]) -> float:
        """Calculate overall confidence in star trail analysis."""
        try:
            if not trails or not celestial_pole:
                return 0.0
            
            # Factors affecting confidence
            num_trails = len(trails)
            pole_confidence = celestial_pole['confidence']
            
            # More trails = higher confidence
            trail_score = min(1.0, num_trails / 10.0)
            
            # Combine scores
            confidence = (trail_score * 0.6 + pole_confidence * 0.4)
            
            return confidence
            
        except Exception as e:
            logger.error(f"Confidence calculation failed: {e}")
            return 0.0
    
    def detect_constellations(self, image: np.ndarray) -> List[Dict]:
        """Detect constellations in image."""
        try:
            # This would implement constellation detection
            # For now, return placeholder
            return []
            
        except Exception as e:
            logger.error(f"Constellation detection failed: {e}")
            return []
    
    def enhance_image_for_star_detection(self, image: np.ndarray) -> np.ndarray:
        """Enhance image for better star detection."""
        try:
            # Apply histogram equalization
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            enhanced_gray = cv2.equalizeHist(gray)
            
            # Apply noise reduction
            enhanced_gray = cv2.bilateralFilter(enhanced_gray, 9, 75, 75)
            
            # Convert back to RGB
            enhanced = cv2.cvtColor(enhanced_gray, cv2.COLOR_GRAY2RGB)
            
            return enhanced
            
        except Exception as e:
            logger.error(f"Image enhancement for star detection failed: {e}")
            return image

# Global tracker instance
_tracker = None

def analyze_star_trails(image: np.ndarray) -> Dict:
    """
    Analyze star trails in image.
    
    Args:
        image: Input image as numpy array
        
    Returns:
        Dictionary containing star trail analysis results
    """
    global _tracker
    
    if _tracker is None:
        _tracker = StarTracker()
    
    return _tracker.analyze_star_trails(image)

def enhance_image_for_star_detection(image: np.ndarray) -> np.ndarray:
    """
    Enhance image for better star detection.
    
    Args:
        image: Input image as numpy array
        
    Returns:
        Enhanced image
    """
    global _tracker
    
    if _tracker is None:
        _tracker = StarTracker()
    
    return _tracker.enhance_image_for_star_detection(image)

def detect_constellations(image: np.ndarray) -> List[Dict]:
    """
    Detect constellations in image.
    
    Args:
        image: Input image as numpy array
        
    Returns:
        List of detected constellations
    """
    global _tracker
    
    if _tracker is None:
        _tracker = StarTracker()
    
    return _tracker.detect_constellations(image)