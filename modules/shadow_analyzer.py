"""
Shadow analysis module for SkyPin
Analyzes shadows to determine sun azimuth using computer vision techniques
"""

import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from scipy import ndimage
from skimage import feature, measure
import math

logger = logging.getLogger(__name__)

class ShadowAnalyzer:
    """Shadow analysis using computer vision techniques."""
    
    def __init__(self):
        """Initialize shadow analyzer."""
        self.min_shadow_length = 50  # Minimum shadow length in pixels
        self.max_shadow_length = 1000  # Maximum shadow length in pixels
        self.edge_threshold_low = 50
        self.edge_threshold_high = 150
    
    def analyze_shadows(self, image: np.ndarray) -> Dict:
        """
        Analyze shadows in image to determine sun azimuth.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Dictionary containing shadow analysis results
        """
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            
            # Detect edges
            edges = self._detect_edges(gray)
            
            # Find shadow lines
            shadow_lines = self._find_shadow_lines(edges)
            
            # Analyze shadow vectors
            shadow_vectors = self._analyze_shadow_vectors(shadow_lines, image.shape)
            
            # Calculate sun azimuth
            sun_azimuth = self._calculate_sun_azimuth(shadow_vectors)
            
            # Calculate confidence
            confidence = self._calculate_confidence(shadow_vectors, image.shape)
            
            return {
                'detected': len(shadow_vectors) > 0,
                'azimuth': sun_azimuth,
                'length': np.mean([v['length'] for v in shadow_vectors]) if shadow_vectors else 0,
                'quality': confidence,
                'vectors': shadow_vectors,
                'method': 'canny_hough'
            }
            
        except Exception as e:
            logger.error(f"Shadow analysis failed: {e}")
            return {
                'detected': False,
                'azimuth': None,
                'length': 0,
                'quality': 0.0,
                'vectors': [],
                'method': 'error'
            }
    
    def _detect_edges(self, gray: np.ndarray) -> np.ndarray:
        """Detect edges using Canny edge detection."""
        try:
            # Apply Gaussian blur to reduce noise
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            
            # Apply Canny edge detection
            edges = cv2.Canny(blurred, self.edge_threshold_low, self.edge_threshold_high)
            
            # Apply morphological operations to clean up edges
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
            
            return edges
            
        except Exception as e:
            logger.error(f"Edge detection failed: {e}")
            return np.zeros_like(gray)
    
    def _find_shadow_lines(self, edges: np.ndarray) -> List[Tuple]:
        """Find shadow lines using Hough line transform."""
        try:
            # Apply Hough line transform
            lines = cv2.HoughLinesP(
                edges,
                rho=1,
                theta=np.pi/180,
                threshold=50,
                minLineLength=self.min_shadow_length,
                maxLineGap=10
            )
            
            if lines is None:
                return []
            
            # Filter and process lines
            shadow_lines = []
            for line in lines:
                x1, y1, x2, y2 = line[0]
                
                # Calculate line properties
                length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
                angle = np.arctan2(y2-y1, x2-x1) * 180 / np.pi
                
                # Filter by length and angle (shadows are typically vertical-ish)
                if (self.min_shadow_length < length < self.max_shadow_length and
                    abs(angle) < 45):  # Within 45 degrees of vertical
                    
                    shadow_lines.append((x1, y1, x2, y2, length, angle))
            
            return shadow_lines
            
        except Exception as e:
            logger.error(f"Line detection failed: {e}")
            return []
    
    def _analyze_shadow_vectors(self, shadow_lines: List[Tuple], image_shape: Tuple) -> List[Dict]:
        """Analyze shadow vectors to determine direction and quality."""
        try:
            vectors = []
            height, width = image_shape[:2]
            
            for line in shadow_lines:
                x1, y1, x2, y2, length, angle = line
                
                # Calculate shadow vector (direction from object to shadow tip)
                # Shadow points away from the sun
                shadow_vector = np.array([x2-x1, y2-y1])
                
                # Normalize vector
                if np.linalg.norm(shadow_vector) > 0:
                    shadow_vector = shadow_vector / np.linalg.norm(shadow_vector)
                
                # Calculate sun direction (opposite to shadow)
                sun_vector = -shadow_vector
                
                # Convert to azimuth
                azimuth = np.arctan2(sun_vector[0], sun_vector[1]) * 180 / np.pi
                
                # Calculate quality metrics
                quality = self._calculate_vector_quality(line, image_shape)
                
                vectors.append({
                    'start': (x1, y1),
                    'end': (x2, y2),
                    'length': length,
                    'angle': angle,
                    'azimuth': azimuth,
                    'quality': quality,
                    'vector': shadow_vector
                })
            
            return vectors
            
        except Exception as e:
            logger.error(f"Vector analysis failed: {e}")
            return []
    
    def _calculate_vector_quality(self, line: Tuple, image_shape: Tuple) -> float:
        """Calculate quality score for a shadow vector."""
        try:
            x1, y1, x2, y2, length, angle = line
            height, width = image_shape[:2]
            
            # Quality factors
            length_score = min(length / self.max_shadow_length, 1.0)
            
            # Prefer lines closer to vertical (shadows are typically vertical)
            verticality_score = 1.0 - abs(angle) / 90.0
            
            # Prefer lines in the center of the image
            center_x = width / 2
            center_y = height / 2
            line_center_x = (x1 + x2) / 2
            line_center_y = (y1 + y2) / 2
            center_distance = np.sqrt((line_center_x - center_x)**2 + (line_center_y - center_y)**2)
            max_distance = np.sqrt(center_x**2 + center_y**2)
            centrality_score = 1.0 - (center_distance / max_distance)
            
            # Combine scores
            quality = (length_score * 0.4 + verticality_score * 0.4 + centrality_score * 0.2)
            
            return min(quality, 1.0)
            
        except Exception as e:
            logger.error(f"Quality calculation failed: {e}")
            return 0.0
    
    def _calculate_sun_azimuth(self, shadow_vectors: List[Dict]) -> Optional[float]:
        """Calculate sun azimuth from shadow vectors."""
        try:
            if not shadow_vectors:
                return None
            
            # Weight vectors by quality
            weighted_azimuths = []
            total_weight = 0
            
            for vector in shadow_vectors:
                weight = vector['quality']
                azimuth = vector['azimuth']
                
                weighted_azimuths.append(azimuth * weight)
                total_weight += weight
            
            if total_weight > 0:
                # Calculate weighted average azimuth
                avg_azimuth = sum(weighted_azimuths) / total_weight
                
                # Normalize to [0, 360)
                avg_azimuth = avg_azimuth % 360
                if avg_azimuth < 0:
                    avg_azimuth += 360
                
                return avg_azimuth
            
            return None
            
        except Exception as e:
            logger.error(f"Azimuth calculation failed: {e}")
            return None
    
    def _calculate_confidence(self, shadow_vectors: List[Dict], image_shape: Tuple) -> float:
        """Calculate overall confidence in shadow analysis."""
        try:
            if not shadow_vectors:
                return 0.0
            
            # Factors affecting confidence
            num_vectors = len(shadow_vectors)
            vector_consistency = self._calculate_vector_consistency(shadow_vectors)
            avg_quality = np.mean([v['quality'] for v in shadow_vectors])
            
            # Combine factors
            confidence = (
                min(num_vectors / 5, 1.0) * 0.3 +  # More vectors = higher confidence
                vector_consistency * 0.4 +          # Consistency = higher confidence
                avg_quality * 0.3                   # Quality = higher confidence
            )
            
            return min(confidence, 1.0)
            
        except Exception as e:
            logger.error(f"Confidence calculation failed: {e}")
            return 0.0
    
    def _calculate_vector_consistency(self, shadow_vectors: List[Dict]) -> float:
        """Calculate consistency between shadow vectors."""
        try:
            if len(shadow_vectors) < 2:
                return 1.0
            
            azimuths = [v['azimuth'] for v in shadow_vectors]
            
            # Calculate standard deviation of azimuths
            azimuth_std = np.std(azimuths)
            
            # Convert to consistency score (lower std = higher consistency)
            consistency = max(0, 1.0 - azimuth_std / 90.0)
            
            return consistency
            
        except Exception as e:
            logger.error(f"Consistency calculation failed: {e}")
            return 0.0
    
    def detect_shadow_objects(self, image: np.ndarray) -> List[Dict]:
        """Detect objects that cast shadows."""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            
            # Apply edge detection
            edges = self._detect_edges(gray)
            
            # Find contours
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            objects = []
            for contour in contours:
                area = cv2.contourArea(contour)
                
                # Filter by area
                if area > 100:  # Minimum object area
                    # Get bounding box
                    x, y, w, h = cv2.boundingRect(contour)
                    
                    # Calculate aspect ratio
                    aspect_ratio = w / h if h > 0 else 0
                    
                    # Filter by aspect ratio (objects are typically taller than wide)
                    if aspect_ratio < 1.5:  # Height > width
                        objects.append({
                            'contour': contour,
                            'bbox': (x, y, w, h),
                            'area': area,
                            'aspect_ratio': aspect_ratio
                        })
            
            return objects
            
        except Exception as e:
            logger.error(f"Object detection failed: {e}")
            return []
    
    def enhance_shadows(self, image: np.ndarray) -> np.ndarray:
        """Enhance shadow visibility in image."""
        try:
            # Convert to LAB color space
            lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
            l_channel = lab[:, :, 0]
            
            # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced_l = clahe.apply(l_channel)
            
            # Replace L channel
            lab[:, :, 0] = enhanced_l
            
            # Convert back to RGB
            enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
            
            return enhanced
            
        except Exception as e:
            logger.error(f"Shadow enhancement failed: {e}")
            return image

# Global analyzer instance
_analyzer = None

def analyze_shadows(image: np.ndarray) -> Dict:
    """
    Analyze shadows in image to determine sun azimuth.
    
    Args:
        image: Input image as numpy array
        
    Returns:
        Dictionary containing shadow analysis results
    """
    global _analyzer
    
    if _analyzer is None:
        _analyzer = ShadowAnalyzer()
    
    return _analyzer.analyze_shadows(image)

def detect_shadow_objects(image: np.ndarray) -> List[Dict]:
    """
    Detect objects that cast shadows.
    
    Args:
        image: Input image as numpy array
        
    Returns:
        List of detected shadow-casting objects
    """
    global _analyzer
    
    if _analyzer is None:
        _analyzer = ShadowAnalyzer()
    
    return _analyzer.detect_shadow_objects(image)

def enhance_shadows(image: np.ndarray) -> np.ndarray:
    """
    Enhance shadow visibility in image.
    
    Args:
        image: Input image as numpy array
        
    Returns:
        Enhanced image
    """
    global _analyzer
    
    if _analyzer is None:
        _analyzer = ShadowAnalyzer()
    
    return _analyzer.enhance_shadows(image)