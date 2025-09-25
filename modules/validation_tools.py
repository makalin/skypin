"""
Validation tools module for SkyPin
Provides quality assessment and validation tools
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import logging
from datetime import datetime, timezone
import math
from scipy import stats
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

logger = logging.getLogger(__name__)

class ValidationTools:
    """Provides validation and quality assessment tools."""
    
    def __init__(self):
        """Initialize validation tools."""
        self.quality_thresholds = {
            'sun_detection': 0.5,
            'shadow_analysis': 0.4,
            'location_confidence': 0.3,
            'tamper_detection': 0.6
        }
        
    def validate_analysis_results(self, results: Dict) -> Dict:
        """
        Validate analysis results for quality and consistency.
        
        Args:
            results: Analysis results dictionary
            
        Returns:
            Dictionary containing validation results
        """
        try:
            validation_results = {
                'overall_quality': 0.0,
                'quality_score': 0.0,
                'consistency_score': 0.0,
                'completeness_score': 0.0,
                'warnings': [],
                'errors': [],
                'recommendations': [],
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
            
            # Check completeness
            completeness_score = self._check_completeness(results)
            validation_results['completeness_score'] = completeness_score
            
            # Check quality
            quality_score = self._check_quality(results)
            validation_results['quality_score'] = quality_score
            
            # Check consistency
            consistency_score = self._check_consistency(results)
            validation_results['consistency_score'] = consistency_score
            
            # Calculate overall quality
            overall_quality = (completeness_score + quality_score + consistency_score) / 3
            validation_results['overall_quality'] = overall_quality
            
            # Generate warnings and recommendations
            self._generate_warnings_and_recommendations(results, validation_results)
            
            return validation_results
            
        except Exception as e:
            logger.error(f"Validation failed: {e}")
            return {
                'overall_quality': 0.0,
                'quality_score': 0.0,
                'consistency_score': 0.0,
                'completeness_score': 0.0,
                'warnings': [f"Validation error: {e}"],
                'errors': [str(e)],
                'recommendations': [],
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
    
    def _check_completeness(self, results: Dict) -> float:
        """Check completeness of analysis results."""
        try:
            required_fields = [
                'exif_data',
                'sun_detection',
                'shadow_analysis',
                'cloud_detection',
                'moon_detection',
                'star_analysis',
                'tamper_detection',
                'location_result'
            ]
            
            present_fields = 0
            for field in required_fields:
                if field in results and results[field] is not None:
                    present_fields += 1
            
            completeness_score = present_fields / len(required_fields)
            return completeness_score
            
        except Exception as e:
            logger.error(f"Completeness check failed: {e}")
            return 0.0
    
    def _check_quality(self, results: Dict) -> float:
        """Check quality of analysis results."""
        try:
            quality_scores = []
            
            # Check sun detection quality
            sun_data = results.get('sun_detection', {})
            if sun_data.get('detected'):
                sun_confidence = sun_data.get('confidence', 0)
                quality_scores.append(sun_confidence)
            else:
                quality_scores.append(0.0)
            
            # Check shadow analysis quality
            shadow_data = results.get('shadow_analysis', {})
            if shadow_data.get('detected'):
                shadow_quality = shadow_data.get('quality', 0)
                quality_scores.append(shadow_quality)
            else:
                quality_scores.append(0.0)
            
            # Check location result quality
            location_data = results.get('location_result', {})
            best_location = location_data.get('best_location', {})
            if best_location:
                location_confidence = best_location.get('confidence', 0)
                quality_scores.append(location_confidence)
            else:
                quality_scores.append(0.0)
            
            # Check tamper detection quality
            tamper_data = results.get('tamper_detection', {})
            if tamper_data:
                tamper_confidence = tamper_data.get('confidence', 0)
                quality_scores.append(tamper_confidence)
            else:
                quality_scores.append(0.0)
            
            # Calculate average quality score
            if quality_scores:
                quality_score = np.mean(quality_scores)
            else:
                quality_score = 0.0
            
            return quality_score
            
        except Exception as e:
            logger.error(f"Quality check failed: {e}")
            return 0.0
    
    def _check_consistency(self, results: Dict) -> float:
        """Check consistency of analysis results."""
        try:
            consistency_scores = []
            
            # Check sun vs shadow azimuth consistency
            sun_data = results.get('sun_detection', {})
            shadow_data = results.get('shadow_analysis', {})
            
            if (sun_data.get('detected') and shadow_data.get('detected') and
                sun_data.get('azimuth') is not None and shadow_data.get('azimuth') is not None):
                
                sun_azimuth = sun_data['azimuth']
                shadow_azimuth = shadow_data['azimuth']
                
                # Shadows should point away from sun (180° difference)
                expected_shadow_azimuth = (sun_azimuth + 180) % 360
                azimuth_diff = abs(shadow_azimuth - expected_shadow_azimuth)
                azimuth_diff = min(azimuth_diff, 360 - azimuth_diff)  # Handle wraparound
                
                # Consistency score based on azimuth difference
                consistency_score = max(0, 1.0 - azimuth_diff / 90.0)  # 90° = 0 score
                consistency_scores.append(consistency_score)
            
            # Check location vs EXIF GPS consistency
            location_data = results.get('location_result', {})
            exif_data = results.get('exif_data', {})
            
            if (location_data.get('best_location') and 
                exif_data.get('gps_latitude') is not None and 
                exif_data.get('gps_longitude') is not None):
                
                calc_lat = location_data['best_location'].get('latitude')
                calc_lon = location_data['best_location'].get('longitude')
                exif_lat = exif_data['gps_latitude']
                exif_lon = exif_data['gps_longitude']
                
                if calc_lat is not None and calc_lon is not None:
                    # Calculate distance between calculated and EXIF locations
                    distance = self._calculate_distance(calc_lat, calc_lon, exif_lat, exif_lon)
                    
                    # Consistency score based on distance (closer = higher score)
                    consistency_score = max(0, 1.0 - distance / 1000.0)  # 1000km = 0 score
                    consistency_scores.append(consistency_score)
            
            # Calculate average consistency score
            if consistency_scores:
                consistency_score = np.mean(consistency_scores)
            else:
                consistency_score = 0.5  # Neutral score if no consistency checks possible
            
            return consistency_score
            
        except Exception as e:
            logger.error(f"Consistency check failed: {e}")
            return 0.0
    
    def _generate_warnings_and_recommendations(self, results: Dict, validation_results: Dict):
        """Generate warnings and recommendations."""
        try:
            warnings = []
            recommendations = []
            
            # Check for low confidence results
            sun_data = results.get('sun_detection', {})
            if sun_data.get('detected') and sun_data.get('confidence', 0) < 0.3:
                warnings.append("Low confidence sun detection")
                recommendations.append("Try enhancing image contrast for better sun detection")
            
            shadow_data = results.get('shadow_analysis', {})
            if shadow_data.get('detected') and shadow_data.get('quality', 0) < 0.3:
                warnings.append("Low quality shadow analysis")
                recommendations.append("Ensure shadows are sharp and well-defined")
            
            location_data = results.get('location_result', {})
            best_location = location_data.get('best_location', {})
            if best_location and best_location.get('confidence', 0) < 0.3:
                warnings.append("Low confidence location result")
                recommendations.append("Try using images with clearer sky or shadows")
            
            # Check for missing data
            if not sun_data.get('detected') and not shadow_data.get('detected'):
                warnings.append("No sun or shadows detected")
                recommendations.append("Use images with visible sky or sharp shadows")
            
            if not location_data.get('best_location'):
                warnings.append("No location determined")
                recommendations.append("Ensure image has sufficient celestial information")
            
            # Check for tampering
            tamper_data = results.get('tamper_detection', {})
            if tamper_data.get('is_tampered'):
                warnings.append("Image appears to be tampered")
                recommendations.append("Verify image authenticity before analysis")
            
            validation_results['warnings'] = warnings
            validation_results['recommendations'] = recommendations
            
        except Exception as e:
            logger.error(f"Warning generation failed: {e}")
    
    def _calculate_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate distance between two points using Haversine formula."""
        try:
            # Convert to radians
            lat1_rad = math.radians(lat1)
            lon1_rad = math.radians(lon1)
            lat2_rad = math.radians(lat2)
            lon2_rad = math.radians(lon2)
            
            # Haversine formula
            dlat = lat2_rad - lat1_rad
            dlon = lon2_rad - lon1_rad
            
            a = (math.sin(dlat/2)**2 + 
                 math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon/2)**2)
            c = 2 * math.asin(math.sqrt(a))
            
            # Earth's radius in kilometers
            earth_radius = 6371.0
            
            return earth_radius * c
            
        except Exception as e:
            logger.error(f"Distance calculation failed: {e}")
            return 0.0
    
    def validate_image_quality(self, image: np.ndarray) -> Dict:
        """
        Validate image quality for analysis.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Dictionary containing image quality assessment
        """
        try:
            quality_assessment = {
                'overall_quality': 0.0,
                'brightness_score': 0.0,
                'contrast_score': 0.0,
                'sharpness_score': 0.0,
                'noise_score': 0.0,
                'resolution_score': 0.0,
                'recommendations': [],
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
            
            # Check brightness
            brightness_score = self._assess_brightness(image)
            quality_assessment['brightness_score'] = brightness_score
            
            # Check contrast
            contrast_score = self._assess_contrast(image)
            quality_assessment['contrast_score'] = contrast_score
            
            # Check sharpness
            sharpness_score = self._assess_sharpness(image)
            quality_assessment['sharpness_score'] = sharpness_score
            
            # Check noise
            noise_score = self._assess_noise(image)
            quality_assessment['noise_score'] = noise_score
            
            # Check resolution
            resolution_score = self._assess_resolution(image)
            quality_assessment['resolution_score'] = resolution_score
            
            # Calculate overall quality
            overall_quality = np.mean([
                brightness_score, contrast_score, sharpness_score, 
                noise_score, resolution_score
            ])
            quality_assessment['overall_quality'] = overall_quality
            
            # Generate recommendations
            self._generate_image_recommendations(quality_assessment)
            
            return quality_assessment
            
        except Exception as e:
            logger.error(f"Image quality validation failed: {e}")
            return {
                'overall_quality': 0.0,
                'brightness_score': 0.0,
                'contrast_score': 0.0,
                'sharpness_score': 0.0,
                'noise_score': 0.0,
                'resolution_score': 0.0,
                'recommendations': [f"Validation error: {e}"],
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
    
    def _assess_brightness(self, image: np.ndarray) -> float:
        """Assess image brightness."""
        try:
            # Convert to grayscale
            if len(image.shape) == 3:
                gray = np.mean(image, axis=2)
            else:
                gray = image
            
            # Calculate mean brightness
            mean_brightness = np.mean(gray)
            
            # Optimal brightness is around 128 (middle of 0-255 range)
            brightness_score = 1.0 - abs(mean_brightness - 128) / 128.0
            
            return max(0.0, brightness_score)
            
        except Exception as e:
            logger.error(f"Brightness assessment failed: {e}")
            return 0.0
    
    def _assess_contrast(self, image: np.ndarray) -> float:
        """Assess image contrast."""
        try:
            # Convert to grayscale
            if len(image.shape) == 3:
                gray = np.mean(image, axis=2)
            else:
                gray = image
            
            # Calculate standard deviation (contrast measure)
            contrast = np.std(gray)
            
            # Normalize contrast score (higher std = higher contrast)
            contrast_score = min(1.0, contrast / 64.0)  # 64 is a reasonable threshold
            
            return contrast_score
            
        except Exception as e:
            logger.error(f"Contrast assessment failed: {e}")
            return 0.0
    
    def _assess_sharpness(self, image: np.ndarray) -> float:
        """Assess image sharpness."""
        try:
            # Convert to grayscale
            if len(image.shape) == 3:
                gray = np.mean(image, axis=2)
            else:
                gray = image
            
            # Calculate Laplacian variance (sharpness measure)
            laplacian = np.var(np.gradient(gray))
            
            # Normalize sharpness score
            sharpness_score = min(1.0, laplacian / 1000.0)  # 1000 is a reasonable threshold
            
            return sharpness_score
            
        except Exception as e:
            logger.error(f"Sharpness assessment failed: {e}")
            return 0.0
    
    def _assess_noise(self, image: np.ndarray) -> float:
        """Assess image noise level."""
        try:
            # Convert to grayscale
            if len(image.shape) == 3:
                gray = np.mean(image, axis=2)
            else:
                gray = image
            
            # Calculate noise using local standard deviation
            # Low noise = high score
            local_std = np.std(gray)
            noise_score = max(0.0, 1.0 - local_std / 50.0)  # 50 is a reasonable threshold
            
            return noise_score
            
        except Exception as e:
            logger.error(f"Noise assessment failed: {e}")
            return 0.0
    
    def _assess_resolution(self, image: np.ndarray) -> float:
        """Assess image resolution."""
        try:
            height, width = image.shape[:2]
            
            # Check if resolution is adequate
            min_resolution = 200  # Minimum resolution for analysis
            max_resolution = 2000  # Maximum reasonable resolution
            
            if height < min_resolution or width < min_resolution:
                resolution_score = 0.0
            elif height > max_resolution or width > max_resolution:
                resolution_score = 1.0  # High resolution is good
            else:
                # Linear interpolation between min and max
                resolution_score = min(1.0, (height * width) / (max_resolution * max_resolution))
            
            return resolution_score
            
        except Exception as e:
            logger.error(f"Resolution assessment failed: {e}")
            return 0.0
    
    def _generate_image_recommendations(self, quality_assessment: Dict):
        """Generate image quality recommendations."""
        try:
            recommendations = []
            
            if quality_assessment['brightness_score'] < 0.5:
                recommendations.append("Image is too dark or too bright - adjust exposure")
            
            if quality_assessment['contrast_score'] < 0.5:
                recommendations.append("Low contrast - enhance contrast for better analysis")
            
            if quality_assessment['sharpness_score'] < 0.5:
                recommendations.append("Image is blurry - use sharper image for better results")
            
            if quality_assessment['noise_score'] < 0.5:
                recommendations.append("High noise level - use cleaner image if possible")
            
            if quality_assessment['resolution_score'] < 0.5:
                recommendations.append("Low resolution - use higher resolution image")
            
            quality_assessment['recommendations'] = recommendations
            
        except Exception as e:
            logger.error(f"Recommendation generation failed: {e}")
    
    def compare_with_ground_truth(self, results: Dict, ground_truth: Dict) -> Dict:
        """
        Compare analysis results with ground truth.
        
        Args:
            results: Analysis results
            ground_truth: Ground truth data
            
        Returns:
            Dictionary containing comparison metrics
        """
        try:
            comparison = {
                'accuracy': 0.0,
                'precision': 0.0,
                'recall': 0.0,
                'f1_score': 0.0,
                'location_error': None,
                'azimuth_error': None,
                'elevation_error': None,
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
            
            # Compare location results
            if (results.get('location_result', {}).get('best_location') and
                ground_truth.get('latitude') is not None and
                ground_truth.get('longitude') is not None):
                
                calc_lat = results['location_result']['best_location']['latitude']
                calc_lon = results['location_result']['best_location']['longitude']
                true_lat = ground_truth['latitude']
                true_lon = ground_truth['longitude']
                
                # Calculate location error
                location_error = self._calculate_distance(calc_lat, calc_lon, true_lat, true_lon)
                comparison['location_error'] = location_error
            
            # Compare sun position
            if (results.get('sun_detection', {}).get('azimuth') is not None and
                ground_truth.get('sun_azimuth') is not None):
                
                calc_azimuth = results['sun_detection']['azimuth']
                true_azimuth = ground_truth['sun_azimuth']
                
                # Calculate azimuth error
                azimuth_error = abs(calc_azimuth - true_azimuth)
                azimuth_error = min(azimuth_error, 360 - azimuth_error)  # Handle wraparound
                comparison['azimuth_error'] = azimuth_error
            
            # Compare sun elevation
            if (results.get('sun_detection', {}).get('elevation') is not None and
                ground_truth.get('sun_elevation') is not None):
                
                calc_elevation = results['sun_detection']['elevation']
                true_elevation = ground_truth['sun_elevation']
                
                # Calculate elevation error
                elevation_error = abs(calc_elevation - true_elevation)
                comparison['elevation_error'] = elevation_error
            
            return comparison
            
        except Exception as e:
            logger.error(f"Ground truth comparison failed: {e}")
            return {
                'accuracy': 0.0,
                'precision': 0.0,
                'recall': 0.0,
                'f1_score': 0.0,
                'location_error': None,
                'azimuth_error': None,
                'elevation_error': None,
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
    
    def get_quality_metrics(self, results: Dict) -> Dict:
        """Get comprehensive quality metrics."""
        try:
            metrics = {
                'detection_success_rate': 0.0,
                'average_confidence': 0.0,
                'location_accuracy': 0.0,
                'tamper_detection_rate': 0.0,
                'processing_efficiency': 0.0,
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
            
            # Calculate detection success rate
            detections = [
                results.get('sun_detection', {}).get('detected', False),
                results.get('shadow_analysis', {}).get('detected', False),
                results.get('cloud_detection', {}).get('clouds_detected', False),
                results.get('moon_detection', {}).get('detected', False)
            ]
            metrics['detection_success_rate'] = np.mean(detections)
            
            # Calculate average confidence
            confidences = []
            if results.get('sun_detection', {}).get('confidence'):
                confidences.append(results['sun_detection']['confidence'])
            if results.get('shadow_analysis', {}).get('quality'):
                confidences.append(results['shadow_analysis']['quality'])
            if results.get('location_result', {}).get('best_location', {}).get('confidence'):
                confidences.append(results['location_result']['best_location']['confidence'])
            
            if confidences:
                metrics['average_confidence'] = np.mean(confidences)
            
            # Calculate location accuracy
            if results.get('location_result', {}).get('best_location'):
                metrics['location_accuracy'] = results['location_result']['best_location'].get('confidence', 0)
            
            # Calculate tamper detection rate
            if results.get('tamper_detection', {}).get('is_tampered'):
                metrics['tamper_detection_rate'] = 1.0
            else:
                metrics['tamper_detection_rate'] = 0.0
            
            # Calculate processing efficiency (placeholder)
            metrics['processing_efficiency'] = 1.0  # Would be calculated based on processing time
            
            return metrics
            
        except Exception as e:
            logger.error(f"Quality metrics calculation failed: {e}")
            return {
                'detection_success_rate': 0.0,
                'average_confidence': 0.0,
                'location_accuracy': 0.0,
                'tamper_detection_rate': 0.0,
                'processing_efficiency': 0.0,
                'timestamp': datetime.now(timezone.utc).isoformat()
            }

# Global validation tools instance
_validation_tools = None

def validate_analysis_results(results: Dict) -> Dict:
    """
    Validate analysis results for quality and consistency.
    
    Args:
        results: Analysis results dictionary
        
    Returns:
        Dictionary containing validation results
    """
    global _validation_tools
    
    if _validation_tools is None:
        _validation_tools = ValidationTools()
    
    return _validation_tools.validate_analysis_results(results)

def validate_image_quality(image: np.ndarray) -> Dict:
    """
    Validate image quality for analysis.
    
    Args:
        image: Input image as numpy array
        
    Returns:
        Dictionary containing image quality assessment
    """
    global _validation_tools
    
    if _validation_tools is None:
        _validation_tools = ValidationTools()
    
    return _validation_tools.validate_image_quality(image)

def compare_with_ground_truth(results: Dict, ground_truth: Dict) -> Dict:
    """
    Compare analysis results with ground truth.
    
    Args:
        results: Analysis results
        ground_truth: Ground truth data
        
    Returns:
        Dictionary containing comparison metrics
    """
    global _validation_tools
    
    if _validation_tools is None:
        _validation_tools = ValidationTools()
    
    return _validation_tools.compare_with_ground_truth(results, ground_truth)

def get_quality_metrics(results: Dict) -> Dict:
    """
    Get comprehensive quality metrics.
    
    Args:
        results: Analysis results dictionary
        
    Returns:
        Dictionary containing quality metrics
    """
    global _validation_tools
    
    if _validation_tools is None:
        _validation_tools = ValidationTools()
    
    return _validation_tools.get_quality_metrics(results)