"""
Image enhancement module for SkyPin
Provides various image preprocessing and enhancement tools
"""

import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import logging
from scipy import ndimage
from skimage import exposure, filters, restoration, segmentation
from PIL import Image, ImageEnhance, ImageFilter
import math

logger = logging.getLogger(__name__)

class ImageEnhancer:
    """Provides various image enhancement and preprocessing tools."""
    
    def __init__(self):
        """Initialize image enhancer."""
        self.default_clip_limit = 2.0
        self.default_tile_size = (8, 8)
        
    def enhance_for_analysis(self, image: np.ndarray, enhancement_type: str = 'general') -> np.ndarray:
        """
        Enhance image for analysis.
        
        Args:
            image: Input image as numpy array
            enhancement_type: Type of enhancement ('general', 'sun', 'shadow', 'moon', 'stars')
            
        Returns:
            Enhanced image
        """
        try:
            if enhancement_type == 'sun':
                return self._enhance_for_sun_detection(image)
            elif enhancement_type == 'shadow':
                return self._enhance_for_shadow_analysis(image)
            elif enhancement_type == 'moon':
                return self._enhance_for_moon_detection(image)
            elif enhancement_type == 'stars':
                return self._enhance_for_star_detection(image)
            else:
                return self._enhance_general(image)
                
        except Exception as e:
            logger.error(f"Image enhancement failed: {e}")
            return image
    
    def _enhance_general(self, image: np.ndarray) -> np.ndarray:
        """General image enhancement."""
        try:
            # Apply CLAHE for contrast enhancement
            enhanced = self.apply_clahe(image)
            
            # Apply noise reduction
            enhanced = self.reduce_noise(enhanced)
            
            # Apply sharpening
            enhanced = self.sharpen_image(enhanced)
            
            return enhanced
            
        except Exception as e:
            logger.error(f"General enhancement failed: {e}")
            return image
    
    def _enhance_for_sun_detection(self, image: np.ndarray) -> np.ndarray:
        """Enhance image specifically for sun detection."""
        try:
            # Convert to LAB color space
            lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
            
            # Enhance L channel (brightness)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            lab[:, :, 0] = clahe.apply(lab[:, :, 0])
            
            # Convert back to RGB
            enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
            
            # Apply gamma correction to brighten image
            enhanced = self.apply_gamma_correction(enhanced, gamma=1.2)
            
            return enhanced
            
        except Exception as e:
            logger.error(f"Sun detection enhancement failed: {e}")
            return image
    
    def _enhance_for_shadow_analysis(self, image: np.ndarray) -> np.ndarray:
        """Enhance image specifically for shadow analysis."""
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            
            # Apply CLAHE to enhance contrast
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced_gray = clahe.apply(gray)
            
            # Apply edge enhancement
            enhanced_gray = self.enhance_edges(enhanced_gray)
            
            # Convert back to RGB
            enhanced = cv2.cvtColor(enhanced_gray, cv2.COLOR_GRAY2RGB)
            
            return enhanced
            
        except Exception as e:
            logger.error(f"Shadow analysis enhancement failed: {e}")
            return image
    
    def _enhance_for_moon_detection(self, image: np.ndarray) -> np.ndarray:
        """Enhance image specifically for moon detection."""
        try:
            # Apply histogram equalization
            enhanced = self.apply_histogram_equalization(image)
            
            # Apply noise reduction
            enhanced = self.reduce_noise(enhanced)
            
            # Apply gentle sharpening
            enhanced = self.sharpen_image(enhanced, strength=0.5)
            
            return enhanced
            
        except Exception as e:
            logger.error(f"Moon detection enhancement failed: {e}")
            return image
    
    def _enhance_for_star_detection(self, image: np.ndarray) -> np.ndarray:
        """Enhance image specifically for star detection."""
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            
            # Apply histogram equalization
            enhanced_gray = cv2.equalizeHist(gray)
            
            # Apply bilateral filter for noise reduction while preserving edges
            enhanced_gray = cv2.bilateralFilter(enhanced_gray, 9, 75, 75)
            
            # Apply morphological operations to enhance stars
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            enhanced_gray = cv2.morphologyEx(enhanced_gray, cv2.MORPH_TOPHAT, kernel)
            
            # Convert back to RGB
            enhanced = cv2.cvtColor(enhanced_gray, cv2.COLOR_GRAY2RGB)
            
            return enhanced
            
        except Exception as e:
            logger.error(f"Star detection enhancement failed: {e}")
            return image
    
    def apply_clahe(self, image: np.ndarray, clip_limit: float = None, 
                   tile_size: Tuple[int, int] = None) -> np.ndarray:
        """Apply Contrast Limited Adaptive Histogram Equalization."""
        try:
            if clip_limit is None:
                clip_limit = self.default_clip_limit
            if tile_size is None:
                tile_size = self.default_tile_size
            
            # Convert to LAB color space
            lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
            
            # Apply CLAHE to L channel
            clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_size)
            lab[:, :, 0] = clahe.apply(lab[:, :, 0])
            
            # Convert back to RGB
            enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
            
            return enhanced
            
        except Exception as e:
            logger.error(f"CLAHE application failed: {e}")
            return image
    
    def apply_histogram_equalization(self, image: np.ndarray) -> np.ndarray:
        """Apply histogram equalization."""
        try:
            # Convert to YUV color space
            yuv = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
            
            # Apply histogram equalization to Y channel
            yuv[:, :, 0] = cv2.equalizeHist(yuv[:, :, 0])
            
            # Convert back to RGB
            enhanced = cv2.cvtColor(yuv, cv2.COLOR_YUV2RGB)
            
            return enhanced
            
        except Exception as e:
            logger.error(f"Histogram equalization failed: {e}")
            return image
    
    def apply_gamma_correction(self, image: np.ndarray, gamma: float = 1.0) -> np.ndarray:
        """Apply gamma correction."""
        try:
            # Create lookup table
            lookup_table = np.array([((i / 255.0) ** gamma) * 255 for i in range(256)]).astype(np.uint8)
            
            # Apply gamma correction
            enhanced = cv2.LUT(image, lookup_table)
            
            return enhanced
            
        except Exception as e:
            logger.error(f"Gamma correction failed: {e}")
            return image
    
    def reduce_noise(self, image: np.ndarray, method: str = 'bilateral') -> np.ndarray:
        """Reduce noise in image."""
        try:
            if method == 'bilateral':
                # Bilateral filter preserves edges while reducing noise
                enhanced = cv2.bilateralFilter(image, 9, 75, 75)
            elif method == 'gaussian':
                # Gaussian blur
                enhanced = cv2.GaussianBlur(image, (5, 5), 0)
            elif method == 'median':
                # Median filter
                enhanced = cv2.medianBlur(image, 5)
            else:
                enhanced = image
            
            return enhanced
            
        except Exception as e:
            logger.error(f"Noise reduction failed: {e}")
            return image
    
    def sharpen_image(self, image: np.ndarray, strength: float = 1.0) -> np.ndarray:
        """Sharpen image."""
        try:
            # Create sharpening kernel
            kernel = np.array([[-1, -1, -1],
                             [-1,  9, -1],
                             [-1, -1, -1]]) * strength
            
            # Apply convolution
            sharpened = cv2.filter2D(image, -1, kernel)
            
            # Clip values to valid range
            sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)
            
            return sharpened
            
        except Exception as e:
            logger.error(f"Image sharpening failed: {e}")
            return image
    
    def enhance_edges(self, image: np.ndarray) -> np.ndarray:
        """Enhance edges in image."""
        try:
            # Convert to grayscale if needed
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            else:
                gray = image.copy()
            
            # Apply Laplacian filter
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            
            # Convert back to uint8
            laplacian = np.uint8(np.absolute(laplacian))
            
            # Add to original image
            enhanced = cv2.add(gray, laplacian)
            
            # Convert back to RGB if original was RGB
            if len(image.shape) == 3:
                enhanced = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)
            
            return enhanced
            
        except Exception as e:
            logger.error(f"Edge enhancement failed: {e}")
            return image
    
    def correct_exposure(self, image: np.ndarray, target_exposure: float = 0.5) -> np.ndarray:
        """Correct image exposure."""
        try:
            # Convert to float
            img_float = image.astype(np.float32) / 255.0
            
            # Calculate current exposure
            current_exposure = np.mean(img_float)
            
            # Calculate correction factor
            correction_factor = target_exposure / current_exposure
            
            # Apply correction
            corrected = img_float * correction_factor
            
            # Clip to valid range
            corrected = np.clip(corrected, 0, 1)
            
            # Convert back to uint8
            enhanced = (corrected * 255).astype(np.uint8)
            
            return enhanced
            
        except Exception as e:
            logger.error(f"Exposure correction failed: {e}")
            return image
    
    def correct_color_balance(self, image: np.ndarray) -> np.ndarray:
        """Correct color balance."""
        try:
            # Convert to LAB color space
            lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
            
            # Apply color balance correction to A and B channels
            lab[:, :, 1] = cv2.equalizeHist(lab[:, :, 1])  # A channel
            lab[:, :, 2] = cv2.equalizeHist(lab[:, :, 2])  # B channel
            
            # Convert back to RGB
            enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
            
            return enhanced
            
        except Exception as e:
            logger.error(f"Color balance correction failed: {e}")
            return image
    
    def remove_lens_distortion(self, image: np.ndarray, 
                             camera_matrix: np.ndarray = None,
                             distortion_coeffs: np.ndarray = None) -> np.ndarray:
        """Remove lens distortion."""
        try:
            if camera_matrix is None or distortion_coeffs is None:
                # Use default camera parameters
                height, width = image.shape[:2]
                camera_matrix = np.array([[width, 0, width/2],
                                        [0, height, height/2],
                                        [0, 0, 1]], dtype=np.float32)
                distortion_coeffs = np.array([0.1, -0.1, 0, 0, 0], dtype=np.float32)
            
            # Undistort image
            undistorted = cv2.undistort(image, camera_matrix, distortion_coeffs)
            
            return undistorted
            
        except Exception as e:
            logger.error(f"Lens distortion removal failed: {e}")
            return image
    
    def enhance_contrast(self, image: np.ndarray, alpha: float = 1.5, beta: int = 0) -> np.ndarray:
        """Enhance contrast using linear transformation."""
        try:
            # Apply contrast and brightness adjustment
            enhanced = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
            
            return enhanced
            
        except Exception as e:
            logger.error(f"Contrast enhancement failed: {e}")
            return image
    
    def enhance_saturation(self, image: np.ndarray, factor: float = 1.5) -> np.ndarray:
        """Enhance color saturation."""
        try:
            # Convert to HSV
            hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            
            # Enhance saturation
            hsv[:, :, 1] = hsv[:, :, 1] * factor
            
            # Clip to valid range
            hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)
            
            # Convert back to RGB
            enhanced = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
            
            return enhanced
            
        except Exception as e:
            logger.error(f"Saturation enhancement failed: {e}")
            return image
    
    def apply_unsharp_mask(self, image: np.ndarray, radius: float = 1.0, 
                          amount: float = 1.0, threshold: int = 0) -> np.ndarray:
        """Apply unsharp mask filter."""
        try:
            # Convert to float
            img_float = image.astype(np.float32)
            
            # Apply Gaussian blur
            blurred = cv2.GaussianBlur(img_float, (0, 0), radius)
            
            # Calculate unsharp mask
            mask = img_float - blurred
            
            # Apply mask
            enhanced = img_float + amount * mask
            
            # Clip to valid range
            enhanced = np.clip(enhanced, 0, 255).astype(np.uint8)
            
            return enhanced
            
        except Exception as e:
            logger.error(f"Unsharp mask application failed: {e}")
            return image
    
    def enhance_for_astronomical_analysis(self, image: np.ndarray) -> np.ndarray:
        """Enhance image specifically for astronomical analysis."""
        try:
            # Apply CLAHE for contrast
            enhanced = self.apply_clahe(image, clip_limit=3.0)
            
            # Apply noise reduction
            enhanced = self.reduce_noise(enhanced, method='bilateral')
            
            # Apply gentle sharpening
            enhanced = self.sharpen_image(enhanced, strength=0.5)
            
            # Apply gamma correction
            enhanced = self.apply_gamma_correction(enhanced, gamma=1.1)
            
            return enhanced
            
        except Exception as e:
            logger.error(f"Astronomical enhancement failed: {e}")
            return image
    
    def batch_enhance(self, images: List[np.ndarray], 
                     enhancement_type: str = 'general') -> List[np.ndarray]:
        """Enhance multiple images in batch."""
        try:
            enhanced_images = []
            
            for image in images:
                enhanced = self.enhance_for_analysis(image, enhancement_type)
                enhanced_images.append(enhanced)
            
            return enhanced_images
            
        except Exception as e:
            logger.error(f"Batch enhancement failed: {e}")
            return images
    
    def get_enhancement_metrics(self, original: np.ndarray, enhanced: np.ndarray) -> Dict:
        """Get metrics comparing original and enhanced images."""
        try:
            # Calculate various metrics
            original_gray = cv2.cvtColor(original, cv2.COLOR_RGB2GRAY)
            enhanced_gray = cv2.cvtColor(enhanced, cv2.COLOR_RGB2GRAY)
            
            # Contrast metrics
            original_contrast = np.std(original_gray)
            enhanced_contrast = np.std(enhanced_gray)
            
            # Brightness metrics
            original_brightness = np.mean(original_gray)
            enhanced_brightness = np.mean(enhanced_gray)
            
            # Edge strength
            original_edges = cv2.Laplacian(original_gray, cv2.CV_64F)
            enhanced_edges = cv2.Laplacian(enhanced_gray, cv2.CV_64F)
            
            original_edge_strength = np.var(original_edges)
            enhanced_edge_strength = np.var(enhanced_edges)
            
            return {
                'contrast_improvement': enhanced_contrast / original_contrast,
                'brightness_change': enhanced_brightness - original_brightness,
                'edge_improvement': enhanced_edge_strength / original_edge_strength,
                'original_contrast': original_contrast,
                'enhanced_contrast': enhanced_contrast,
                'original_brightness': original_brightness,
                'enhanced_brightness': enhanced_brightness
            }
            
        except Exception as e:
            logger.error(f"Enhancement metrics calculation failed: {e}")
            return {}

# Global enhancer instance
_enhancer = None

def enhance_for_analysis(image: np.ndarray, enhancement_type: str = 'general') -> np.ndarray:
    """
    Enhance image for analysis.
    
    Args:
        image: Input image as numpy array
        enhancement_type: Type of enhancement
        
    Returns:
        Enhanced image
    """
    global _enhancer
    
    if _enhancer is None:
        _enhancer = ImageEnhancer()
    
    return _enhancer.enhance_for_analysis(image, enhancement_type)

def apply_clahe(image: np.ndarray, clip_limit: float = None, 
               tile_size: Tuple[int, int] = None) -> np.ndarray:
    """
    Apply CLAHE to image.
    
    Args:
        image: Input image
        clip_limit: CLAHE clip limit
        tile_size: CLAHE tile size
        
    Returns:
        Enhanced image
    """
    global _enhancer
    
    if _enhancer is None:
        _enhancer = ImageEnhancer()
    
    return _enhancer.apply_clahe(image, clip_limit, tile_size)

def enhance_for_astronomical_analysis(image: np.ndarray) -> np.ndarray:
    """
    Enhance image for astronomical analysis.
    
    Args:
        image: Input image
        
    Returns:
        Enhanced image
    """
    global _enhancer
    
    if _enhancer is None:
        _enhancer = ImageEnhancer()
    
    return _enhancer.enhance_for_astronomical_analysis(image)