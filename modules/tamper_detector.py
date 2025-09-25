"""
Tamper detection module for SkyPin
Detects image manipulation using Error-Level Analysis and JPEG ghost detection
"""

import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from PIL import Image
import io
from scipy import ndimage
from skimage import feature, measure
import math

logger = logging.getLogger(__name__)

class TamperDetector:
    """Detects image tampering using various forensic techniques."""
    
    def __init__(self):
        """Initialize tamper detector."""
        self.error_level_threshold = 0.1
        self.jpeg_ghost_threshold = 0.05
        self.noise_threshold = 0.02
        
    def detect_tampering(self, image: np.ndarray) -> Dict:
        """
        Detect image tampering using multiple techniques.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Dictionary containing tamper detection results
        """
        try:
            # Convert to grayscale if needed
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            else:
                gray = image.copy()
            
            # Perform various tamper detection techniques
            error_level_score = self._error_level_analysis(gray)
            jpeg_ghost_score = self._jpeg_ghost_detection(gray)
            noise_analysis_score = self._noise_analysis(gray)
            compression_analysis_score = self._compression_analysis(gray)
            
            # Calculate overall tamper score
            tamper_score = self._calculate_tamper_score(
                error_level_score, jpeg_ghost_score, 
                noise_analysis_score, compression_analysis_score
            )
            
            # Determine if image is tampered
            is_tampered = tamper_score > 0.5
            
            return {
                'is_tampered': is_tampered,
                'score': tamper_score,
                'error_level_score': error_level_score,
                'jpeg_ghost_score': jpeg_ghost_score,
                'noise_analysis_score': noise_analysis_score,
                'compression_analysis_score': compression_analysis_score,
                'confidence': abs(tamper_score - 0.5) * 2,  # Distance from 0.5
                'method': 'multi_technique'
            }
            
        except Exception as e:
            logger.error(f"Tamper detection failed: {e}")
            return {
                'is_tampered': False,
                'score': 0.0,
                'error_level_score': 0.0,
                'jpeg_ghost_score': 0.0,
                'noise_analysis_score': 0.0,
                'compression_analysis_score': 0.0,
                'confidence': 0.0,
                'method': 'error'
            }
    
    def _error_level_analysis(self, gray: np.ndarray) -> float:
        """Perform Error-Level Analysis (ELA) to detect tampering."""
        try:
            # Convert to float
            img_float = gray.astype(np.float32)
            
            # Apply JPEG compression simulation
            # This is a simplified version - in practice, you'd use actual JPEG compression
            compressed = self._simulate_jpeg_compression(img_float)
            
            # Calculate error level
            error_level = np.abs(img_float - compressed)
            
            # Normalize error level
            error_level_norm = error_level / 255.0
            
            # Calculate tamper score based on error level distribution
            mean_error = np.mean(error_level_norm)
            std_error = np.std(error_level_norm)
            
            # Higher mean and std indicate potential tampering
            tamper_score = min(1.0, (mean_error + std_error) / 2.0)
            
            return tamper_score
            
        except Exception as e:
            logger.error(f"Error level analysis failed: {e}")
            return 0.0
    
    def _simulate_jpeg_compression(self, img: np.ndarray) -> np.ndarray:
        """Simulate JPEG compression for ELA."""
        try:
            # Convert to PIL Image
            img_pil = Image.fromarray(img.astype(np.uint8))
            
            # Save to bytes with JPEG compression
            buffer = io.BytesIO()
            img_pil.save(buffer, format='JPEG', quality=95)
            buffer.seek(0)
            
            # Load back
            compressed_pil = Image.open(buffer)
            compressed = np.array(compressed_pil, dtype=np.float32)
            
            return compressed
            
        except Exception as e:
            logger.error(f"JPEG compression simulation failed: {e}")
            return img
    
    def _jpeg_ghost_detection(self, gray: np.ndarray) -> float:
        """Detect JPEG ghosts (double compression artifacts)."""
        try:
            # Convert to float
            img_float = gray.astype(np.float32)
            
            # Apply DCT-like transformation
            dct_result = self._apply_dct_transform(img_float)
            
            # Look for ghosting patterns
            ghost_score = self._detect_ghosting_patterns(dct_result)
            
            return ghost_score
            
        except Exception as e:
            logger.error(f"JPEG ghost detection failed: {e}")
            return 0.0
    
    def _apply_dct_transform(self, img: np.ndarray) -> np.ndarray:
        """Apply DCT-like transformation to detect compression artifacts."""
        try:
            # Simple DCT approximation using block processing
            block_size = 8
            height, width = img.shape
            
            dct_result = np.zeros_like(img)
            
            for i in range(0, height - block_size, block_size):
                for j in range(0, width - block_size, block_size):
                    block = img[i:i+block_size, j:j+block_size]
                    
                    # Simple DCT approximation
                    dct_block = self._simple_dct(block)
                    dct_result[i:i+block_size, j:j+block_size] = dct_block
            
            return dct_result
            
        except Exception as e:
            logger.error(f"DCT transform failed: {e}")
            return img
    
    def _simple_dct(self, block: np.ndarray) -> np.ndarray:
        """Simple DCT approximation."""
        try:
            # This is a simplified DCT - in practice, you'd use proper DCT
            result = np.zeros_like(block)
            
            for u in range(block.shape[0]):
                for v in range(block.shape[1]):
                    sum_val = 0
                    for x in range(block.shape[0]):
                        for y in range(block.shape[1]):
                            sum_val += block[x, y] * np.cos((2*x+1)*u*np.pi/(2*block.shape[0])) * np.cos((2*y+1)*v*np.pi/(2*block.shape[1]))
                    
                    result[u, v] = sum_val
            
            return result
            
        except Exception as e:
            logger.error(f"Simple DCT failed: {e}")
            return block
    
    def _detect_ghosting_patterns(self, dct_result: np.ndarray) -> float:
        """Detect ghosting patterns in DCT result."""
        try:
            # Look for periodic patterns that indicate double compression
            fft_result = np.fft.fft2(dct_result)
            fft_magnitude = np.abs(fft_result)
            
            # Calculate periodicity score
            periodicity_score = self._calculate_periodicity(fft_magnitude)
            
            return periodicity_score
            
        except Exception as e:
            logger.error(f"Ghosting pattern detection failed: {e}")
            return 0.0
    
    def _calculate_periodicity(self, fft_magnitude: np.ndarray) -> float:
        """Calculate periodicity score from FFT magnitude."""
        try:
            # Look for peaks in FFT that indicate periodic patterns
            height, width = fft_magnitude.shape
            
            # Focus on low-frequency components
            low_freq = fft_magnitude[:height//4, :width//4]
            
            # Calculate peak strength
            peak_strength = np.max(low_freq) / np.mean(low_freq)
            
            # Normalize to [0, 1]
            periodicity_score = min(1.0, peak_strength / 10.0)
            
            return periodicity_score
            
        except Exception as e:
            logger.error(f"Periodicity calculation failed: {e}")
            return 0.0
    
    def _noise_analysis(self, gray: np.ndarray) -> float:
        """Analyze noise patterns to detect tampering."""
        try:
            # Calculate local noise variance
            noise_variance = self._calculate_local_noise_variance(gray)
            
            # Analyze noise distribution
            noise_distribution_score = self._analyze_noise_distribution(noise_variance)
            
            return noise_distribution_score
            
        except Exception as e:
            logger.error(f"Noise analysis failed: {e}")
            return 0.0
    
    def _calculate_local_noise_variance(self, gray: np.ndarray) -> np.ndarray:
        """Calculate local noise variance."""
        try:
            # Apply Laplacian filter to detect edges
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            
            # Calculate local variance
            kernel = np.ones((5, 5), np.float32) / 25
            local_variance = cv2.filter2D(laplacian**2, -1, kernel)
            
            return local_variance
            
        except Exception as e:
            logger.error(f"Local noise variance calculation failed: {e}")
            return np.zeros_like(gray)
    
    def _analyze_noise_distribution(self, noise_variance: np.ndarray) -> float:
        """Analyze noise distribution for tampering indicators."""
        try:
            # Calculate statistics of noise variance
            mean_noise = np.mean(noise_variance)
            std_noise = np.std(noise_variance)
            
            # Look for unusual patterns
            # Tampered images often have inconsistent noise patterns
            consistency_score = 1.0 - (std_noise / (mean_noise + 1e-6))
            
            return max(0.0, consistency_score)
            
        except Exception as e:
            logger.error(f"Noise distribution analysis failed: {e}")
            return 0.0
    
    def _compression_analysis(self, gray: np.ndarray) -> float:
        """Analyze compression artifacts to detect tampering."""
        try:
            # Look for compression artifacts
            artifacts_score = self._detect_compression_artifacts(gray)
            
            return artifacts_score
            
        except Exception as e:
            logger.error(f"Compression analysis failed: {e}")
            return 0.0
    
    def _detect_compression_artifacts(self, gray: np.ndarray) -> float:
        """Detect compression artifacts."""
        try:
            # Look for block artifacts (8x8 blocks typical in JPEG)
            block_artifacts = self._detect_block_artifacts(gray)
            
            # Look for ringing artifacts
            ringing_artifacts = self._detect_ringing_artifacts(gray)
            
            # Combine scores
            combined_score = (block_artifacts + ringing_artifacts) / 2.0
            
            return combined_score
            
        except Exception as e:
            logger.error(f"Compression artifact detection failed: {e}")
            return 0.0
    
    def _detect_block_artifacts(self, gray: np.ndarray) -> float:
        """Detect block artifacts from JPEG compression."""
        try:
            # Look for 8x8 block patterns
            block_size = 8
            height, width = gray.shape
            
            block_artifacts = 0
            
            for i in range(0, height - block_size, block_size):
                for j in range(0, width - block_size, block_size):
                    block = gray[i:i+block_size, j:j+block_size]
                    
                    # Calculate block variance
                    block_var = np.var(block)
                    
                    # High variance indicates potential artifacts
                    if block_var > np.var(gray) * 2:
                        block_artifacts += 1
            
            # Normalize by number of blocks
            num_blocks = (height // block_size) * (width // block_size)
            artifact_score = block_artifacts / num_blocks if num_blocks > 0 else 0
            
            return artifact_score
            
        except Exception as e:
            logger.error(f"Block artifact detection failed: {e}")
            return 0.0
    
    def _detect_ringing_artifacts(self, gray: np.ndarray) -> float:
        """Detect ringing artifacts from compression."""
        try:
            # Apply edge detection
            edges = cv2.Canny(gray, 50, 150)
            
            # Look for ringing patterns around edges
            ringing_score = self._calculate_ringing_score(gray, edges)
            
            return ringing_score
            
        except Exception as e:
            logger.error(f"Ringing artifact detection failed: {e}")
            return 0.0
    
    def _calculate_ringing_score(self, gray: np.ndarray, edges: np.ndarray) -> float:
        """Calculate ringing score around edges."""
        try:
            # Dilate edges to create edge regions
            kernel = np.ones((3, 3), np.uint8)
            edge_regions = cv2.dilate(edges, kernel, iterations=2)
            
            # Calculate variance in edge regions
            edge_variance = np.var(gray[edge_regions > 0])
            non_edge_variance = np.var(gray[edge_regions == 0])
            
            # High variance in edge regions indicates ringing
            if non_edge_variance > 0:
                ringing_score = min(1.0, edge_variance / non_edge_variance)
            else:
                ringing_score = 0.0
            
            return ringing_score
            
        except Exception as e:
            logger.error(f"Ringing score calculation failed: {e}")
            return 0.0
    
    def _calculate_tamper_score(self, error_level_score: float, jpeg_ghost_score: float,
                               noise_analysis_score: float, compression_analysis_score: float) -> float:
        """Calculate overall tamper score from individual scores."""
        try:
            # Weight the different scores
            weights = {
                'error_level': 0.4,
                'jpeg_ghost': 0.3,
                'noise_analysis': 0.2,
                'compression_analysis': 0.1
            }
            
            tamper_score = (
                error_level_score * weights['error_level'] +
                jpeg_ghost_score * weights['jpeg_ghost'] +
                noise_analysis_score * weights['noise_analysis'] +
                compression_analysis_score * weights['compression_analysis']
            )
            
            return min(1.0, tamper_score)
            
        except Exception as e:
            logger.error(f"Tamper score calculation failed: {e}")
            return 0.0
    
    def analyze_image_metadata(self, image_bytes: bytes) -> Dict:
        """Analyze image metadata for tampering indicators."""
        try:
            # This would analyze EXIF data, compression history, etc.
            # For now, return a placeholder
            return {
                'metadata_tamper_score': 0.0,
                'compression_history': 'unknown',
                'exif_inconsistencies': False
            }
            
        except Exception as e:
            logger.error(f"Metadata analysis failed: {e}")
            return {'metadata_tamper_score': 0.0}

# Global detector instance
_detector = None

def detect_tampering(image: np.ndarray) -> Dict:
    """
    Detect image tampering using multiple techniques.
    
    Args:
        image: Input image as numpy array
        
    Returns:
        Dictionary containing tamper detection results
    """
    global _detector
    
    if _detector is None:
        _detector = TamperDetector()
    
    return _detector.detect_tampering(image)

def analyze_image_metadata(image_bytes: bytes) -> Dict:
    """
    Analyze image metadata for tampering indicators.
    
    Args:
        image_bytes: Raw image data
        
    Returns:
        Dictionary containing metadata analysis results
    """
    global _detector
    
    if _detector is None:
        _detector = TamperDetector()
    
    return _detector.analyze_image_metadata(image_bytes)