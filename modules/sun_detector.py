"""
Sun detection module for SkyPin using YOLOv8
Detects sun position and lens flare in images
"""

import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional
from ultralytics import YOLO
import logging
from pathlib import Path
import requests
import os

logger = logging.getLogger(__name__)

class SunDetector:
    """Sun detection using YOLOv8 model."""
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize sun detector.
        
        Args:
            model_path: Path to YOLOv8 model file. If None, will download default model.
        """
        self.model = None
        self.model_path = model_path or self._get_default_model_path()
        self._load_model()
    
    def _get_default_model_path(self) -> str:
        """Get path to default sun detection model."""
        models_dir = Path(__file__).parent.parent / "models"
        models_dir.mkdir(exist_ok=True)
        return str(models_dir / "sun_detector.pt")
    
    def _load_model(self):
        """Load YOLOv8 model."""
        try:
            if os.path.exists(self.model_path):
                self.model = YOLO(self.model_path)
                logger.info(f"Loaded sun detection model from {self.model_path}")
            else:
                # Download default model or use pre-trained YOLOv8
                logger.info("Downloading sun detection model...")
                self.model = YOLO('yolov8n.pt')  # Use nano model for speed
                # In a real implementation, you would train a custom model for sun detection
                logger.warning("Using generic YOLOv8 model - custom sun detection model not found")
                
        except Exception as e:
            logger.error(f"Error loading sun detection model: {e}")
            # Fallback to traditional computer vision methods
            self.model = None
    
    def detect_sun(self, image: np.ndarray) -> Dict:
        """
        Detect sun position in image.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Dictionary containing sun detection results
        """
        if self.model is not None:
            return self._detect_with_yolo(image)
        else:
            return self._detect_with_cv(image)
    
    def _detect_with_yolo(self, image: np.ndarray) -> Dict:
        """Detect sun using YOLOv8 model."""
        try:
            # Run inference
            results = self.model(image, conf=0.3, verbose=False)
            
            # Process results
            detections = []
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        # Get bounding box coordinates
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        confidence = box.conf[0].cpu().numpy()
                        class_id = int(box.cls[0].cpu().numpy())
                        
                        detections.append({
                            'bbox': [int(x1), int(y1), int(x2), int(y2)],
                            'confidence': float(confidence),
                            'class_id': class_id
                        })
            
            # Find best sun detection
            if detections:
                best_detection = max(detections, key=lambda x: x['confidence'])
                return self._process_detection(best_detection, image.shape)
            else:
                return self._detect_with_cv(image)
                
        except Exception as e:
            logger.error(f"YOLO detection failed: {e}")
            return self._detect_with_cv(image)
    
    def _detect_with_cv(self, image: np.ndarray) -> Dict:
        """Fallback sun detection using traditional computer vision."""
        try:
            # Convert to different color spaces
            hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
            
            # Detect bright regions (potential sun)
            bright_regions = self._detect_bright_regions(image, hsv, lab)
            
            # Detect lens flare
            flare_regions = self._detect_lens_flare(image)
            
            # Combine detections
            if bright_regions['detected'] or flare_regions['detected']:
                # Use the detection with higher confidence
                if bright_regions['confidence'] > flare_regions['confidence']:
                    return bright_regions
                else:
                    return flare_regions
            
            return {
                'detected': False,
                'confidence': 0.0,
                'azimuth': None,
                'elevation': None,
                'center': None,
                'bbox': None,
                'method': 'cv_fallback'
            }
            
        except Exception as e:
            logger.error(f"CV detection failed: {e}")
            return {
                'detected': False,
                'confidence': 0.0,
                'azimuth': None,
                'elevation': None,
                'center': None,
                'bbox': None,
                'method': 'error'
            }
    
    def _detect_bright_regions(self, image: np.ndarray, hsv: np.ndarray, lab: np.ndarray) -> Dict:
        """Detect bright regions that could be the sun."""
        try:
            height, width = image.shape[:2]
            
            # Create mask for bright regions
            # Look for high brightness and low saturation (sun is bright white/yellow)
            brightness_mask = cv2.inRange(hsv, (0, 0, 200), (180, 30, 255))
            
            # Also look in LAB color space for bright regions
            l_channel = lab[:, :, 0]
            bright_mask_lab = cv2.threshold(l_channel, 200, 255, cv2.THRESH_BINARY)[1]
            
            # Combine masks
            combined_mask = cv2.bitwise_or(brightness_mask, bright_mask_lab)
            
            # Find contours
            contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                # Find largest contour
                largest_contour = max(contours, key=cv2.contourArea)
                area = cv2.contourArea(largest_contour)
                
                # Filter by area (sun should be reasonably sized)
                min_area = (width * height) * 0.001  # 0.1% of image
                max_area = (width * height) * 0.1    # 10% of image
                
                if min_area < area < max_area:
                    # Get bounding box
                    x, y, w, h = cv2.boundingRect(largest_contour)
                    
                    # Calculate center
                    center_x = x + w // 2
                    center_y = y + h // 2
                    
                    # Calculate azimuth and elevation
                    azimuth, elevation = self._pixel_to_angles(center_x, center_y, width, height)
                    
                    # Calculate confidence based on area and position
                    confidence = min(area / max_area, 1.0)
                    
                    return {
                        'detected': True,
                        'confidence': confidence,
                        'azimuth': azimuth,
                        'elevation': elevation,
                        'center': (center_x, center_y),
                        'bbox': [x, y, x + w, y + h],
                        'method': 'bright_regions'
                    }
            
            return {
                'detected': False,
                'confidence': 0.0,
                'azimuth': None,
                'elevation': None,
                'center': None,
                'bbox': None,
                'method': 'bright_regions'
            }
            
        except Exception as e:
            logger.error(f"Bright region detection failed: {e}")
            return {'detected': False, 'confidence': 0.0}
    
    def _detect_lens_flare(self, image: np.ndarray) -> Dict:
        """Detect lens flare patterns."""
        try:
            height, width = image.shape[:2]
            
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            
            # Apply Gaussian blur to reduce noise
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            
            # Detect bright spots using morphological operations
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            bright_spots = cv2.morphologyEx(blurred, cv2.MORPH_TOPHAT, kernel)
            
            # Threshold to get bright regions
            _, thresh = cv2.threshold(bright_spots, 50, 255, cv2.THRESH_BINARY)
            
            # Find contours
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                # Find the brightest contour
                brightest_contour = None
                max_brightness = 0
                
                for contour in contours:
                    # Get average brightness in contour area
                    mask = np.zeros(gray.shape, dtype=np.uint8)
                    cv2.fillPoly(mask, [contour], 255)
                    brightness = np.mean(gray[mask > 0])
                    
                    if brightness > max_brightness:
                        max_brightness = brightness
                        brightest_contour = contour
                
                if brightest_contour is not None:
                    # Get center of brightest contour
                    M = cv2.moments(brightest_contour)
                    if M["m00"] != 0:
                        center_x = int(M["m10"] / M["m00"])
                        center_y = int(M["m01"] / M["m00"])
                        
                        # Calculate azimuth and elevation
                        azimuth, elevation = self._pixel_to_angles(center_x, center_y, width, height)
                        
                        # Calculate confidence based on brightness
                        confidence = min(max_brightness / 255.0, 1.0)
                        
                        return {
                            'detected': True,
                            'confidence': confidence,
                            'azimuth': azimuth,
                            'elevation': elevation,
                            'center': (center_x, center_y),
                            'bbox': cv2.boundingRect(brightest_contour),
                            'method': 'lens_flare'
                        }
            
            return {
                'detected': False,
                'confidence': 0.0,
                'azimuth': None,
                'elevation': None,
                'center': None,
                'bbox': None,
                'method': 'lens_flare'
            }
            
        except Exception as e:
            logger.error(f"Lens flare detection failed: {e}")
            return {'detected': False, 'confidence': 0.0}
    
    def _pixel_to_angles(self, x: int, y: int, width: int, height: int) -> Tuple[float, float]:
        """
        Convert pixel coordinates to azimuth and elevation angles.
        
        Args:
            x, y: Pixel coordinates
            width, height: Image dimensions
            
        Returns:
            Tuple of (azimuth, elevation) in degrees
        """
        # Normalize coordinates to [-1, 1]
        x_norm = (2 * x / width) - 1
        y_norm = (2 * y / height) - 1
        
        # Calculate azimuth (horizontal angle)
        azimuth = np.arctan2(x_norm, 1) * 180 / np.pi
        
        # Calculate elevation (vertical angle)
        # Assuming image represents a field of view of ~60 degrees
        fov = 60  # degrees
        elevation = np.arcsin(y_norm * np.sin(fov * np.pi / 180)) * 180 / np.pi
        
        return azimuth, elevation
    
    def _process_detection(self, detection: Dict, image_shape: Tuple) -> Dict:
        """Process YOLO detection result."""
        try:
            bbox = detection['bbox']
            confidence = detection['confidence']
            
            # Calculate center
            center_x = (bbox[0] + bbox[2]) // 2
            center_y = (bbox[1] + bbox[3]) // 2
            
            # Calculate azimuth and elevation
            height, width = image_shape[:2]
            azimuth, elevation = self._pixel_to_angles(center_x, center_y, width, height)
            
            return {
                'detected': True,
                'confidence': confidence,
                'azimuth': azimuth,
                'elevation': elevation,
                'center': (center_x, center_y),
                'bbox': bbox,
                'method': 'yolo'
            }
            
        except Exception as e:
            logger.error(f"Error processing detection: {e}")
            return {'detected': False, 'confidence': 0.0}

# Global detector instance
_detector = None

def detect_sun_position(image: np.ndarray) -> Dict:
    """
    Detect sun position in image.
    
    Args:
        image: Input image as numpy array
        
    Returns:
        Dictionary containing sun detection results
    """
    global _detector
    
    if _detector is None:
        _detector = SunDetector()
    
    return _detector.detect_sun(image)

def download_sun_model(url: str, save_path: str) -> bool:
    """
    Download sun detection model from URL.
    
    Args:
        url: URL to download model from
        save_path: Path to save model file
        
    Returns:
        True if download successful, False otherwise
    """
    try:
        logger.info(f"Downloading sun model from {url}")
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        with open(save_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        logger.info(f"Model downloaded to {save_path}")
        return True
        
    except Exception as e:
        logger.error(f"Error downloading model: {e}")
        return False