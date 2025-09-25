"""
Utility functions for SkyPin
Common helper functions and utilities
"""

import numpy as np
import cv2
from typing import Tuple, Optional, Dict, List
import logging
from datetime import datetime, timezone
import math

logger = logging.getLogger(__name__)

def validate_image(image: np.ndarray) -> bool:
    """
    Validate image for processing.
    
    Args:
        image: Input image as numpy array
        
    Returns:
        True if image is valid, False otherwise
    """
    try:
        # Check if image is not None
        if image is None:
            return False
        
        # Check if image has correct shape
        if len(image.shape) not in [2, 3]:
            return False
        
        # Check if image has reasonable dimensions
        height, width = image.shape[:2]
        if height < 100 or width < 100:
            return False
        
        # Check if image has reasonable size
        if height > 10000 or width > 10000:
            return False
        
        # Check if image has valid data type
        if image.dtype not in [np.uint8, np.uint16, np.float32, np.float64]:
            return False
        
        # Check if image has valid pixel values
        if image.dtype == np.uint8:
            if np.any(image < 0) or np.any(image > 255):
                return False
        elif image.dtype == np.uint16:
            if np.any(image < 0) or np.any(image > 65535):
                return False
        
        return True
        
    except Exception as e:
        logger.error(f"Image validation failed: {e}")
        return False

def resize_image(image: np.ndarray, max_size: Tuple[int, int] = (1024, 1024)) -> np.ndarray:
    """
    Resize image while maintaining aspect ratio.
    
    Args:
        image: Input image
        max_size: Maximum size (width, height)
        
    Returns:
        Resized image
    """
    try:
        height, width = image.shape[:2]
        max_width, max_height = max_size
        
        # Calculate scaling factor
        scale_w = max_width / width
        scale_h = max_height / height
        scale = min(scale_w, scale_h, 1.0)  # Don't upscale
        
        if scale < 1.0:
            new_width = int(width * scale)
            new_height = int(height * scale)
            
            # Resize image
            resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
            return resized
        
        return image
        
    except Exception as e:
        logger.error(f"Image resize failed: {e}")
        return image

def normalize_image(image: np.ndarray) -> np.ndarray:
    """
    Normalize image to [0, 1] range.
    
    Args:
        image: Input image
        
    Returns:
        Normalized image
    """
    try:
        if image.dtype == np.uint8:
            return image.astype(np.float32) / 255.0
        elif image.dtype == np.uint16:
            return image.astype(np.float32) / 65535.0
        else:
            # Assume already normalized or in float range
            return image.astype(np.float32)
            
    except Exception as e:
        logger.error(f"Image normalization failed: {e}")
        return image

def denormalize_image(image: np.ndarray, dtype: np.dtype = np.uint8) -> np.ndarray:
    """
    Denormalize image from [0, 1] range to specified dtype.
    
    Args:
        image: Normalized image
        dtype: Target data type
        
    Returns:
        Denormalized image
    """
    try:
        if dtype == np.uint8:
            return (image * 255).astype(np.uint8)
        elif dtype == np.uint16:
            return (image * 65535).astype(np.uint16)
        else:
            return image.astype(dtype)
            
    except Exception as e:
        logger.error(f"Image denormalization failed: {e}")
        return image

def format_coordinates(latitude: float, longitude: float, precision: int = 6) -> Tuple[str, str]:
    """
    Format coordinates for display.
    
    Args:
        latitude: Latitude in degrees
        longitude: Longitude in degrees
        precision: Number of decimal places
        
    Returns:
        Tuple of formatted (latitude, longitude) strings
    """
    try:
        lat_str = f"{latitude:.{precision}f}°"
        lon_str = f"{longitude:.{precision}f}°"
        
        # Add direction indicators
        if latitude >= 0:
            lat_str += " N"
        else:
            lat_str += " S"
        
        if longitude >= 0:
            lon_str += " E"
        else:
            lon_str += " W"
        
        return lat_str, lon_str
        
    except Exception as e:
        logger.error(f"Coordinate formatting failed: {e}")
        return f"{latitude:.6f}°", f"{longitude:.6f}°"

def calculate_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate distance between two points using Haversine formula.
    
    Args:
        lat1, lon1: First point coordinates
        lat2, lon2: Second point coordinates
        
    Returns:
        Distance in kilometers
    """
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

def calculate_bearing(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate bearing between two points.
    
    Args:
        lat1, lon1: First point coordinates
        lat2, lon2: Second point coordinates
        
    Returns:
        Bearing in degrees (0-360)
    """
    try:
        # Convert to radians
        lat1_rad = math.radians(lat1)
        lon1_rad = math.radians(lon1)
        lat2_rad = math.radians(lat2)
        lon2_rad = math.radians(lon2)
        
        dlon = lon2_rad - lon1_rad
        
        y = math.sin(dlon) * math.cos(lat2_rad)
        x = (math.cos(lat1_rad) * math.sin(lat2_rad) - 
             math.sin(lat1_rad) * math.cos(lat2_rad) * math.cos(dlon))
        
        bearing = math.atan2(y, x)
        bearing = math.degrees(bearing)
        bearing = (bearing + 360) % 360
        
        return bearing
        
    except Exception as e:
        logger.error(f"Bearing calculation failed: {e}")
        return 0.0

def angle_difference(angle1: float, angle2: float) -> float:
    """
    Calculate the difference between two angles.
    
    Args:
        angle1, angle2: Angles in degrees
        
    Returns:
        Difference in degrees (0-180)
    """
    try:
        diff = abs(angle1 - angle2)
        return min(diff, 360 - diff)
        
    except Exception as e:
        logger.error(f"Angle difference calculation failed: {e}")
        return 0.0

def normalize_angle(angle: float) -> float:
    """
    Normalize angle to [0, 360) range.
    
    Args:
        angle: Angle in degrees
        
    Returns:
        Normalized angle
    """
    try:
        while angle < 0:
            angle += 360
        while angle >= 360:
            angle -= 360
        return angle
        
    except Exception as e:
        logger.error(f"Angle normalization failed: {e}")
        return 0.0

def create_timezone_aware_datetime(timestamp: datetime, timezone_str: str) -> datetime:
    """
    Create timezone-aware datetime from timestamp and timezone string.
    
    Args:
        timestamp: Naive datetime
        timezone_str: Timezone string (e.g., 'UTC', 'America/New_York')
        
    Returns:
        Timezone-aware datetime
    """
    try:
        import pytz
        
        if timezone_str.upper() == 'UTC':
            tz = pytz.UTC
        else:
            tz = pytz.timezone(timezone_str)
        
        return tz.localize(timestamp)
        
    except Exception as e:
        logger.error(f"Timezone conversion failed: {e}")
        return timestamp.replace(tzinfo=timezone.utc)

def calculate_sun_angle(latitude: float, longitude: float, timestamp: datetime) -> Tuple[float, float]:
    """
    Calculate sun angle for given location and time.
    
    Args:
        latitude: Latitude in degrees
        longitude: Longitude in degrees
        timestamp: Timestamp
        
    Returns:
        Tuple of (azimuth, elevation) in degrees
    """
    try:
        # This is a simplified calculation
        # In practice, you'd use proper astronomical calculations
        
        # Convert to radians
        lat_rad = math.radians(latitude)
        lon_rad = math.radians(longitude)
        
        # Calculate day of year
        day_of_year = timestamp.timetuple().tm_yday
        
        # Calculate solar declination
        declination = math.radians(23.45 * math.sin(math.radians(360 * (284 + day_of_year) / 365)))
        
        # Calculate hour angle
        hour = timestamp.hour + timestamp.minute / 60.0
        hour_angle = math.radians(15 * (hour - 12))
        
        # Calculate elevation
        elevation = math.asin(
            math.sin(lat_rad) * math.sin(declination) +
            math.cos(lat_rad) * math.cos(declination) * math.cos(hour_angle)
        )
        
        # Calculate azimuth
        azimuth = math.atan2(
            math.sin(hour_angle),
            math.cos(hour_angle) * math.sin(lat_rad) - math.tan(declination) * math.cos(lat_rad)
        )
        
        # Convert to degrees
        elevation_deg = math.degrees(elevation)
        azimuth_deg = math.degrees(azimuth)
        
        # Normalize azimuth
        azimuth_deg = (azimuth_deg + 360) % 360
        
        return azimuth_deg, elevation_deg
        
    except Exception as e:
        logger.error(f"Sun angle calculation failed: {e}")
        return 0.0, 0.0

def create_grid_points(lat_range: Tuple[float, float], lon_range: Tuple[float, float], 
                      resolution: float) -> List[Tuple[float, float]]:
    """
    Create grid points for brute force search.
    
    Args:
        lat_range: Latitude range (min, max)
        lon_range: Longitude range (min, max)
        resolution: Grid resolution in degrees
        
    Returns:
        List of (latitude, longitude) tuples
    """
    try:
        points = []
        
        lat_min, lat_max = lat_range
        lon_min, lon_max = lon_range
        
        lat = lat_min
        while lat <= lat_max:
            lon = lon_min
            while lon <= lon_max:
                points.append((lat, lon))
                lon += resolution
            lat += resolution
        
        return points
        
    except Exception as e:
        logger.error(f"Grid point creation failed: {e}")
        return []

def calculate_confidence_from_error(error: float, max_error: float = 90.0) -> float:
    """
    Calculate confidence from error value.
    
    Args:
        error: Error value
        max_error: Maximum expected error
        
    Returns:
        Confidence value (0-1)
    """
    try:
        confidence = max(0.0, 1.0 - (error / max_error))
        return min(1.0, confidence)
        
    except Exception as e:
        logger.error(f"Confidence calculation failed: {e}")
        return 0.0

def smooth_confidence_surface(surface: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    """
    Smooth confidence surface using Gaussian filter.
    
    Args:
        surface: Confidence surface
        kernel_size: Kernel size for smoothing
        
    Returns:
        Smoothed surface
    """
    try:
        from scipy import ndimage
        
        # Apply Gaussian filter
        smoothed = ndimage.gaussian_filter(surface, sigma=kernel_size/3)
        
        return smoothed
        
    except Exception as e:
        logger.error(f"Surface smoothing failed: {e}")
        return surface

def find_peaks_in_surface(surface: np.ndarray, threshold: float = 0.5) -> List[Tuple[int, int]]:
    """
    Find peaks in confidence surface.
    
    Args:
        surface: Confidence surface
        threshold: Minimum peak threshold
        
    Returns:
        List of peak coordinates
    """
    try:
        from scipy import ndimage
        
        # Find local maxima
        local_maxima = ndimage.maximum_filter(surface, size=3) == surface
        
        # Apply threshold
        peaks = local_maxima & (surface > threshold)
        
        # Get peak coordinates
        peak_coords = np.where(peaks)
        peaks_list = list(zip(peak_coords[0], peak_coords[1]))
        
        return peaks_list
        
    except Exception as e:
        logger.error(f"Peak finding failed: {e}")
        return []

def interpolate_surface(surface: np.ndarray, factor: int = 2) -> np.ndarray:
    """
    Interpolate surface to higher resolution.
    
    Args:
        surface: Input surface
        factor: Interpolation factor
        
    Returns:
        Interpolated surface
    """
    try:
        # Use OpenCV for interpolation
        height, width = surface.shape
        new_height = height * factor
        new_width = width * factor
        
        interpolated = cv2.resize(surface, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
        
        return interpolated
        
    except Exception as e:
        logger.error(f"Surface interpolation failed: {e}")
        return surface

def calculate_surface_statistics(surface: np.ndarray) -> Dict:
    """
    Calculate statistics for confidence surface.
    
    Args:
        surface: Confidence surface
        
    Returns:
        Dictionary containing statistics
    """
    try:
        return {
            'mean': np.mean(surface),
            'std': np.std(surface),
            'min': np.min(surface),
            'max': np.max(surface),
            'median': np.median(surface),
            'q25': np.percentile(surface, 25),
            'q75': np.percentile(surface, 75)
        }
        
    except Exception as e:
        logger.error(f"Surface statistics calculation failed: {e}")
        return {}

def create_mask_from_surface(surface: np.ndarray, threshold: float) -> np.ndarray:
    """
    Create binary mask from confidence surface.
    
    Args:
        surface: Confidence surface
        threshold: Threshold value
        
    Returns:
        Binary mask
    """
    try:
        return (surface > threshold).astype(np.uint8)
        
    except Exception as e:
        logger.error(f"Mask creation failed: {e}")
        return np.zeros_like(surface, dtype=np.uint8)

def apply_mask_to_surface(surface: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Apply mask to confidence surface.
    
    Args:
        surface: Confidence surface
        mask: Binary mask
        
    Returns:
        Masked surface
    """
    try:
        return surface * mask
        
    except Exception as e:
        logger.error(f"Mask application failed: {e}")
        return surface