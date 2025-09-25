"""
EXIF data extraction module for SkyPin
Extracts timestamp, camera settings, and orientation from image metadata
"""

import io
from datetime import datetime, timezone
from typing import Dict, Optional, Tuple
import exifread
import piexif
from PIL import Image
import logging

logger = logging.getLogger(__name__)

def extract_exif_data(image_bytes: bytes) -> Dict:
    """
    Extract EXIF data from image bytes.
    
    Args:
        image_bytes: Raw image data
        
    Returns:
        Dictionary containing extracted EXIF information
    """
    try:
        # Try with exifread first
        exif_data = _extract_with_exifread(image_bytes)
        
        # Try with piexif for additional data
        piexif_data = _extract_with_piexif(image_bytes)
        
        # Merge data, preferring piexif for conflicts
        merged_data = {**exif_data, **piexif_data}
        
        return merged_data
        
    except Exception as e:
        logger.error(f"Error extracting EXIF data: {e}")
        return {}

def _extract_with_exifread(image_bytes: bytes) -> Dict:
    """Extract EXIF data using exifread library."""
    data = {}
    
    try:
        tags = exifread.process_file(io.BytesIO(image_bytes), details=False)
        
        # Extract timestamp
        if 'EXIF DateTimeOriginal' in tags:
            timestamp_str = str(tags['EXIF DateTimeOriginal'])
            data['timestamp'] = _parse_timestamp(timestamp_str)
        
        # Extract camera information
        if 'Image Make' in tags:
            data['make'] = str(tags['Image Make'])
        if 'Image Model' in tags:
            data['model'] = str(tags['Image Model'])
        
        # Extract focal length
        if 'EXIF FocalLength' in tags:
            focal_length = tags['EXIF FocalLength']
            data['focal_length'] = f"{focal_length}"
        
        # Extract orientation
        if 'Image Orientation' in tags:
            orientation = tags['Image Orientation']
            data['orientation'] = _parse_orientation(int(str(orientation)))
        
        # Extract GPS data if available
        gps_data = _extract_gps_data(tags)
        if gps_data:
            data.update(gps_data)
            
    except Exception as e:
        logger.warning(f"exifread extraction failed: {e}")
    
    return data

def _extract_with_piexif(image_bytes: bytes) -> Dict:
    """Extract EXIF data using piexif library."""
    data = {}
    
    try:
        exif_dict = piexif.load(image_bytes)
        
        # Extract timestamp
        if 'Exif' in exif_dict and piexif.ExifIFD.DateTimeOriginal in exif_dict['Exif']:
            timestamp_bytes = exif_dict['Exif'][piexif.ExifIFD.DateTimeOriginal]
            timestamp_str = timestamp_bytes.decode('utf-8')
            data['timestamp'] = _parse_timestamp(timestamp_str)
        
        # Extract camera information
        if '0th' in exif_dict:
            if piexif.ImageIFD.Make in exif_dict['0th']:
                data['make'] = exif_dict['0th'][piexif.ImageIFD.Make].decode('utf-8')
            if piexif.ImageIFD.Model in exif_dict['0th']:
                data['model'] = exif_dict['0th'][piexif.ImageIFD.Model].decode('utf-8')
        
        # Extract focal length
        if 'Exif' in exif_dict and piexif.ExifIFD.FocalLength in exif_dict['Exif']:
            focal_length = exif_dict['Exif'][piexif.ExifIFD.FocalLength]
            data['focal_length'] = f"{focal_length[0]}/{focal_length[1]}"
        
        # Extract orientation
        if '0th' in exif_dict and piexif.ImageIFD.Orientation in exif_dict['0th']:
            orientation = exif_dict['0th'][piexif.ImageIFD.Orientation]
            data['orientation'] = _parse_orientation(orientation)
            
    except Exception as e:
        logger.warning(f"piexif extraction failed: {e}")
    
    return data

def _parse_timestamp(timestamp_str: str) -> Optional[datetime]:
    """Parse timestamp string to datetime object."""
    try:
        # Common EXIF timestamp formats
        formats = [
            "%Y:%m:%d %H:%M:%S",
            "%Y-%m-%d %H:%M:%S",
            "%Y:%m:%d %H:%M:%S.%f"
        ]
        
        for fmt in formats:
            try:
                return datetime.strptime(timestamp_str, fmt).replace(tzinfo=timezone.utc)
            except ValueError:
                continue
        
        logger.warning(f"Could not parse timestamp: {timestamp_str}")
        return None
        
    except Exception as e:
        logger.error(f"Error parsing timestamp: {e}")
        return None

def _parse_orientation(orientation_code: int) -> str:
    """Parse orientation code to human-readable string."""
    orientations = {
        1: "Normal",
        2: "Mirrored horizontally",
        3: "Rotated 180°",
        4: "Mirrored vertically",
        5: "Mirrored horizontally, rotated 90° CCW",
        6: "Rotated 90° CW",
        7: "Mirrored horizontally, rotated 90° CW",
        8: "Rotated 90° CCW"
    }
    
    return orientations.get(orientation_code, f"Unknown ({orientation_code})")

def _extract_gps_data(tags) -> Dict:
    """Extract GPS data from EXIF tags."""
    gps_data = {}
    
    try:
        # GPS Latitude
        if 'GPS GPSLatitude' in tags and 'GPS GPSLatitudeRef' in tags:
            lat = _parse_gps_coordinate(
                str(tags['GPS GPSLatitude']),
                str(tags['GPS GPSLatitudeRef'])
            )
            if lat is not None:
                gps_data['gps_latitude'] = lat
        
        # GPS Longitude
        if 'GPS GPSLongitude' in tags and 'GPS GPSLongitudeRef' in tags:
            lon = _parse_gps_coordinate(
                str(tags['GPS GPSLongitude']),
                str(tags['GPS GPSLongitudeRef'])
            )
            if lon is not None:
                gps_data['gps_longitude'] = lon
        
        # GPS Altitude
        if 'GPS GPSAltitude' in tags:
            altitude = tags['GPS GPSAltitude']
            gps_data['gps_altitude'] = float(altitude)
            
    except Exception as e:
        logger.warning(f"Error extracting GPS data: {e}")
    
    return gps_data

def _parse_gps_coordinate(coord_str: str, ref: str) -> Optional[float]:
    """Parse GPS coordinate string to decimal degrees."""
    try:
        # Remove brackets and split by comma
        coord_str = coord_str.strip('[]')
        parts = coord_str.split(',')
        
        if len(parts) >= 2:
            degrees = float(parts[0].strip())
            minutes = float(parts[1].strip())
            seconds = float(parts[2].strip()) if len(parts) > 2 else 0.0
            
            decimal_degrees = degrees + minutes/60.0 + seconds/3600.0
            
            # Apply reference (N/S, E/W)
            if ref.upper() in ['S', 'W']:
                decimal_degrees = -decimal_degrees
            
            return decimal_degrees
            
    except Exception as e:
        logger.warning(f"Error parsing GPS coordinate: {e}")
    
    return None

def get_camera_info(exif_data: Dict) -> str:
    """Get formatted camera information string."""
    make = exif_data.get('make', 'Unknown')
    model = exif_data.get('model', 'Unknown')
    return f"{make} {model}".strip()

def validate_timestamp(timestamp: datetime) -> bool:
    """Validate that timestamp is reasonable for analysis."""
    if timestamp is None:
        return False
    
    # Check if timestamp is not too old or in the future
    now = datetime.now(timezone.utc)
    min_date = datetime(1990, 1, 1, tzinfo=timezone.utc)
    max_date = now.replace(year=now.year + 1)
    
    return min_date <= timestamp <= max_date

def get_timezone_offset(timestamp: datetime) -> Optional[str]:
    """Extract timezone information from timestamp if available."""
    if timestamp and timestamp.tzinfo:
        offset = timestamp.utcoffset()
        if offset:
            hours = offset.total_seconds() / 3600
            return f"UTC{hours:+03.0f}:00"
    
    return None