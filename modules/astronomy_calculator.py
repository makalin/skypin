"""
Astronomy calculator module for SkyPin using Skyfield
Performs astronomical calculations to determine location from sun position
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timezone, timedelta
import logging
from skyfield.api import load, Topos
from skyfield.timelib import Time
from skyfield.positionlib import Apparent
import math

logger = logging.getLogger(__name__)

class AstronomyCalculator:
    """Astronomical calculations for location determination."""
    
    def __init__(self):
        """Initialize astronomy calculator."""
        self.ephemeris = None
        self._load_ephemeris()
        
        # Grid parameters
        self.lat_range = (-90, 90)
        self.lon_range = (-180, 180)
        self.grid_resolution = 1.0  # degrees
        
    def _load_ephemeris(self):
        """Load astronomical ephemeris data."""
        try:
            # Load JPL ephemeris (this will download if not present)
            self.ephemeris = load('de421.bsp')
            logger.info("Loaded JPL ephemeris data")
        except Exception as e:
            logger.error(f"Failed to load ephemeris: {e}")
            self.ephemeris = None
    
    def calculate_location(self, sun_data: Dict, shadow_data: Dict, 
                         exif_data: Dict, grid_resolution: float = 1.0,
                         timezone_override: Optional[str] = None) -> Dict:
        """
        Calculate location from sun position and shadow data.
        
        Args:
            sun_data: Sun detection results
            shadow_data: Shadow analysis results
            exif_data: EXIF metadata
            grid_resolution: Grid resolution in degrees
            timezone_override: Override timezone if needed
            
        Returns:
            Dictionary containing location calculation results
        """
        try:
            if not self.ephemeris:
                raise ValueError("Ephemeris data not loaded")
            
            # Extract timestamp
            timestamp = exif_data.get('timestamp')
            if not timestamp:
                raise ValueError("No timestamp found in EXIF data")
            
            # Get sun position
            sun_azimuth = sun_data.get('azimuth')
            sun_elevation = sun_data.get('elevation')
            
            if sun_azimuth is None or sun_elevation is None:
                raise ValueError("Sun position not detected")
            
            # Get shadow azimuth (if available)
            shadow_azimuth = shadow_data.get('azimuth')
            
            # Use shadow azimuth if more reliable than sun azimuth
            if shadow_azimuth is not None and shadow_data.get('quality', 0) > 0.5:
                azimuth = shadow_azimuth
                azimuth_confidence = shadow_data.get('quality', 0)
            else:
                azimuth = sun_azimuth
                azimuth_confidence = sun_data.get('confidence', 0)
            
            # Create time grid for brute force search
            time_grid = self._create_time_grid(timestamp, timezone_override)
            
            # Create location grid
            location_grid = self._create_location_grid(grid_resolution)
            
            # Perform brute force search
            best_matches = self._brute_force_search(
                azimuth, sun_elevation, time_grid, location_grid
            )
            
            # Calculate best location
            best_location = self._calculate_best_location(best_matches)
            
            # Calculate uncertainty
            uncertainty = self._calculate_uncertainty(best_matches)
            
            return {
                'best_location': best_location,
                'uncertainty': uncertainty,
                'matches': best_matches,
                'confidence': best_location.get('confidence', 0) if best_location else 0,
                'method': 'brute_force_astronomy'
            }
            
        except Exception as e:
            logger.error(f"Location calculation failed: {e}")
            return {
                'best_location': None,
                'uncertainty': None,
                'matches': [],
                'confidence': 0,
                'method': 'error'
            }
    
    def _create_time_grid(self, timestamp: datetime, timezone_override: Optional[str]) -> List[datetime]:
        """Create time grid around the given timestamp."""
        try:
            # Create time range (±1 hour around timestamp)
            time_range = timedelta(hours=1)
            start_time = timestamp - time_range
            end_time = timestamp + time_range
            
            # Create grid with 15-minute intervals
            time_grid = []
            current_time = start_time
            while current_time <= end_time:
                time_grid.append(current_time)
                current_time += timedelta(minutes=15)
            
            return time_grid
            
        except Exception as e:
            logger.error(f"Time grid creation failed: {e}")
            return [timestamp]
    
    def _create_location_grid(self, resolution: float) -> List[Tuple[float, float]]:
        """Create location grid for brute force search."""
        try:
            locations = []
            
            # Create latitude grid
            lat_start = self.lat_range[0]
            lat_end = self.lat_range[1]
            lat_step = resolution
            
            # Create longitude grid
            lon_start = self.lon_range[0]
            lon_end = self.lon_range[1]
            lon_step = resolution
            
            # Generate grid points
            lat = lat_start
            while lat <= lat_end:
                lon = lon_start
                while lon <= lon_end:
                    locations.append((lat, lon))
                    lon += lon_step
                lat += lat_step
            
            return locations
            
        except Exception as e:
            logger.error(f"Location grid creation failed: {e}")
            return []
    
    def _brute_force_search(self, observed_azimuth: float, observed_elevation: float,
                           time_grid: List[datetime], location_grid: List[Tuple[float, float]]) -> List[Dict]:
        """Perform brute force search for best location matches."""
        try:
            matches = []
            
            for lat, lon in location_grid:
                for timestamp in time_grid:
                    try:
                        # Calculate predicted sun position
                        predicted_azimuth, predicted_elevation = self._calculate_sun_position(
                            lat, lon, timestamp
                        )
                        
                        if predicted_azimuth is not None and predicted_elevation is not None:
                            # Calculate error
                            azimuth_error = self._angle_difference(observed_azimuth, predicted_azimuth)
                            elevation_error = abs(observed_elevation - predicted_elevation)
                            
                            # Calculate total error (weighted)
                            total_error = (azimuth_error * 0.7 + elevation_error * 0.3)
                            
                            # Calculate confidence
                            confidence = max(0, 1.0 - total_error / 90.0)
                            
                            matches.append({
                                'latitude': lat,
                                'longitude': lon,
                                'timestamp': timestamp,
                                'predicted_azimuth': predicted_azimuth,
                                'predicted_elevation': predicted_elevation,
                                'azimuth_error': azimuth_error,
                                'elevation_error': elevation_error,
                                'total_error': total_error,
                                'confidence': confidence
                            })
                            
                    except Exception as e:
                        logger.debug(f"Error calculating sun position for {lat}, {lon}: {e}")
                        continue
            
            # Sort by confidence (highest first)
            matches.sort(key=lambda x: x['confidence'], reverse=True)
            
            return matches
            
        except Exception as e:
            logger.error(f"Brute force search failed: {e}")
            return []
    
    def _calculate_sun_position(self, latitude: float, longitude: float, timestamp: datetime) -> Tuple[Optional[float], Optional[float]]:
        """Calculate sun position for given location and time."""
        try:
            # Convert to Skyfield time
            ts = load.timescale()
            skyfield_time = ts.from_datetime(timestamp)
            
            # Create observer location
            observer = Topos(latitude_degrees=latitude, longitude_degrees=longitude)
            
            # Get sun position
            sun = self.ephemeris['sun']
            astrometric = observer.at(skyfield_time).observe(sun)
            apparent = astrometric.apparent()
            
            # Get altitude and azimuth
            alt, az, distance = apparent.altaz()
            
            # Convert to degrees
            elevation = alt.degrees
            azimuth = az.degrees
            
            return azimuth, elevation
            
        except Exception as e:
            logger.debug(f"Sun position calculation failed: {e}")
            return None, None
    
    def _angle_difference(self, angle1: float, angle2: float) -> float:
        """Calculate the difference between two angles."""
        diff = abs(angle1 - angle2)
        return min(diff, 360 - diff)
    
    def _calculate_best_location(self, matches: List[Dict]) -> Optional[Dict]:
        """Calculate the best location from matches."""
        try:
            if not matches:
                return None
            
            # Take top 1% of matches
            top_matches = matches[:max(1, len(matches) // 100)]
            
            if not top_matches:
                return None
            
            # Calculate weighted average
            total_weight = 0
            weighted_lat = 0
            weighted_lon = 0
            weighted_confidence = 0
            
            for match in top_matches:
                weight = match['confidence']
                total_weight += weight
                weighted_lat += match['latitude'] * weight
                weighted_lon += match['longitude'] * weight
                weighted_confidence += weight
            
            if total_weight > 0:
                avg_lat = weighted_lat / total_weight
                avg_lon = weighted_lon / total_weight
                avg_confidence = weighted_confidence / total_weight
                
                return {
                    'latitude': avg_lat,
                    'longitude': avg_lon,
                    'confidence': avg_confidence,
                    'num_matches': len(top_matches)
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Best location calculation failed: {e}")
            return None
    
    def _calculate_uncertainty(self, matches: List[Dict]) -> Optional[Dict]:
        """Calculate uncertainty ellipse from matches."""
        try:
            if len(matches) < 3:
                return None
            
            # Take top matches
            top_matches = matches[:min(50, len(matches))]
            
            latitudes = [m['latitude'] for m in top_matches]
            longitudes = [m['longitude'] for m in top_matches]
            
            # Calculate covariance matrix
            lat_mean = np.mean(latitudes)
            lon_mean = np.mean(longitudes)
            
            lat_var = np.var(latitudes)
            lon_var = np.var(longitudes)
            lat_lon_cov = np.cov(latitudes, longitudes)[0, 1]
            
            # Calculate eigenvalues and eigenvectors
            cov_matrix = np.array([[lat_var, lat_lon_cov], [lat_lon_cov, lon_var]])
            eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
            
            # Calculate semi-major and semi-minor axes
            semi_major = np.sqrt(eigenvalues[1]) * 111.32  # Convert to km
            semi_minor = np.sqrt(eigenvalues[0]) * 111.32  # Convert to km
            
            # Calculate orientation
            orientation = np.arctan2(eigenvectors[1, 1], eigenvectors[0, 1]) * 180 / np.pi
            
            return {
                'semi_major': semi_major,
                'semi_minor': semi_minor,
                'orientation': orientation,
                'center_lat': lat_mean,
                'center_lon': lon_mean
            }
            
        except Exception as e:
            logger.error(f"Uncertainty calculation failed: {e}")
            return None
    
    def calculate_sun_position_for_location(self, latitude: float, longitude: float, 
                                         timestamp: datetime) -> Tuple[Optional[float], Optional[float]]:
        """Calculate sun position for a specific location and time."""
        return self._calculate_sun_position(latitude, longitude, timestamp)
    
    def get_sun_rise_set(self, latitude: float, longitude: float, 
                        date: datetime) -> Dict:
        """Calculate sunrise and sunset times for a location."""
        try:
            ts = load.timescale()
            observer = Topos(latitude_degrees=latitude, longitude_degrees=longitude)
            sun = self.ephemeris['sun']
            
            # Calculate sunrise and sunset
            t0 = ts.from_datetime(date.replace(hour=0, minute=0, second=0))
            t1 = ts.from_datetime(date.replace(hour=23, minute=59, second=59))
            
            times, events = observer.find_discrete(t0, t1, sun)
            
            sunrise = None
            sunset = None
            
            for t, event in zip(times, events):
                if event == 0:  # Sunrise
                    sunrise = t.utc_datetime()
                elif event == 1:  # Sunset
                    sunset = t.utc_datetime()
            
            return {
                'sunrise': sunrise,
                'sunset': sunset,
                'day_length': (sunset - sunrise).total_seconds() / 3600 if sunrise and sunset else None
            }
            
        except Exception as e:
            logger.error(f"Sunrise/sunset calculation failed: {e}")
            return {'sunrise': None, 'sunset': None, 'day_length': None}

# Global calculator instance
_calculator = None

def calculate_location(sun_data: Dict, shadow_data: Dict, exif_data: Dict, 
                     grid_resolution: float = 1.0, timezone_override: Optional[str] = None) -> Dict:
    """
    Calculate location from sun position and shadow data.
    
    Args:
        sun_data: Sun detection results
        shadow_data: Shadow analysis results
        exif_data: EXIF metadata
        grid_resolution: Grid resolution in degrees
        timezone_override: Override timezone if needed
        
    Returns:
        Dictionary containing location calculation results
    """
    global _calculator
    
    if _calculator is None:
        _calculator = AstronomyCalculator()
    
    return _calculator.calculate_location(sun_data, shadow_data, exif_data, grid_resolution, timezone_override)

def calculate_sun_position_for_location(latitude: float, longitude: float, 
                                      timestamp: datetime) -> Tuple[Optional[float], Optional[float]]:
    """
    Calculate sun position for a specific location and time.
    
    Args:
        latitude: Latitude in degrees
        longitude: Longitude in degrees
        timestamp: Timestamp
        
    Returns:
        Tuple of (azimuth, elevation) in degrees
    """
    global _calculator
    
    if _calculator is None:
        _calculator = AstronomyCalculator()
    
    return _calculator.calculate_sun_position_for_location(latitude, longitude, timestamp)

def get_sun_rise_set(latitude: float, longitude: float, date: datetime) -> Dict:
    """
    Calculate sunrise and sunset times for a location.
    
    Args:
        latitude: Latitude in degrees
        longitude: Longitude in degrees
        date: Date
        
    Returns:
        Dictionary containing sunrise/sunset times
    """
    global _calculator
    
    if _calculator is None:
        _calculator = AstronomyCalculator()
    
    return _calculator.get_sun_rise_set(latitude, longitude, date)