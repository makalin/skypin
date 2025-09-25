"""
Confidence mapping module for SkyPin
Creates confidence surfaces and GeoJSON heat maps from location matches
"""

import numpy as np
import json
from typing import Dict, List, Tuple, Optional
import logging
from scipy import ndimage
from scipy.spatial.distance import cdist
import geopandas as gpd
from shapely.geometry import Point, Polygon
import pandas as pd

logger = logging.getLogger(__name__)

class ConfidenceMapper:
    """Creates confidence maps and heat maps from location matches."""
    
    def __init__(self):
        """Initialize confidence mapper."""
        self.kernel_size = 5  # Kernel size for smoothing
        self.confidence_threshold = 0.1  # Minimum confidence threshold
        
    def create_confidence_map(self, location_results: Dict, confidence_threshold: float = 0.1) -> Dict:
        """
        Create confidence map from location calculation results.
        
        Args:
            location_results: Results from astronomy calculator
            confidence_threshold: Minimum confidence threshold
            
        Returns:
            Dictionary containing confidence map data
        """
        try:
            matches = location_results.get('matches', [])
            if not matches:
                return {'heatmap_data': [], 'confidence_surface': None}
            
            # Filter matches by confidence threshold
            filtered_matches = [m for m in matches if m['confidence'] >= confidence_threshold]
            
            if not filtered_matches:
                return {'heatmap_data': [], 'confidence_surface': None}
            
            # Create heatmap data
            heatmap_data = self._create_heatmap_data(filtered_matches)
            
            # Create confidence surface
            confidence_surface = self._create_confidence_surface(filtered_matches)
            
            # Create GeoJSON
            geojson_data = self._create_geojson(filtered_matches)
            
            return {
                'heatmap_data': heatmap_data,
                'confidence_surface': confidence_surface,
                'geojson': geojson_data,
                'num_points': len(filtered_matches),
                'confidence_range': (min(m['confidence'] for m in filtered_matches),
                                   max(m['confidence'] for m in filtered_matches))
            }
            
        except Exception as e:
            logger.error(f"Confidence map creation failed: {e}")
            return {'heatmap_data': [], 'confidence_surface': None}
    
    def _create_heatmap_data(self, matches: List[Dict]) -> List[List[float]]:
        """Create heatmap data for visualization."""
        try:
            heatmap_data = []
            
            for match in matches:
                lat = match['latitude']
                lon = match['longitude']
                confidence = match['confidence']
                
                # Add point to heatmap data
                heatmap_data.append([lat, lon, confidence])
            
            return heatmap_data
            
        except Exception as e:
            logger.error(f"Heatmap data creation failed: {e}")
            return []
    
    def _create_confidence_surface(self, matches: List[Dict]) -> Optional[Dict]:
        """Create confidence surface using kernel density estimation."""
        try:
            if len(matches) < 3:
                return None
            
            # Extract coordinates and confidences
            lats = np.array([m['latitude'] for m in matches])
            lons = np.array([m['longitude'] for m in matches])
            confidences = np.array([m['confidence'] for m in matches])
            
            # Create grid
            lat_min, lat_max = lats.min(), lats.max()
            lon_min, lon_max = lons.min(), lons.max()
            
            # Add padding
            lat_padding = (lat_max - lat_min) * 0.1
            lon_padding = (lon_max - lon_min) * 0.1
            
            lat_min -= lat_padding
            lat_max += lat_padding
            lon_min -= lon_padding
            lon_max += lon_padding
            
            # Create grid points
            lat_grid = np.linspace(lat_min, lat_max, 50)
            lon_grid = np.linspace(lon_min, lon_max, 50)
            lat_mesh, lon_mesh = np.meshgrid(lat_grid, lon_grid)
            
            # Calculate kernel density
            confidence_surface = self._calculate_kernel_density(
                lats, lons, confidences, lat_mesh, lon_mesh
            )
            
            return {
                'lat_grid': lat_grid.tolist(),
                'lon_grid': lon_grid.tolist(),
                'confidence_surface': confidence_surface.tolist(),
                'bounds': {
                    'lat_min': lat_min,
                    'lat_max': lat_max,
                    'lon_min': lon_min,
                    'lon_max': lon_max
                }
            }
            
        except Exception as e:
            logger.error(f"Confidence surface creation failed: {e}")
            return None
    
    def _calculate_kernel_density(self, lats: np.ndarray, lons: np.ndarray, 
                                confidences: np.ndarray, lat_mesh: np.ndarray, 
                                lon_mesh: np.ndarray) -> np.ndarray:
        """Calculate kernel density estimation."""
        try:
            # Create coordinate arrays
            points = np.column_stack([lats, lons])
            grid_points = np.column_stack([lat_mesh.ravel(), lon_mesh.ravel()])
            
            # Calculate distances
            distances = cdist(grid_points, points)
            
            # Apply Gaussian kernel
            sigma = 0.5  # Bandwidth
            kernel = np.exp(-distances**2 / (2 * sigma**2))
            
            # Weight by confidence
            weighted_kernel = kernel * confidences
            
            # Sum over points
            density = np.sum(weighted_kernel, axis=1)
            
            # Reshape to grid
            density_grid = density.reshape(lat_mesh.shape)
            
            # Normalize
            if density_grid.max() > 0:
                density_grid = density_grid / density_grid.max()
            
            return density_grid
            
        except Exception as e:
            logger.error(f"Kernel density calculation failed: {e}")
            return np.zeros_like(lat_mesh)
    
    def _create_geojson(self, matches: List[Dict]) -> Dict:
        """Create GeoJSON from matches."""
        try:
            features = []
            
            for i, match in enumerate(matches):
                lat = match['latitude']
                lon = match['longitude']
                confidence = match['confidence']
                
                # Create point geometry
                point = Point(lon, lat)  # Note: GeoJSON uses lon, lat order
                
                # Create feature
                feature = {
                    'type': 'Feature',
                    'geometry': {
                        'type': 'Point',
                        'coordinates': [lon, lat]
                    },
                    'properties': {
                        'confidence': confidence,
                        'latitude': lat,
                        'longitude': lon,
                        'azimuth_error': match.get('azimuth_error', 0),
                        'elevation_error': match.get('elevation_error', 0),
                        'total_error': match.get('total_error', 0)
                    }
                }
                
                features.append(feature)
            
            # Create GeoJSON
            geojson = {
                'type': 'FeatureCollection',
                'features': features
            }
            
            return geojson
            
        except Exception as e:
            logger.error(f"GeoJSON creation failed: {e}")
            return {'type': 'FeatureCollection', 'features': []}
    
    def create_confidence_contours(self, confidence_surface: Dict, 
                                 levels: List[float] = [0.1, 0.3, 0.5, 0.7, 0.9]) -> List[Dict]:
        """Create confidence contours from confidence surface."""
        try:
            if not confidence_surface:
                return []
            
            lat_grid = np.array(confidence_surface['lat_grid'])
            lon_grid = np.array(confidence_surface['lon_grid'])
            surface = np.array(confidence_surface['confidence_surface'])
            
            contours = []
            
            for level in levels:
                # Find contour at this level
                contour_points = self._find_contour_points(surface, level)
                
                if contour_points:
                    # Convert to lat/lon coordinates
                    contour_coords = []
                    for point in contour_points:
                        lat_idx, lon_idx = point
                        lat = lat_grid[lat_idx]
                        lon = lon_grid[lon_idx]
                        contour_coords.append([lon, lat])  # GeoJSON order
                    
                    contours.append({
                        'level': level,
                        'coordinates': contour_coords,
                        'type': 'contour'
                    })
            
            return contours
            
        except Exception as e:
            logger.error(f"Contour creation failed: {e}")
            return []
    
    def _find_contour_points(self, surface: np.ndarray, level: float) -> List[Tuple[int, int]]:
        """Find contour points at given level."""
        try:
            # Find points where surface crosses the level
            contour_points = []
            
            for i in range(surface.shape[0] - 1):
                for j in range(surface.shape[1] - 1):
                    # Check if level crosses between adjacent points
                    if ((surface[i, j] <= level < surface[i, j+1]) or
                        (surface[i, j] >= level > surface[i, j+1]) or
                        (surface[i, j] <= level < surface[i+1, j]) or
                        (surface[i, j] >= level > surface[i+1, j])):
                        contour_points.append((i, j))
            
            return contour_points
            
        except Exception as e:
            logger.error(f"Contour point finding failed: {e}")
            return []
    
    def create_uncertainty_ellipse(self, matches: List[Dict]) -> Optional[Dict]:
        """Create uncertainty ellipse from matches."""
        try:
            if len(matches) < 3:
                return None
            
            # Extract coordinates
            lats = np.array([m['latitude'] for m in matches])
            lons = np.array([m['longitude'] for m in matches])
            confidences = np.array([m['confidence'] for m in matches])
            
            # Calculate weighted center
            total_weight = np.sum(confidences)
            center_lat = np.sum(lats * confidences) / total_weight
            center_lon = np.sum(lons * confidences) / total_weight
            
            # Calculate covariance matrix
            lat_diff = lats - center_lat
            lon_diff = lons - center_lon
            
            lat_var = np.sum(confidences * lat_diff**2) / total_weight
            lon_var = np.sum(confidences * lon_diff**2) / total_weight
            lat_lon_cov = np.sum(confidences * lat_diff * lon_diff) / total_weight
            
            cov_matrix = np.array([[lat_var, lat_lon_cov], [lat_lon_cov, lon_var]])
            
            # Calculate eigenvalues and eigenvectors
            eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
            
            # Calculate semi-major and semi-minor axes
            semi_major = np.sqrt(eigenvalues[1]) * 111.32  # Convert to km
            semi_minor = np.sqrt(eigenvalues[0]) * 111.32  # Convert to km
            
            # Calculate orientation
            orientation = np.arctan2(eigenvectors[1, 1], eigenvectors[0, 1]) * 180 / np.pi
            
            return {
                'center_lat': center_lat,
                'center_lon': center_lon,
                'semi_major': semi_major,
                'semi_minor': semi_minor,
                'orientation': orientation,
                'area': np.pi * semi_major * semi_minor
            }
            
        except Exception as e:
            logger.error(f"Uncertainty ellipse creation failed: {e}")
            return None
    
    def export_geojson(self, geojson_data: Dict, filename: str) -> bool:
        """Export GeoJSON data to file."""
        try:
            with open(filename, 'w') as f:
                json.dump(geojson_data, f, indent=2)
            
            logger.info(f"GeoJSON exported to {filename}")
            return True
            
        except Exception as e:
            logger.error(f"GeoJSON export failed: {e}")
            return False
    
    def create_confidence_statistics(self, matches: List[Dict]) -> Dict:
        """Create confidence statistics from matches."""
        try:
            if not matches:
                return {}
            
            confidences = [m['confidence'] for m in matches]
            
            return {
                'mean': np.mean(confidences),
                'median': np.median(confidences),
                'std': np.std(confidences),
                'min': np.min(confidences),
                'max': np.max(confidences),
                'q25': np.percentile(confidences, 25),
                'q75': np.percentile(confidences, 75),
                'count': len(confidences)
            }
            
        except Exception as e:
            logger.error(f"Confidence statistics creation failed: {e}")
            return {}

# Global mapper instance
_mapper = None

def create_confidence_map(location_results: Dict, confidence_threshold: float = 0.1) -> Dict:
    """
    Create confidence map from location calculation results.
    
    Args:
        location_results: Results from astronomy calculator
        confidence_threshold: Minimum confidence threshold
        
    Returns:
        Dictionary containing confidence map data
    """
    global _mapper
    
    if _mapper is None:
        _mapper = ConfidenceMapper()
    
    return _mapper.create_confidence_map(location_results, confidence_threshold)

def create_uncertainty_ellipse(matches: List[Dict]) -> Optional[Dict]:
    """
    Create uncertainty ellipse from matches.
    
    Args:
        matches: List of location matches
        
    Returns:
        Dictionary containing uncertainty ellipse data
    """
    global _mapper
    
    if _mapper is None:
        _mapper = ConfidenceMapper()
    
    return _mapper.create_uncertainty_ellipse(matches)

def export_geojson(geojson_data: Dict, filename: str) -> bool:
    """
    Export GeoJSON data to file.
    
    Args:
        geojson_data: GeoJSON data
        filename: Output filename
        
    Returns:
        True if successful, False otherwise
    """
    global _mapper
    
    if _mapper is None:
        _mapper = ConfidenceMapper()
    
    return _mapper.export_geojson(geojson_data, filename)