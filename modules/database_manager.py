"""
Database manager module for SkyPin
Manages storage and retrieval of analysis results
"""

import sqlite3
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timezone
import hashlib
import numpy as np
from contextlib import contextmanager

logger = logging.getLogger(__name__)

class DatabaseManager:
    """Manages SQLite database for storing analysis results."""
    
    def __init__(self, db_path: str = "skypin.db"):
        """
        Initialize database manager.
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_database()
    
    def _init_database(self):
        """Initialize database tables."""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                # Create images table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS images (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        file_path TEXT UNIQUE NOT NULL,
                        file_name TEXT NOT NULL,
                        file_size INTEGER,
                        file_hash TEXT UNIQUE,
                        image_width INTEGER,
                        image_height INTEGER,
                        image_channels INTEGER,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # Create exif_data table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS exif_data (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        image_id INTEGER NOT NULL,
                        timestamp TEXT,
                        make TEXT,
                        model TEXT,
                        focal_length TEXT,
                        orientation TEXT,
                        gps_latitude REAL,
                        gps_longitude REAL,
                        gps_altitude REAL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (image_id) REFERENCES images (id)
                    )
                """)
                
                # Create sun_detections table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS sun_detections (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        image_id INTEGER NOT NULL,
                        detected BOOLEAN NOT NULL,
                        azimuth REAL,
                        elevation REAL,
                        confidence REAL,
                        center_x INTEGER,
                        center_y INTEGER,
                        bbox_x1 INTEGER,
                        bbox_y1 INTEGER,
                        bbox_x2 INTEGER,
                        bbox_y2 INTEGER,
                        method TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (image_id) REFERENCES images (id)
                    )
                """)
                
                # Create shadow_analyses table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS shadow_analyses (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        image_id INTEGER NOT NULL,
                        detected BOOLEAN NOT NULL,
                        azimuth REAL,
                        length REAL,
                        quality REAL,
                        method TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (image_id) REFERENCES images (id)
                    )
                """)
                
                # Create cloud_detections table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS cloud_detections (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        image_id INTEGER NOT NULL,
                        clouds_detected BOOLEAN NOT NULL,
                        cloud_coverage REAL,
                        sky_coverage REAL,
                        weather_conditions TEXT,
                        confidence REAL,
                        method TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (image_id) REFERENCES images (id)
                    )
                """)
                
                # Create moon_detections table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS moon_detections (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        image_id INTEGER NOT NULL,
                        detected BOOLEAN NOT NULL,
                        center_x INTEGER,
                        center_y INTEGER,
                        radius REAL,
                        brightness REAL,
                        phase TEXT,
                        azimuth REAL,
                        elevation REAL,
                        confidence REAL,
                        method TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (image_id) REFERENCES images (id)
                    )
                """)
                
                # Create star_analyses table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS star_analyses (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        image_id INTEGER NOT NULL,
                        stars_detected INTEGER,
                        trails_detected INTEGER,
                        celestial_pole_x REAL,
                        celestial_pole_y REAL,
                        exposure_time REAL,
                        confidence REAL,
                        method TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (image_id) REFERENCES images (id)
                    )
                """)
                
                # Create tamper_detections table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS tamper_detections (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        image_id INTEGER NOT NULL,
                        is_tampered BOOLEAN NOT NULL,
                        score REAL,
                        error_level_score REAL,
                        jpeg_ghost_score REAL,
                        noise_analysis_score REAL,
                        compression_analysis_score REAL,
                        confidence REAL,
                        method TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (image_id) REFERENCES images (id)
                    )
                """)
                
                # Create location_results table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS location_results (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        image_id INTEGER NOT NULL,
                        latitude REAL,
                        longitude REAL,
                        confidence REAL,
                        uncertainty_semi_major REAL,
                        uncertainty_semi_minor REAL,
                        uncertainty_orientation REAL,
                        num_matches INTEGER,
                        method TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (image_id) REFERENCES images (id)
                    )
                """)
                
                # Create confidence_maps table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS confidence_maps (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        image_id INTEGER NOT NULL,
                        heatmap_data TEXT,  -- JSON
                        confidence_surface TEXT,  -- JSON
                        geojson_data TEXT,  -- JSON
                        num_points INTEGER,
                        confidence_range_min REAL,
                        confidence_range_max REAL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (image_id) REFERENCES images (id)
                    )
                """)
                
                # Create analysis_sessions table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS analysis_sessions (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        session_name TEXT,
                        description TEXT,
                        total_images INTEGER,
                        successful_analyses INTEGER,
                        failed_analyses INTEGER,
                        processing_time REAL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # Create indexes
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_images_file_path ON images (file_path)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_images_file_hash ON images (file_hash)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_exif_image_id ON exif_data (image_id)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_sun_image_id ON sun_detections (image_id)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_shadow_image_id ON shadow_analyses (image_id)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_cloud_image_id ON cloud_detections (image_id)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_moon_image_id ON moon_detections (image_id)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_star_image_id ON star_analyses (image_id)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_tamper_image_id ON tamper_detections (image_id)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_location_image_id ON location_results (image_id)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_confidence_image_id ON confidence_maps (image_id)")
                
                conn.commit()
                logger.info("Database initialized successfully")
                
        except Exception as e:
            logger.error(f"Database initialization failed: {e}")
            raise
    
    @contextmanager
    def _get_connection(self):
        """Get database connection with context manager."""
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()
    
    def _calculate_file_hash(self, file_path: str) -> str:
        """Calculate SHA256 hash of file."""
        try:
            with open(file_path, 'rb') as f:
                return hashlib.sha256(f.read()).hexdigest()
        except Exception as e:
            logger.error(f"Hash calculation failed: {e}")
            return ""
    
    def store_image(self, file_path: str, image_shape: Tuple[int, int, int]) -> int:
        """
        Store image metadata in database.
        
        Args:
            file_path: Path to image file
            image_shape: Image shape (height, width, channels)
            
        Returns:
            Image ID
        """
        try:
            file_path = str(Path(file_path).resolve())
            file_name = Path(file_path).name
            file_size = Path(file_path).stat().st_size
            file_hash = self._calculate_file_hash(file_path)
            height, width, channels = image_shape
            
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                # Check if image already exists
                cursor.execute("SELECT id FROM images WHERE file_hash = ?", (file_hash,))
                existing = cursor.fetchone()
                
                if existing:
                    return existing['id']
                
                # Insert new image
                cursor.execute("""
                    INSERT INTO images (file_path, file_name, file_size, file_hash, 
                                      image_width, image_height, image_channels)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (file_path, file_name, file_size, file_hash, width, height, channels))
                
                image_id = cursor.lastrowid
                conn.commit()
                
                logger.info(f"Stored image {file_name} with ID {image_id}")
                return image_id
                
        except Exception as e:
            logger.error(f"Failed to store image: {e}")
            raise
    
    def store_exif_data(self, image_id: int, exif_data: Dict) -> int:
        """Store EXIF data."""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    INSERT INTO exif_data (image_id, timestamp, make, model, focal_length,
                                         orientation, gps_latitude, gps_longitude, gps_altitude)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    image_id,
                    exif_data.get('timestamp'),
                    exif_data.get('make'),
                    exif_data.get('model'),
                    exif_data.get('focal_length'),
                    exif_data.get('orientation'),
                    exif_data.get('gps_latitude'),
                    exif_data.get('gps_longitude'),
                    exif_data.get('gps_altitude')
                ))
                
                exif_id = cursor.lastrowid
                conn.commit()
                
                return exif_id
                
        except Exception as e:
            logger.error(f"Failed to store EXIF data: {e}")
            raise
    
    def store_sun_detection(self, image_id: int, sun_result: Dict) -> int:
        """Store sun detection results."""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                center = sun_result.get('center')
                bbox = sun_result.get('bbox')
                
                cursor.execute("""
                    INSERT INTO sun_detections (image_id, detected, azimuth, elevation, confidence,
                                              center_x, center_y, bbox_x1, bbox_y1, bbox_x2, bbox_y2, method)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    image_id,
                    sun_result.get('detected', False),
                    sun_result.get('azimuth'),
                    sun_result.get('elevation'),
                    sun_result.get('confidence'),
                    center[0] if center else None,
                    center[1] if center else None,
                    bbox[0] if bbox and len(bbox) > 0 else None,
                    bbox[1] if bbox and len(bbox) > 1 else None,
                    bbox[2] if bbox and len(bbox) > 2 else None,
                    bbox[3] if bbox and len(bbox) > 3 else None,
                    sun_result.get('method')
                ))
                
                sun_id = cursor.lastrowid
                conn.commit()
                
                return sun_id
                
        except Exception as e:
            logger.error(f"Failed to store sun detection: {e}")
            raise
    
    def store_shadow_analysis(self, image_id: int, shadow_result: Dict) -> int:
        """Store shadow analysis results."""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    INSERT INTO shadow_analyses (image_id, detected, azimuth, length, quality, method)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    image_id,
                    shadow_result.get('detected', False),
                    shadow_result.get('azimuth'),
                    shadow_result.get('length'),
                    shadow_result.get('quality'),
                    shadow_result.get('method')
                ))
                
                shadow_id = cursor.lastrowid
                conn.commit()
                
                return shadow_id
                
        except Exception as e:
            logger.error(f"Failed to store shadow analysis: {e}")
            raise
    
    def store_cloud_detection(self, image_id: int, cloud_result: Dict) -> int:
        """Store cloud detection results."""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    INSERT INTO cloud_detections (image_id, clouds_detected, cloud_coverage,
                                                sky_coverage, weather_conditions, confidence, method)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    image_id,
                    cloud_result.get('clouds_detected', False),
                    cloud_result.get('cloud_coverage'),
                    cloud_result.get('sky_coverage'),
                    cloud_result.get('weather_conditions'),
                    cloud_result.get('confidence'),
                    cloud_result.get('method')
                ))
                
                cloud_id = cursor.lastrowid
                conn.commit()
                
                return cloud_id
                
        except Exception as e:
            logger.error(f"Failed to store cloud detection: {e}")
            raise
    
    def store_moon_detection(self, image_id: int, moon_result: Dict) -> int:
        """Store moon detection results."""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                center = moon_result.get('center')
                
                cursor.execute("""
                    INSERT INTO moon_detections (image_id, detected, center_x, center_y, radius,
                                               brightness, phase, azimuth, elevation, confidence, method)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    image_id,
                    moon_result.get('detected', False),
                    center[0] if center else None,
                    center[1] if center else None,
                    moon_result.get('radius'),
                    moon_result.get('brightness'),
                    moon_result.get('phase'),
                    moon_result.get('azimuth'),
                    moon_result.get('elevation'),
                    moon_result.get('confidence'),
                    moon_result.get('method')
                ))
                
                moon_id = cursor.lastrowid
                conn.commit()
                
                return moon_id
                
        except Exception as e:
            logger.error(f"Failed to store moon detection: {e}")
            raise
    
    def store_star_analysis(self, image_id: int, star_result: Dict) -> int:
        """Store star analysis results."""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                celestial_pole = star_result.get('celestial_pole')
                
                cursor.execute("""
                    INSERT INTO star_analyses (image_id, stars_detected, trails_detected,
                                             celestial_pole_x, celestial_pole_y, exposure_time,
                                             confidence, method)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    image_id,
                    star_result.get('stars_detected'),
                    star_result.get('trails_detected'),
                    celestial_pole['center'][0] if celestial_pole and celestial_pole.get('center') else None,
                    celestial_pole['center'][1] if celestial_pole and celestial_pole.get('center') else None,
                    star_result.get('exposure_time'),
                    star_result.get('confidence'),
                    star_result.get('method')
                ))
                
                star_id = cursor.lastrowid
                conn.commit()
                
                return star_id
                
        except Exception as e:
            logger.error(f"Failed to store star analysis: {e}")
            raise
    
    def store_tamper_detection(self, image_id: int, tamper_result: Dict) -> int:
        """Store tamper detection results."""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    INSERT INTO tamper_detections (image_id, is_tampered, score, error_level_score,
                                                 jpeg_ghost_score, noise_analysis_score,
                                                 compression_analysis_score, confidence, method)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    image_id,
                    tamper_result.get('is_tampered', False),
                    tamper_result.get('score'),
                    tamper_result.get('error_level_score'),
                    tamper_result.get('jpeg_ghost_score'),
                    tamper_result.get('noise_analysis_score'),
                    tamper_result.get('compression_analysis_score'),
                    tamper_result.get('confidence'),
                    tamper_result.get('method')
                ))
                
                tamper_id = cursor.lastrowid
                conn.commit()
                
                return tamper_id
                
        except Exception as e:
            logger.error(f"Failed to store tamper detection: {e}")
            raise
    
    def store_location_result(self, image_id: int, location_result: Dict) -> int:
        """Store location calculation results."""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                best_location = location_result.get('best_location')
                uncertainty = location_result.get('uncertainty')
                
                cursor.execute("""
                    INSERT INTO location_results (image_id, latitude, longitude, confidence,
                                               uncertainty_semi_major, uncertainty_semi_minor,
                                               uncertainty_orientation, num_matches, method)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    image_id,
                    best_location.get('latitude') if best_location else None,
                    best_location.get('longitude') if best_location else None,
                    best_location.get('confidence') if best_location else None,
                    uncertainty.get('semi_major') if uncertainty else None,
                    uncertainty.get('semi_minor') if uncertainty else None,
                    uncertainty.get('orientation') if uncertainty else None,
                    len(location_result.get('matches', [])),
                    location_result.get('method')
                ))
                
                location_id = cursor.lastrowid
                conn.commit()
                
                return location_id
                
        except Exception as e:
            logger.error(f"Failed to store location result: {e}")
            raise
    
    def store_confidence_map(self, image_id: int, confidence_map: Dict) -> int:
        """Store confidence map data."""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    INSERT INTO confidence_maps (image_id, heatmap_data, confidence_surface,
                                              geojson_data, num_points, confidence_range_min,
                                              confidence_range_max)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    image_id,
                    json.dumps(confidence_map.get('heatmap_data')),
                    json.dumps(confidence_map.get('confidence_surface')),
                    json.dumps(confidence_map.get('geojson')),
                    confidence_map.get('num_points'),
                    confidence_map.get('confidence_range', [0, 0])[0],
                    confidence_map.get('confidence_range', [0, 0])[1]
                ))
                
                confidence_id = cursor.lastrowid
                conn.commit()
                
                return confidence_id
                
        except Exception as e:
            logger.error(f"Failed to store confidence map: {e}")
            raise
    
    def get_image_by_hash(self, file_hash: str) -> Optional[Dict]:
        """Get image by file hash."""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM images WHERE file_hash = ?", (file_hash,))
                row = cursor.fetchone()
                
                if row:
                    return dict(row)
                return None
                
        except Exception as e:
            logger.error(f"Failed to get image by hash: {e}")
            return None
    
    def get_analysis_results(self, image_id: int) -> Dict:
        """Get all analysis results for an image."""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                # Get image info
                cursor.execute("SELECT * FROM images WHERE id = ?", (image_id,))
                image = cursor.fetchone()
                
                if not image:
                    return {}
                
                results = {'image': dict(image)}
                
                # Get EXIF data
                cursor.execute("SELECT * FROM exif_data WHERE image_id = ?", (image_id,))
                exif = cursor.fetchone()
                if exif:
                    results['exif_data'] = dict(exif)
                
                # Get sun detection
                cursor.execute("SELECT * FROM sun_detections WHERE image_id = ?", (image_id,))
                sun = cursor.fetchone()
                if sun:
                    results['sun_detection'] = dict(sun)
                
                # Get shadow analysis
                cursor.execute("SELECT * FROM shadow_analyses WHERE image_id = ?", (image_id,))
                shadow = cursor.fetchone()
                if shadow:
                    results['shadow_analysis'] = dict(shadow)
                
                # Get cloud detection
                cursor.execute("SELECT * FROM cloud_detections WHERE image_id = ?", (image_id,))
                cloud = cursor.fetchone()
                if cloud:
                    results['cloud_detection'] = dict(cloud)
                
                # Get moon detection
                cursor.execute("SELECT * FROM moon_detections WHERE image_id = ?", (image_id,))
                moon = cursor.fetchone()
                if moon:
                    results['moon_detection'] = dict(moon)
                
                # Get star analysis
                cursor.execute("SELECT * FROM star_analyses WHERE image_id = ?", (image_id,))
                star = cursor.fetchone()
                if star:
                    results['star_analysis'] = dict(star)
                
                # Get tamper detection
                cursor.execute("SELECT * FROM tamper_detections WHERE image_id = ?", (image_id,))
                tamper = cursor.fetchone()
                if tamper:
                    results['tamper_detection'] = dict(tamper)
                
                # Get location result
                cursor.execute("SELECT * FROM location_results WHERE image_id = ?", (image_id,))
                location = cursor.fetchone()
                if location:
                    results['location_result'] = dict(location)
                
                # Get confidence map
                cursor.execute("SELECT * FROM confidence_maps WHERE image_id = ?", (image_id,))
                confidence = cursor.fetchone()
                if confidence:
                    confidence_dict = dict(confidence)
                    # Parse JSON fields
                    if confidence_dict.get('heatmap_data'):
                        confidence_dict['heatmap_data'] = json.loads(confidence_dict['heatmap_data'])
                    if confidence_dict.get('confidence_surface'):
                        confidence_dict['confidence_surface'] = json.loads(confidence_dict['confidence_surface'])
                    if confidence_dict.get('geojson_data'):
                        confidence_dict['geojson_data'] = json.loads(confidence_dict['geojson_data'])
                    results['confidence_map'] = confidence_dict
                
                return results
                
        except Exception as e:
            logger.error(f"Failed to get analysis results: {e}")
            return {}
    
    def get_statistics(self) -> Dict:
        """Get database statistics."""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                stats = {}
                
                # Count images
                cursor.execute("SELECT COUNT(*) as count FROM images")
                stats['total_images'] = cursor.fetchone()['count']
                
                # Count analyses
                cursor.execute("SELECT COUNT(*) as count FROM sun_detections")
                stats['sun_detections'] = cursor.fetchone()['count']
                
                cursor.execute("SELECT COUNT(*) as count FROM shadow_analyses")
                stats['shadow_analyses'] = cursor.fetchone()['count']
                
                cursor.execute("SELECT COUNT(*) as count FROM cloud_detections")
                stats['cloud_detections'] = cursor.fetchone()['count']
                
                cursor.execute("SELECT COUNT(*) as count FROM moon_detections")
                stats['moon_detections'] = cursor.fetchone()['count']
                
                cursor.execute("SELECT COUNT(*) as count FROM star_analyses")
                stats['star_analyses'] = cursor.fetchone()['count']
                
                cursor.execute("SELECT COUNT(*) as count FROM tamper_detections")
                stats['tamper_detections'] = cursor.fetchone()['count']
                
                cursor.execute("SELECT COUNT(*) as count FROM location_results")
                stats['location_results'] = cursor.fetchone()['count']
                
                # Database size
                stats['database_size'] = self.db_path.stat().st_size
                
                return stats
                
        except Exception as e:
            logger.error(f"Failed to get statistics: {e}")
            return {}
    
    def cleanup_old_data(self, days_old: int = 30):
        """Clean up old data."""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                # Delete old data (this is a simplified cleanup)
                cursor.execute("""
                    DELETE FROM images WHERE created_at < datetime('now', '-{} days')
                """.format(days_old))
                
                conn.commit()
                logger.info(f"Cleaned up data older than {days_old} days")
                
        except Exception as e:
            logger.error(f"Failed to cleanup old data: {e}")

# Global database manager instance
_db_manager = None

def get_database_manager(db_path: str = "skypin.db") -> DatabaseManager:
    """
    Get database manager instance.
    
    Args:
        db_path: Path to database file
        
    Returns:
        DatabaseManager instance
    """
    global _db_manager
    
    if _db_manager is None:
        _db_manager = DatabaseManager(db_path)
    
    return _db_manager

def store_analysis_results(file_path: str, image_shape: Tuple[int, int, int], 
                         results: Dict) -> int:
    """
    Store complete analysis results for an image.
    
    Args:
        file_path: Path to image file
        image_shape: Image shape
        results: Analysis results dictionary
        
    Returns:
        Image ID
    """
    db = get_database_manager()
    
    # Store image
    image_id = db.store_image(file_path, image_shape)
    
    # Store EXIF data
    if results.get('exif_data'):
        db.store_exif_data(image_id, results['exif_data'])
    
    # Store sun detection
    if results.get('sun_detection'):
        db.store_sun_detection(image_id, results['sun_detection'])
    
    # Store shadow analysis
    if results.get('shadow_analysis'):
        db.store_shadow_analysis(image_id, results['shadow_analysis'])
    
    # Store cloud detection
    if results.get('cloud_detection'):
        db.store_cloud_detection(image_id, results['cloud_detection'])
    
    # Store moon detection
    if results.get('moon_detection'):
        db.store_moon_detection(image_id, results['moon_detection'])
    
    # Store star analysis
    if results.get('star_analysis'):
        db.store_star_analysis(image_id, results['star_analysis'])
    
    # Store tamper detection
    if results.get('tamper_detection'):
        db.store_tamper_detection(image_id, results['tamper_detection'])
    
    # Store location result
    if results.get('location_result'):
        db.store_location_result(image_id, results['location_result'])
    
    # Store confidence map
    if results.get('confidence_map'):
        db.store_confidence_map(image_id, results['confidence_map'])
    
    return image_id