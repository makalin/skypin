"""
Batch processing module for SkyPin
Processes multiple images in batch for efficient analysis
"""

import os
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Callable, Any
import logging
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count
import pandas as pd
from datetime import datetime, timezone
import numpy as np
from PIL import Image

# Import SkyPin modules
from modules.exif_extractor import extract_exif_data
from modules.sun_detector import detect_sun_position
from modules.shadow_analyzer import analyze_shadows
from modules.astronomy_calculator import calculate_location
from modules.confidence_mapper import create_confidence_map
from modules.tamper_detector import detect_tampering
from modules.cloud_detector import detect_clouds
from modules.moon_detector import detect_moon
from modules.star_tracker import analyze_star_trails
from modules.image_enhancer import enhance_for_analysis
from modules.utils import validate_image

logger = logging.getLogger(__name__)

class BatchProcessor:
    """Processes multiple images in batch."""
    
    def __init__(self, max_workers: int = None, use_multiprocessing: bool = False):
        """
        Initialize batch processor.
        
        Args:
            max_workers: Maximum number of worker threads/processes
            use_multiprocessing: Whether to use multiprocessing instead of threading
        """
        self.max_workers = max_workers or min(cpu_count(), 8)
        self.use_multiprocessing = use_multiprocessing
        self.results = []
        self.errors = []
        
    def process_directory(self, directory: str, output_file: str = None, 
                         file_patterns: List[str] = None) -> Dict:
        """
        Process all images in a directory.
        
        Args:
            directory: Directory containing images
            output_file: Output file for results
            file_patterns: File patterns to match (e.g., ['*.jpg', '*.png'])
            
        Returns:
            Dictionary containing processing results
        """
        try:
            if file_patterns is None:
                file_patterns = ['*.jpg', '*.jpeg', '*.png', '*.heic']
            
            # Find all image files
            image_files = []
            for pattern in file_patterns:
                image_files.extend(Path(directory).glob(pattern))
                image_files.extend(Path(directory).glob(pattern.upper()))
            
            if not image_files:
                logger.warning(f"No image files found in {directory}")
                return {'processed': 0, 'errors': 0, 'results': []}
            
            logger.info(f"Found {len(image_files)} image files to process")
            
            # Process images
            results = self.process_files([str(f) for f in image_files])
            
            # Save results if output file specified
            if output_file:
                self.save_results(results, output_file)
            
            return results
            
        except Exception as e:
            logger.error(f"Directory processing failed: {e}")
            return {'processed': 0, 'errors': 1, 'results': [], 'error': str(e)}
    
    def process_files(self, file_paths: List[str]) -> Dict:
        """
        Process multiple image files.
        
        Args:
            file_paths: List of file paths to process
            
        Returns:
            Dictionary containing processing results
        """
        try:
            start_time = time.time()
            results = []
            errors = []
            
            # Choose executor type
            executor_class = ProcessPoolExecutor if self.use_multiprocessing else ThreadPoolExecutor
            
            with executor_class(max_workers=self.max_workers) as executor:
                # Submit all tasks
                future_to_file = {
                    executor.submit(self._process_single_file, file_path): file_path
                    for file_path in file_paths
                }
                
                # Process completed tasks
                for future in as_completed(future_to_file):
                    file_path = future_to_file[future]
                    try:
                        result = future.result()
                        if result.get('error'):
                            errors.append(result)
                        else:
                            results.append(result)
                    except Exception as e:
                        logger.error(f"Error processing {file_path}: {e}")
                        errors.append({
                            'file_path': file_path,
                            'error': str(e),
                            'timestamp': datetime.now(timezone.utc).isoformat()
                        })
            
            processing_time = time.time() - start_time
            
            return {
                'processed': len(results),
                'errors': len(errors),
                'results': results,
                'errors': errors,
                'processing_time': processing_time,
                'files_per_second': len(file_paths) / processing_time if processing_time > 0 else 0
            }
            
        except Exception as e:
            logger.error(f"File processing failed: {e}")
            return {'processed': 0, 'errors': 1, 'results': [], 'error': str(e)}
    
    def _process_single_file(self, file_path: str) -> Dict:
        """Process a single image file."""
        try:
            start_time = time.time()
            
            # Load image
            image = self._load_image(file_path)
            if image is None:
                return {
                    'file_path': file_path,
                    'error': 'Failed to load image',
                    'timestamp': datetime.now(timezone.utc).isoformat()
                }
            
            # Validate image
            if not validate_image(image):
                return {
                    'file_path': file_path,
                    'error': 'Invalid image format',
                    'timestamp': datetime.now(timezone.utc).isoformat()
                }
            
            # Extract EXIF data
            exif_data = self._extract_exif_from_file(file_path)
            
            # Detect sun position
            sun_result = detect_sun_position(image)
            
            # Analyze shadows
            shadow_result = analyze_shadows(image)
            
            # Detect clouds
            cloud_result = detect_clouds(image)
            
            # Detect moon
            moon_result = detect_moon(image)
            
            # Analyze star trails
            star_result = analyze_star_trails(image)
            
            # Detect tampering
            tamper_result = detect_tampering(image)
            
            # Calculate location if possible
            location_result = None
            if sun_result['detected'] or moon_result['detected']:
                location_result = calculate_location(
                    sun_result, shadow_result, exif_data, grid_resolution=1.0
                )
            
            # Create confidence map if location found
            confidence_map = None
            if location_result and location_result.get('best_location'):
                confidence_map = create_confidence_map(location_result)
            
            processing_time = time.time() - start_time
            
            return {
                'file_path': file_path,
                'file_name': Path(file_path).name,
                'file_size': os.path.getsize(file_path),
                'image_shape': image.shape,
                'exif_data': exif_data,
                'sun_detection': sun_result,
                'shadow_analysis': shadow_result,
                'cloud_detection': cloud_result,
                'moon_detection': moon_result,
                'star_analysis': star_result,
                'tamper_detection': tamper_result,
                'location_result': location_result,
                'confidence_map': confidence_map,
                'processing_time': processing_time,
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error processing {file_path}: {e}")
            return {
                'file_path': file_path,
                'error': str(e),
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
    
    def _load_image(self, file_path: str) -> Optional[np.ndarray]:
        """Load image from file."""
        try:
            with Image.open(file_path) as img:
                # Convert to RGB if needed
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # Convert to numpy array
                image = np.array(img)
                
                return image
                
        except Exception as e:
            logger.error(f"Failed to load image {file_path}: {e}")
            return None
    
    def _extract_exif_from_file(self, file_path: str) -> Dict:
        """Extract EXIF data from file."""
        try:
            with open(file_path, 'rb') as f:
                image_bytes = f.read()
            
            return extract_exif_data(image_bytes)
            
        except Exception as e:
            logger.error(f"Failed to extract EXIF from {file_path}: {e}")
            return {}
    
    def save_results(self, results: Dict, output_file: str) -> bool:
        """Save processing results to file."""
        try:
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            if output_file.endswith('.json'):
                with open(output_file, 'w') as f:
                    json.dump(results, f, indent=2, default=str)
            elif output_file.endswith('.csv'):
                self._save_results_csv(results, output_file)
            else:
                # Default to JSON
                with open(output_file, 'w') as f:
                    json.dump(results, f, indent=2, default=str)
            
            logger.info(f"Results saved to {output_file}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save results: {e}")
            return False
    
    def _save_results_csv(self, results: Dict, output_file: str) -> None:
        """Save results as CSV."""
        try:
            # Flatten results for CSV
            flattened_results = []
            
            for result in results.get('results', []):
                flattened = {
                    'file_path': result.get('file_path', ''),
                    'file_name': result.get('file_name', ''),
                    'file_size': result.get('file_size', 0),
                    'processing_time': result.get('processing_time', 0),
                    'timestamp': result.get('timestamp', ''),
                    
                    # Sun detection
                    'sun_detected': result.get('sun_detection', {}).get('detected', False),
                    'sun_azimuth': result.get('sun_detection', {}).get('azimuth'),
                    'sun_elevation': result.get('sun_detection', {}).get('elevation'),
                    'sun_confidence': result.get('sun_detection', {}).get('confidence', 0),
                    
                    # Shadow analysis
                    'shadow_detected': result.get('shadow_analysis', {}).get('detected', False),
                    'shadow_azimuth': result.get('shadow_analysis', {}).get('azimuth'),
                    'shadow_quality': result.get('shadow_analysis', {}).get('quality', 0),
                    
                    # Cloud detection
                    'clouds_detected': result.get('cloud_detection', {}).get('clouds_detected', False),
                    'cloud_coverage': result.get('cloud_detection', {}).get('cloud_coverage', 0),
                    'weather_conditions': result.get('cloud_detection', {}).get('weather_conditions', ''),
                    
                    # Moon detection
                    'moon_detected': result.get('moon_detection', {}).get('detected', False),
                    'moon_phase': result.get('moon_detection', {}).get('phase', ''),
                    'moon_confidence': result.get('moon_detection', {}).get('confidence', 0),
                    
                    # Star analysis
                    'stars_detected': result.get('star_analysis', {}).get('stars_detected', 0),
                    'trails_detected': result.get('star_analysis', {}).get('trails_detected', 0),
                    
                    # Tamper detection
                    'is_tampered': result.get('tamper_detection', {}).get('is_tampered', False),
                    'tamper_score': result.get('tamper_detection', {}).get('score', 0),
                    
                    # Location results
                    'location_found': result.get('location_result', {}).get('best_location') is not None,
                    'latitude': result.get('location_result', {}).get('best_location', {}).get('latitude'),
                    'longitude': result.get('location_result', {}).get('best_location', {}).get('longitude'),
                    'location_confidence': result.get('location_result', {}).get('best_location', {}).get('confidence', 0),
                }
                
                flattened_results.append(flattened)
            
            # Create DataFrame and save
            df = pd.DataFrame(flattened_results)
            df.to_csv(output_file, index=False)
            
        except Exception as e:
            logger.error(f"Failed to save CSV results: {e}")
            raise
    
    def process_with_callback(self, file_paths: List[str], 
                            callback: Callable[[Dict], None]) -> Dict:
        """
        Process files with callback for progress updates.
        
        Args:
            file_paths: List of file paths to process
            callback: Callback function called for each completed file
            
        Returns:
            Dictionary containing processing results
        """
        try:
            start_time = time.time()
            results = []
            errors = []
            
            executor_class = ProcessPoolExecutor if self.use_multiprocessing else ThreadPoolExecutor
            
            with executor_class(max_workers=self.max_workers) as executor:
                future_to_file = {
                    executor.submit(self._process_single_file, file_path): file_path
                    for file_path in file_paths
                }
                
                for future in as_completed(future_to_file):
                    file_path = future_to_file[future]
                    try:
                        result = future.result()
                        if result.get('error'):
                            errors.append(result)
                        else:
                            results.append(result)
                        
                        # Call callback
                        callback(result)
                        
                    except Exception as e:
                        logger.error(f"Error processing {file_path}: {e}")
                        error_result = {
                            'file_path': file_path,
                            'error': str(e),
                            'timestamp': datetime.now(timezone.utc).isoformat()
                        }
                        errors.append(error_result)
                        callback(error_result)
            
            processing_time = time.time() - start_time
            
            return {
                'processed': len(results),
                'errors': len(errors),
                'results': results,
                'errors': errors,
                'processing_time': processing_time,
                'files_per_second': len(file_paths) / processing_time if processing_time > 0 else 0
            }
            
        except Exception as e:
            logger.error(f"Callback processing failed: {e}")
            return {'processed': 0, 'errors': 1, 'results': [], 'error': str(e)}
    
    def get_processing_statistics(self, results: Dict) -> Dict:
        """Get statistics from processing results."""
        try:
            if not results.get('results'):
                return {}
            
            results_list = results['results']
            
            # Calculate statistics
            stats = {
                'total_files': len(results_list),
                'successful_files': len([r for r in results_list if not r.get('error')]),
                'failed_files': len([r for r in results_list if r.get('error')]),
                'total_processing_time': sum(r.get('processing_time', 0) for r in results_list),
                'average_processing_time': np.mean([r.get('processing_time', 0) for r in results_list]),
                
                # Detection statistics
                'sun_detected_count': len([r for r in results_list if r.get('sun_detection', {}).get('detected')]),
                'shadow_detected_count': len([r for r in results_list if r.get('shadow_analysis', {}).get('detected')]),
                'clouds_detected_count': len([r for r in results_list if r.get('cloud_detection', {}).get('clouds_detected')]),
                'moon_detected_count': len([r for r in results_list if r.get('moon_detection', {}).get('detected')]),
                'location_found_count': len([r for r in results_list if r.get('location_result', {}).get('best_location')]),
                'tampered_count': len([r for r in results_list if r.get('tamper_detection', {}).get('is_tampered')]),
                
                # Confidence statistics
                'average_sun_confidence': np.mean([r.get('sun_detection', {}).get('confidence', 0) for r in results_list]),
                'average_location_confidence': np.mean([r.get('location_result', {}).get('best_location', {}).get('confidence', 0) for r in results_list]),
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"Statistics calculation failed: {e}")
            return {}

# Global processor instance
_processor = None

def process_directory(directory: str, output_file: str = None, 
                     file_patterns: List[str] = None, max_workers: int = None) -> Dict:
    """
    Process all images in a directory.
    
    Args:
        directory: Directory containing images
        output_file: Output file for results
        file_patterns: File patterns to match
        max_workers: Maximum number of workers
        
    Returns:
        Dictionary containing processing results
    """
    global _processor
    
    if _processor is None:
        _processor = BatchProcessor(max_workers=max_workers)
    
    return _processor.process_directory(directory, output_file, file_patterns)

def process_files(file_paths: List[str], max_workers: int = None) -> Dict:
    """
    Process multiple image files.
    
    Args:
        file_paths: List of file paths to process
        max_workers: Maximum number of workers
        
    Returns:
        Dictionary containing processing results
    """
    global _processor
    
    if _processor is None:
        _processor = BatchProcessor(max_workers=max_workers)
    
    return _processor.process_files(file_paths)

def get_processing_statistics(results: Dict) -> Dict:
    """
    Get statistics from processing results.
    
    Args:
        results: Processing results dictionary
        
    Returns:
        Dictionary containing statistics
    """
    global _processor
    
    if _processor is None:
        _processor = BatchProcessor()
    
    return _processor.get_processing_statistics(results)