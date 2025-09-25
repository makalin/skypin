#!/usr/bin/env python3
"""
Command Line Interface for SkyPin
Provides command-line access to all SkyPin functionality
"""

import argparse
import sys
import json
import os
from pathlib import Path
from typing import Dict, List, Optional
import logging

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
from modules.batch_processor import process_directory, process_files, get_processing_statistics
from modules.database_manager import get_database_manager, store_analysis_results
from modules.export_tools import export_to_json, export_to_csv, export_to_kml, export_to_html_report
from modules.validation_tools import validate_analysis_results, validate_image_quality
from modules.performance_tools import start_performance_monitoring, stop_performance_monitoring, get_performance_summary
from modules.utils import validate_image
from config import get_config

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def analyze_single_image(image_path: str, output_file: str = None, 
                        confidence_threshold: float = 0.7, 
                        grid_resolution: float = 1.0,
                        enable_tamper_detection: bool = True,
                        timezone_override: str = None) -> Dict:
    """Analyze a single image."""
    try:
        from PIL import Image
        import numpy as np
        
        # Load image
        image = Image.open(image_path)
        image_array = np.array(image)
        
        # Validate image
        if not validate_image(image_array):
            raise ValueError("Invalid image format")
        
        logger.info(f"Analyzing image: {image_path}")
        
        # Extract EXIF data
        with open(image_path, 'rb') as f:
            image_bytes = f.read()
        exif_data = extract_exif_data(image_bytes)
        
        # Detect sun position
        sun_result = detect_sun_position(image_array)
        
        # Analyze shadows
        shadow_result = analyze_shadows(image_array)
        
        # Detect clouds
        cloud_result = detect_clouds(image_array)
        
        # Detect moon
        moon_result = detect_moon(image_array)
        
        # Analyze star trails
        star_result = analyze_star_trails(image_array)
        
        # Detect tampering
        tamper_result = None
        if enable_tamper_detection:
            tamper_result = detect_tampering(image_array)
        
        # Calculate location
        location_result = calculate_location(
            sun_result, shadow_result, exif_data, 
            grid_resolution, timezone_override
        )
        
        # Create confidence map
        confidence_map = None
        if location_result.get('best_location'):
            confidence_map = create_confidence_map(location_result, confidence_threshold)
        
        # Compile results
        results = {
            'file_path': image_path,
            'file_name': Path(image_path).name,
            'exif_data': exif_data,
            'sun_detection': sun_result,
            'shadow_analysis': shadow_result,
            'cloud_detection': cloud_result,
            'moon_detection': moon_result,
            'star_analysis': star_result,
            'tamper_detection': tamper_result,
            'location_result': location_result,
            'confidence_map': confidence_map
        }
        
        # Save results if output file specified
        if output_file:
            export_to_json(results, output_file)
            logger.info(f"Results saved to: {output_file}")
        
        return results
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        return {'error': str(e)}

def batch_analyze(input_path: str, output_file: str = None, 
                 file_patterns: List[str] = None, 
                 max_workers: int = None) -> Dict:
    """Analyze multiple images in batch."""
    try:
        input_path = Path(input_path)
        
        if input_path.is_file():
            # Single file
            results = process_files([str(input_path)])
        elif input_path.is_dir():
            # Directory
            results = process_directory(str(input_path), output_file, file_patterns)
        else:
            raise ValueError(f"Invalid input path: {input_path}")
        
        # Save results if output file specified
        if output_file and not input_path.is_dir():
            export_to_json(results, output_file)
            logger.info(f"Results saved to: {output_file}")
        
        return results
        
    except Exception as e:
        logger.error(f"Batch analysis failed: {e}")
        return {'error': str(e)}

def validate_results(results: Dict) -> Dict:
    """Validate analysis results."""
    try:
        validation = validate_analysis_results(results)
        return validation
    except Exception as e:
        logger.error(f"Validation failed: {e}")
        return {'error': str(e)}

def export_results(results: Dict, output_file: str, format: str = 'json') -> bool:
    """Export results to various formats."""
    try:
        if format.lower() == 'json':
            return export_to_json(results, output_file)
        elif format.lower() == 'csv':
            return export_to_csv(results, output_file)
        elif format.lower() == 'kml':
            return export_to_kml(results, output_file)
        elif format.lower() == 'html':
            return export_to_html_report(results, output_file)
        else:
            raise ValueError(f"Unsupported format: {format}")
    except Exception as e:
        logger.error(f"Export failed: {e}")
        return False

def setup_database(db_path: str = "skypin.db") -> bool:
    """Setup database."""
    try:
        db = get_database_manager(db_path)
        logger.info(f"Database setup complete: {db_path}")
        return True
    except Exception as e:
        logger.error(f"Database setup failed: {e}")
        return False

def start_api_server(host: str = "0.0.0.0", port: int = 5000, debug: bool = False):
    """Start API server."""
    try:
        from api_server import app
        logger.info(f"Starting API server on {host}:{port}")
        app.run(host=host, port=port, debug=debug)
    except Exception as e:
        logger.error(f"API server failed: {e}")

def main():
    """Main CLI function."""
    parser = argparse.ArgumentParser(
        description="SkyPin - Celestial GPS from a single photo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze a single image
  python cli.py analyze image.jpg --output results.json
  
  # Batch analyze a directory
  python cli.py batch /path/to/images --output batch_results.json
  
  # Validate results
  python cli.py validate results.json
  
  # Export to different formats
  python cli.py export results.json --format csv --output results.csv
  
  # Start API server
  python cli.py api --port 5000
  
  # Setup database
  python cli.py setup-db --db-path skypin.db
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Analyze command
    analyze_parser = subparsers.add_parser('analyze', help='Analyze a single image')
    analyze_parser.add_argument('image_path', help='Path to image file')
    analyze_parser.add_argument('--output', '-o', help='Output file path')
    analyze_parser.add_argument('--confidence-threshold', type=float, default=0.7,
                               help='Confidence threshold (default: 0.7)')
    analyze_parser.add_argument('--grid-resolution', type=float, default=1.0,
                               help='Grid resolution in degrees (default: 1.0)')
    analyze_parser.add_argument('--no-tamper-detection', action='store_true',
                               help='Disable tamper detection')
    analyze_parser.add_argument('--timezone-override', help='Override timezone')
    
    # Batch command
    batch_parser = subparsers.add_parser('batch', help='Batch analyze images')
    batch_parser.add_argument('input_path', help='Path to image file or directory')
    batch_parser.add_argument('--output', '-o', help='Output file path')
    batch_parser.add_argument('--file-patterns', nargs='+', 
                             default=['*.jpg', '*.jpeg', '*.png', '*.heic'],
                             help='File patterns to match')
    batch_parser.add_argument('--max-workers', type=int, help='Maximum number of workers')
    
    # Validate command
    validate_parser = subparsers.add_parser('validate', help='Validate analysis results')
    validate_parser.add_argument('results_file', help='Path to results file')
    validate_parser.add_argument('--output', '-o', help='Output file path')
    
    # Export command
    export_parser = subparsers.add_parser('export', help='Export results to different formats')
    export_parser.add_argument('results_file', help='Path to results file')
    export_parser.add_argument('--format', '-f', choices=['json', 'csv', 'kml', 'html'],
                              default='json', help='Export format')
    export_parser.add_argument('--output', '-o', required=True, help='Output file path')
    
    # Setup database command
    setup_db_parser = subparsers.add_parser('setup-db', help='Setup database')
    setup_db_parser.add_argument('--db-path', default='skypin.db', help='Database file path')
    
    # API server command
    api_parser = subparsers.add_parser('api', help='Start API server')
    api_parser.add_argument('--host', default='0.0.0.0', help='Host address')
    api_parser.add_argument('--port', type=int, default=5000, help='Port number')
    api_parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    
    # Performance monitoring command
    perf_parser = subparsers.add_parser('performance', help='Performance monitoring')
    perf_parser.add_argument('--start', action='store_true', help='Start monitoring')
    perf_parser.add_argument('--stop', action='store_true', help='Stop monitoring')
    perf_parser.add_argument('--summary', action='store_true', help='Show performance summary')
    
    # Config command
    config_parser = subparsers.add_parser('config', help='Show configuration')
    config_parser.add_argument('--section', help='Show specific configuration section')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    try:
        if args.command == 'analyze':
            results = analyze_single_image(
                args.image_path,
                args.output,
                args.confidence_threshold,
                args.grid_resolution,
                not args.no_tamper_detection,
                args.timezone_override
            )
            
            if 'error' in results:
                logger.error(f"Analysis failed: {results['error']}")
                sys.exit(1)
            else:
                logger.info("Analysis completed successfully")
                
        elif args.command == 'batch':
            results = batch_analyze(
                args.input_path,
                args.output,
                args.file_patterns,
                args.max_workers
            )
            
            if 'error' in results:
                logger.error(f"Batch analysis failed: {results['error']}")
                sys.exit(1)
            else:
                logger.info(f"Batch analysis completed: {results.get('processed', 0)} files processed")
                
        elif args.command == 'validate':
            with open(args.results_file, 'r') as f:
                results = json.load(f)
            
            validation = validate_results(results)
            
            if args.output:
                with open(args.output, 'w') as f:
                    json.dump(validation, f, indent=2)
                logger.info(f"Validation results saved to: {args.output}")
            else:
                print(json.dumps(validation, indent=2))
                
        elif args.command == 'export':
            with open(args.results_file, 'r') as f:
                results = json.load(f)
            
            success = export_results(results, args.output, args.format)
            if success:
                logger.info(f"Results exported to: {args.output}")
            else:
                logger.error("Export failed")
                sys.exit(1)
                
        elif args.command == 'setup-db':
            success = setup_database(args.db_path)
            if not success:
                sys.exit(1)
                
        elif args.command == 'api':
            start_api_server(args.host, args.port, args.debug)
            
        elif args.command == 'performance':
            if args.start:
                start_performance_monitoring()
                logger.info("Performance monitoring started")
            elif args.stop:
                stop_performance_monitoring()
                logger.info("Performance monitoring stopped")
            elif args.summary:
                summary = get_performance_summary()
                print(json.dumps(summary, indent=2))
            else:
                logger.error("Please specify --start, --stop, or --summary")
                sys.exit(1)
                
        elif args.command == 'config':
            config = get_config(args.section)
            print(json.dumps(config, indent=2))
            
    except KeyboardInterrupt:
        logger.info("Operation cancelled by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Command failed: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()