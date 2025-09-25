"""
Export tools module for SkyPin
Provides various export formats for analysis results
"""

import json
import csv
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
import logging
from datetime import datetime, timezone
import pandas as pd
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import io
import base64

logger = logging.getLogger(__name__)

class ExportTools:
    """Provides various export formats for analysis results."""
    
    def __init__(self):
        """Initialize export tools."""
        self.default_font_size = 12
        self.default_image_size = (800, 600)
        
    def export_to_json(self, results: Dict, output_file: str, 
                      include_images: bool = False) -> bool:
        """
        Export results to JSON format.
        
        Args:
            results: Analysis results dictionary
            output_file: Output file path
            include_images: Whether to include image data
            
        Returns:
            True if successful, False otherwise
        """
        try:
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Prepare data for export
            export_data = self._prepare_export_data(results, include_images)
            
            # Write JSON file
            with open(output_file, 'w') as f:
                json.dump(export_data, f, indent=2, default=str)
            
            logger.info(f"Results exported to JSON: {output_file}")
            return True
            
        except Exception as e:
            logger.error(f"JSON export failed: {e}")
            return False
    
    def export_to_csv(self, results: Dict, output_file: str) -> bool:
        """
        Export results to CSV format.
        
        Args:
            results: Analysis results dictionary
            output_file: Output file path
            
        Returns:
            True if successful, False otherwise
        """
        try:
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Flatten results for CSV
            flattened_data = self._flatten_results_for_csv(results)
            
            # Create DataFrame
            df = pd.DataFrame(flattened_data)
            
            # Write CSV file
            df.to_csv(output_file, index=False)
            
            logger.info(f"Results exported to CSV: {output_file}")
            return True
            
        except Exception as e:
            logger.error(f"CSV export failed: {e}")
            return False
    
    def export_to_xml(self, results: Dict, output_file: str) -> bool:
        """
        Export results to XML format.
        
        Args:
            results: Analysis results dictionary
            output_file: Output file path
            
        Returns:
            True if successful, False otherwise
        """
        try:
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Create XML structure
            root = ET.Element("skypin_analysis")
            root.set("timestamp", datetime.now(timezone.utc).isoformat())
            root.set("version", "1.0.0")
            
            # Add results
            self._add_results_to_xml(root, results)
            
            # Write XML file
            tree = ET.ElementTree(root)
            tree.write(output_file, encoding='utf-8', xml_declaration=True)
            
            logger.info(f"Results exported to XML: {output_file}")
            return True
            
        except Exception as e:
            logger.error(f"XML export failed: {e}")
            return False
    
    def export_to_kml(self, results: Dict, output_file: str) -> bool:
        """
        Export location results to KML format for Google Earth.
        
        Args:
            results: Analysis results dictionary
            output_file: Output file path
            
        Returns:
            True if successful, False otherwise
        """
        try:
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Create KML structure
            kml_root = ET.Element("kml")
            kml_root.set("xmlns", "http://www.opengis.net/kml/2.2")
            
            document = ET.SubElement(kml_root, "Document")
            
            # Add name and description
            name = ET.SubElement(document, "name")
            name.text = "SkyPin Analysis Results"
            
            description = ET.SubElement(document, "description")
            description.text = f"Generated on {datetime.now(timezone.utc).isoformat()}"
            
            # Add location points
            self._add_locations_to_kml(document, results)
            
            # Write KML file
            tree = ET.ElementTree(kml_root)
            tree.write(output_file, encoding='utf-8', xml_declaration=True)
            
            logger.info(f"Results exported to KML: {output_file}")
            return True
            
        except Exception as e:
            logger.error(f"KML export failed: {e}")
            return False
    
    def export_to_geojson(self, results: Dict, output_file: str) -> bool:
        """
        Export results to GeoJSON format.
        
        Args:
            results: Analysis results dictionary
            output_file: Output file path
            
        Returns:
            True if successful, False otherwise
        """
        try:
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Create GeoJSON structure
            geojson = {
                "type": "FeatureCollection",
                "features": [],
                "properties": {
                    "generated": datetime.now(timezone.utc).isoformat(),
                    "generator": "SkyPin",
                    "version": "1.0.0"
                }
            }
            
            # Add location features
            self._add_locations_to_geojson(geojson, results)
            
            # Write GeoJSON file
            with open(output_file, 'w') as f:
                json.dump(geojson, f, indent=2)
            
            logger.info(f"Results exported to GeoJSON: {output_file}")
            return True
            
        except Exception as e:
            logger.error(f"GeoJSON export failed: {e}")
            return False
    
    def export_to_html_report(self, results: Dict, output_file: str, 
                            template_file: str = None) -> bool:
        """
        Export results to HTML report.
        
        Args:
            results: Analysis results dictionary
            output_file: Output file path
            template_file: Custom HTML template file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Generate HTML content
            html_content = self._generate_html_report(results, template_file)
            
            # Write HTML file
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            logger.info(f"HTML report exported: {output_file}")
            return True
            
        except Exception as e:
            logger.error(f"HTML export failed: {e}")
            return False
    
    def export_to_pdf_report(self, results: Dict, output_file: str) -> bool:
        """
        Export results to PDF report.
        
        Args:
            results: Analysis results dictionary
            output_file: Output file path
            
        Returns:
            True if successful, False otherwise
        """
        try:
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Generate PDF content
            pdf_content = self._generate_pdf_report(results)
            
            # Write PDF file
            with open(output_file, 'wb') as f:
                f.write(pdf_content)
            
            logger.info(f"PDF report exported: {output_file}")
            return True
            
        except Exception as e:
            logger.error(f"PDF export failed: {e}")
            return False
    
    def export_visualization(self, results: Dict, output_file: str, 
                           visualization_type: str = 'summary') -> bool:
        """
        Export visualization image.
        
        Args:
            results: Analysis results dictionary
            output_file: Output file path
            visualization_type: Type of visualization ('summary', 'confidence', 'analysis')
            
        Returns:
            True if successful, False otherwise
        """
        try:
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Generate visualization
            if visualization_type == 'summary':
                image = self._create_summary_visualization(results)
            elif visualization_type == 'confidence':
                image = self._create_confidence_visualization(results)
            elif visualization_type == 'analysis':
                image = self._create_analysis_visualization(results)
            else:
                image = self._create_summary_visualization(results)
            
            # Save image
            image.save(output_file)
            
            logger.info(f"Visualization exported: {output_file}")
            return True
            
        except Exception as e:
            logger.error(f"Visualization export failed: {e}")
            return False
    
    def _prepare_export_data(self, results: Dict, include_images: bool = False) -> Dict:
        """Prepare data for export."""
        try:
            export_data = {
                'metadata': {
                    'exported_at': datetime.now(timezone.utc).isoformat(),
                    'exporter': 'SkyPin',
                    'version': '1.0.0'
                },
                'results': results
            }
            
            if include_images and 'image_data' in results:
                # Convert image to base64
                image_data = results['image_data']
                if isinstance(image_data, np.ndarray):
                    image = Image.fromarray(image_data)
                    buffer = io.BytesIO()
                    image.save(buffer, format='PNG')
                    export_data['image_base64'] = base64.b64encode(buffer.getvalue()).decode()
            
            return export_data
            
        except Exception as e:
            logger.error(f"Data preparation failed: {e}")
            return results
    
    def _flatten_results_for_csv(self, results: Dict) -> List[Dict]:
        """Flatten results for CSV export."""
        try:
            flattened = []
            
            # Extract basic info
            basic_info = {
                'file_path': results.get('file_path', ''),
                'file_name': results.get('file_name', ''),
                'timestamp': results.get('timestamp', ''),
                'processing_time': results.get('processing_time', 0)
            }
            
            # Extract sun detection
            sun_data = results.get('sun_detection', {})
            sun_info = {
                'sun_detected': sun_data.get('detected', False),
                'sun_azimuth': sun_data.get('azimuth'),
                'sun_elevation': sun_data.get('elevation'),
                'sun_confidence': sun_data.get('confidence', 0)
            }
            
            # Extract shadow analysis
            shadow_data = results.get('shadow_analysis', {})
            shadow_info = {
                'shadow_detected': shadow_data.get('detected', False),
                'shadow_azimuth': shadow_data.get('azimuth'),
                'shadow_quality': shadow_data.get('quality', 0)
            }
            
            # Extract cloud detection
            cloud_data = results.get('cloud_detection', {})
            cloud_info = {
                'clouds_detected': cloud_data.get('clouds_detected', False),
                'cloud_coverage': cloud_data.get('cloud_coverage', 0),
                'weather_conditions': cloud_data.get('weather_conditions', '')
            }
            
            # Extract moon detection
            moon_data = results.get('moon_detection', {})
            moon_info = {
                'moon_detected': moon_data.get('detected', False),
                'moon_phase': moon_data.get('phase', ''),
                'moon_confidence': moon_data.get('confidence', 0)
            }
            
            # Extract star analysis
            star_data = results.get('star_analysis', {})
            star_info = {
                'stars_detected': star_data.get('stars_detected', 0),
                'trails_detected': star_data.get('trails_detected', 0)
            }
            
            # Extract tamper detection
            tamper_data = results.get('tamper_detection', {})
            tamper_info = {
                'is_tampered': tamper_data.get('is_tampered', False),
                'tamper_score': tamper_data.get('score', 0)
            }
            
            # Extract location results
            location_data = results.get('location_result', {})
            best_location = location_data.get('best_location', {})
            location_info = {
                'location_found': best_location is not None,
                'latitude': best_location.get('latitude'),
                'longitude': best_location.get('longitude'),
                'location_confidence': best_location.get('confidence', 0)
            }
            
            # Combine all info
            combined_info = {
                **basic_info,
                **sun_info,
                **shadow_info,
                **cloud_info,
                **moon_info,
                **star_info,
                **tamper_info,
                **location_info
            }
            
            flattened.append(combined_info)
            
            return flattened
            
        except Exception as e:
            logger.error(f"CSV flattening failed: {e}")
            return []
    
    def _add_results_to_xml(self, parent: ET.Element, results: Dict):
        """Add results to XML structure."""
        try:
            for key, value in results.items():
                if isinstance(value, dict):
                    element = ET.SubElement(parent, key)
                    self._add_results_to_xml(element, value)
                elif isinstance(value, list):
                    element = ET.SubElement(parent, key)
                    for item in value:
                        if isinstance(item, dict):
                            item_element = ET.SubElement(element, "item")
                            self._add_results_to_xml(item_element, item)
                        else:
                            item_element = ET.SubElement(element, "item")
                            item_element.text = str(item)
                else:
                    element = ET.SubElement(parent, key)
                    element.text = str(value)
                    
        except Exception as e:
            logger.error(f"XML addition failed: {e}")
    
    def _add_locations_to_kml(self, document: ET.Element, results: Dict):
        """Add location data to KML."""
        try:
            location_data = results.get('location_result', {})
            best_location = location_data.get('best_location', {})
            
            if best_location:
                latitude = best_location.get('latitude')
                longitude = best_location.get('longitude')
                confidence = best_location.get('confidence', 0)
                
                if latitude is not None and longitude is not None:
                    # Create placemark
                    placemark = ET.SubElement(document, "Placemark")
                    
                    name = ET.SubElement(placemark, "name")
                    name.text = f"SkyPin Location (Confidence: {confidence:.1%})"
                    
                    description = ET.SubElement(placemark, "description")
                    description.text = f"Latitude: {latitude:.6f}, Longitude: {longitude:.6f}"
                    
                    point = ET.SubElement(placemark, "Point")
                    coordinates = ET.SubElement(point, "coordinates")
                    coordinates.text = f"{longitude},{latitude},0"
                    
        except Exception as e:
            logger.error(f"KML location addition failed: {e}")
    
    def _add_locations_to_geojson(self, geojson: Dict, results: Dict):
        """Add location data to GeoJSON."""
        try:
            location_data = results.get('location_result', {})
            best_location = location_data.get('best_location', {})
            
            if best_location:
                latitude = best_location.get('latitude')
                longitude = best_location.get('longitude')
                confidence = best_location.get('confidence', 0)
                
                if latitude is not None and longitude is not None:
                    feature = {
                        "type": "Feature",
                        "geometry": {
                            "type": "Point",
                            "coordinates": [longitude, latitude]
                        },
                        "properties": {
                            "confidence": confidence,
                            "latitude": latitude,
                            "longitude": longitude,
                            "source": "SkyPin"
                        }
                    }
                    
                    geojson["features"].append(feature)
                    
        except Exception as e:
            logger.error(f"GeoJSON location addition failed: {e}")
    
    def _generate_html_report(self, results: Dict, template_file: str = None) -> str:
        """Generate HTML report."""
        try:
            if template_file and Path(template_file).exists():
                with open(template_file, 'r') as f:
                    template = f.read()
            else:
                template = self._get_default_html_template()
            
            # Replace placeholders
            html = template.replace('{{TITLE}}', 'SkyPin Analysis Report')
            html = html.replace('{{TIMESTAMP}}', datetime.now(timezone.utc).isoformat())
            html = html.replace('{{RESULTS}}', json.dumps(results, indent=2, default=str))
            
            return html
            
        except Exception as e:
            logger.error(f"HTML generation failed: {e}")
            return f"<html><body><h1>Error generating report: {e}</h1></body></html>"
    
    def _get_default_html_template(self) -> str:
        """Get default HTML template."""
        return """
<!DOCTYPE html>
<html>
<head>
    <title>{{TITLE}}</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        .header { background-color: #f0f0f0; padding: 20px; border-radius: 5px; }
        .section { margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }
        .success { color: green; }
        .warning { color: orange; }
        .error { color: red; }
        pre { background-color: #f5f5f5; padding: 10px; border-radius: 3px; overflow-x: auto; }
    </style>
</head>
<body>
    <div class="header">
        <h1>{{TITLE}}</h1>
        <p>Generated on: {{TIMESTAMP}}</p>
    </div>
    
    <div class="section">
        <h2>Analysis Results</h2>
        <pre>{{RESULTS}}</pre>
    </div>
</body>
</html>
        """
    
    def _generate_pdf_report(self, results: Dict) -> bytes:
        """Generate PDF report."""
        try:
            # This is a simplified PDF generation
            # In practice, you'd use a library like reportlab
            html_content = self._generate_html_report(results)
            
            # For now, return HTML as bytes (would be converted to PDF)
            return html_content.encode('utf-8')
            
        except Exception as e:
            logger.error(f"PDF generation failed: {e}")
            return b"Error generating PDF report"
    
    def _create_summary_visualization(self, results: Dict) -> Image.Image:
        """Create summary visualization."""
        try:
            # Create image
            img = Image.new('RGB', self.default_image_size, 'white')
            draw = ImageDraw.Draw(img)
            
            # Draw title
            title = "SkyPin Analysis Summary"
            draw.text((50, 50), title, fill='black')
            
            # Draw results
            y_offset = 100
            for key, value in results.items():
                if isinstance(value, dict):
                    text = f"{key}: {json.dumps(value, indent=2)}"
                else:
                    text = f"{key}: {value}"
                
                # Wrap text
                lines = text.split('\n')
                for line in lines:
                    draw.text((50, y_offset), line, fill='black')
                    y_offset += 20
            
            return img
            
        except Exception as e:
            logger.error(f"Summary visualization failed: {e}")
            return Image.new('RGB', self.default_image_size, 'white')
    
    def _create_confidence_visualization(self, results: Dict) -> Image.Image:
        """Create confidence visualization."""
        try:
            # Create image
            img = Image.new('RGB', self.default_image_size, 'white')
            draw = ImageDraw.Draw(img)
            
            # Draw confidence bars
            y_offset = 50
            confidences = [
                ('Sun Detection', results.get('sun_detection', {}).get('confidence', 0)),
                ('Shadow Analysis', results.get('shadow_analysis', {}).get('quality', 0)),
                ('Location Result', results.get('location_result', {}).get('best_location', {}).get('confidence', 0))
            ]
            
            for name, confidence in confidences:
                # Draw bar
                bar_width = int(confidence * 300)
                draw.rectangle([50, y_offset, 50 + bar_width, y_offset + 20], fill='green')
                draw.rectangle([50, y_offset, 350, y_offset + 20], outline='black')
                
                # Draw text
                draw.text((360, y_offset), f"{name}: {confidence:.2f}", fill='black')
                y_offset += 40
            
            return img
            
        except Exception as e:
            logger.error(f"Confidence visualization failed: {e}")
            return Image.new('RGB', self.default_image_size, 'white')
    
    def _create_analysis_visualization(self, results: Dict) -> Image.Image:
        """Create analysis visualization."""
        try:
            # Create image
            img = Image.new('RGB', self.default_image_size, 'white')
            draw = ImageDraw.Draw(img)
            
            # Draw analysis results
            y_offset = 50
            
            # Sun detection
            sun_data = results.get('sun_detection', {})
            if sun_data.get('detected'):
                draw.text((50, y_offset), f"Sun detected at azimuth {sun_data.get('azimuth', 0):.1f}°", fill='green')
            else:
                draw.text((50, y_offset), "No sun detected", fill='red')
            y_offset += 30
            
            # Shadow analysis
            shadow_data = results.get('shadow_analysis', {})
            if shadow_data.get('detected'):
                draw.text((50, y_offset), f"Shadows detected with quality {shadow_data.get('quality', 0):.2f}", fill='green')
            else:
                draw.text((50, y_offset), "No shadows detected", fill='red')
            y_offset += 30
            
            # Location result
            location_data = results.get('location_result', {})
            best_location = location_data.get('best_location', {})
            if best_location:
                lat = best_location.get('latitude', 0)
                lon = best_location.get('longitude', 0)
                conf = best_location.get('confidence', 0)
                draw.text((50, y_offset), f"Location: {lat:.6f}, {lon:.6f} (Confidence: {conf:.2f})", fill='green')
            else:
                draw.text((50, y_offset), "No location determined", fill='red')
            
            return img
            
        except Exception as e:
            logger.error(f"Analysis visualization failed: {e}")
            return Image.new('RGB', self.default_image_size, 'white')

# Global export tools instance
_export_tools = None

def export_to_json(results: Dict, output_file: str, include_images: bool = False) -> bool:
    """
    Export results to JSON format.
    
    Args:
        results: Analysis results dictionary
        output_file: Output file path
        include_images: Whether to include image data
        
    Returns:
        True if successful, False otherwise
    """
    global _export_tools
    
    if _export_tools is None:
        _export_tools = ExportTools()
    
    return _export_tools.export_to_json(results, output_file, include_images)

def export_to_csv(results: Dict, output_file: str) -> bool:
    """
    Export results to CSV format.
    
    Args:
        results: Analysis results dictionary
        output_file: Output file path
        
    Returns:
        True if successful, False otherwise
    """
    global _export_tools
    
    if _export_tools is None:
        _export_tools = ExportTools()
    
    return _export_tools.export_to_csv(results, output_file)

def export_to_kml(results: Dict, output_file: str) -> bool:
    """
    Export location results to KML format.
    
    Args:
        results: Analysis results dictionary
        output_file: Output file path
        
    Returns:
        True if successful, False otherwise
    """
    global _export_tools
    
    if _export_tools is None:
        _export_tools = ExportTools()
    
    return _export_tools.export_to_kml(results, output_file)

def export_to_html_report(results: Dict, output_file: str, template_file: str = None) -> bool:
    """
    Export results to HTML report.
    
    Args:
        results: Analysis results dictionary
        output_file: Output file path
        template_file: Custom HTML template file
        
    Returns:
        True if successful, False otherwise
    """
    global _export_tools
    
    if _export_tools is None:
        _export_tools = ExportTools()
    
    return _export_tools.export_to_html_report(results, output_file, template_file)