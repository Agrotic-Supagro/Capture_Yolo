import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.segmentation import random_walker
import os
import argparse
from pathlib import Path
import json
from datetime import datetime, timedelta
import pandas as pd
import re
from matplotlib.dates import DateFormatter
import seaborn as sns
from skimage import color

class MultiGrapeTimeSeriesMonitor:
    def __init__(self, beta=130, mode='bf'):
        """
        Initialize multi-grape growth monitoring with Random Walker segmentation
        
        Args:
            beta (int): Random walker beta parameter (edge weighting)
            mode (str): Solver mode ('bf' or 'cg_mg')
        """
        self.beta = beta
        self.mode = mode
        
        # Supported image extensions
        self.supported_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        
        # Multi-grape time series data
        self.multi_grape_data = []
        
    def extract_timestamp_from_filename(self, filename):
        """
        Extract timestamp from filename using various common patterns
        
        Supported patterns:
        - YYYY-MM-DD_HH-MM-SS
        - YYYY_MM_DD_HH_MM_SS
        - YYYYMMDD_HHMMSS
        - IMG_YYYYMMDD_HHMMSS
        - grape_YYYY-MM-DD_HH-MM
        - Or sequential numbering: 001, 002, etc.
        """
        filename = Path(filename).stem.lower()
        
        # Try to extract date patterns
        patterns = [
            r'(\d{4}-\d{2}-\d{2})',  # YYYY-MM-DD
            r'(\d{4}_\d{2}_\d{2})',  # YYYY_MM_DD
            r'(\d{8})',              # YYYYMMDD
        ]
        
        for pattern in patterns:
            match = re.search(pattern, filename)
            if match:
                date_str = match.group(1)
                try:
                    if '-' in date_str:
                        return datetime.strptime(date_str, '%Y-%m-%d')
                    elif '_' in date_str:
                        return datetime.strptime(date_str, '%Y_%m_%d')
                    elif len(date_str) == 8:
                        return datetime.strptime(date_str, '%Y%m%d')
                except ValueError:
                    continue
        return None

    def load_image(self, image_path):
        """Load and prepare image for segmentation"""
        try:
            image = cv2.imread(str(image_path))
            if image is None:
                raise ValueError(f"Could not load image: {image_path}")
            
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            return image, image_rgb, image_gray
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            return None, None, None
    
    def create_automatic_markers(self, image_gray):
        """
        Create automatic markers for random walker segmentation
        Optimized for grape detection
        """
        h, w = image_gray.shape
        markers = np.zeros_like(image_gray, dtype=np.int32)
        
        # Adaptive edge width based on image size
        edge_width = max(5, min(h, w) // 100)
        
        # Background markers (edges of image)
        markers[0:edge_width, :] = 1  # Top edge
        markers[-edge_width:, :] = 1  # Bottom edge
        markers[:, 0:edge_width] = 1  # Left edge
        markers[:, -edge_width:] = 1  # Right edge
        
        # Center-based foreground markers for grape clusters
        center_h, center_w = h // 2, w // 2
        
        # Multiple foreground regions to capture grape clusters
        regions = [
            (center_h - h//6, center_w - w//6, center_h - h//8, center_w - w//8),
            (center_h + h//8, center_w - w//6, center_h + h//6, center_w - w//8),
            (center_h - h//6, center_w + w//8, center_h - h//8, center_w + w//6),
            (center_h + h//8, center_w + w//8, center_h + h//6, center_w + w//6),
            (center_h - h//12, center_w - w//12, center_h + h//12, center_w + w//12)
        ]
        
        for i, (y1, x1, y2, x2) in enumerate(regions):
            y1, y2 = max(0, y1), min(h, y2)
            x1, x2 = max(0, x1), min(w, x2)
            
            if y2 > y1 and x2 > x1:
                markers[y1:y2, x1:x2] = 2
        
        return markers
    
    def calculate_days_from_start_per_grape(self, all_data):
        """Calculate days from start for each grape individually"""
        # Group by grape_id
        grape_groups = {}
        for data in all_data:
            grape_id = data['grape_id']
            if grape_id not in grape_groups:
                grape_groups[grape_id] = []
            grape_groups[grape_id].append(data)
        
        # Calculate days from start for each grape
        for grape_id, grape_data in grape_groups.items():
            grape_data.sort(key=lambda x: x['timestamp'])
            if grape_data:
                start_time = grape_data[0]['timestamp']
                for data in grape_data:
                    data['days_from_start'] = (data['timestamp'] - start_time).total_seconds() / 86400
    
    def add_date_axis_weekly(self, ax, days_data, timestamps_data):
        """Add top x-axis with dates every 7 days in dd/mm format"""
        ax2 = ax.twiny()
        ax2.set_xlim(ax.get_xlim())
    
        # Convert to lists for easier handling
        days_list = list(days_data)
        timestamps_list = list(timestamps_data)
    
        # Find positions for weekly ticks (every 7 days)
        min_day = min(days_list)
        max_day = max(days_list)
    
        # Create weekly tick positions starting from first day
        weekly_positions = []
        weekly_dates = []
    
        for target_day in range(int(min_day), int(max_day) + 1, 7):
            # Find closest available day within tolerance
            closest_day = None
            min_diff = float('inf')
            
            for day in days_list:
                diff = abs(day - target_day)
                if diff < min_diff:
                    min_diff = diff
                    closest_day = day
            
            # Only add if within reasonable tolerance (3 days)
            if closest_day is not None and min_diff <= 3:
                if closest_day not in weekly_positions:  # Avoid duplicates
                    weekly_positions.append(closest_day)
                    idx = days_list.index(closest_day)
                    weekly_dates.append(timestamps_list[idx].strftime('%d/%m'))
    
        # If we still don't have any positions, use available data points
        if not weekly_positions:
            # Use every available data point, but limit to reasonable number
            available_days = sorted(days_list)
            step = max(1, len(available_days) // 5)  # Show max 5 dates
            for i in range(0, len(available_days), step):
                day = available_days[i]
                weekly_positions.append(day)
                idx = days_list.index(day)
                weekly_dates.append(timestamps_list[idx].strftime('%d/%m'))
    
        if weekly_positions:
            ax2.set_xticks(weekly_positions)
            ax2.set_xticklabels(weekly_dates, rotation=45, ha='right')
    
        ax2.set_xlabel('Date (dd/mm)', fontsize=10)
        return ax2
    
    def create_color_based_markers(self, image_rgb, image_gray):
        """Create markers based on color characteristics - works for green and dark grapes"""
        h, w = image_gray.shape
        markers = np.zeros_like(image_gray, dtype=np.int32)
        
        # Background markers (edges)
        edge_width = max(5, min(h, w) // 100)
        markers[0:edge_width, :] = 1
        markers[-edge_width:, :] = 1
        markers[:, 0:edge_width] = 1
        markers[:, -edge_width:] = 1
        
        # Convert to different color spaces for better grape detection
        hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
        lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)
        
        # Create multiple masks for different grape colors
        grape_mask = np.zeros((h, w), dtype=np.uint8)
        
        # Green grapes (adjust ranges based on your specific images)
        green_lower = np.array([35, 40, 40])   # HSV
        green_upper = np.array([85, 255, 255])
        green_mask = cv2.inRange(hsv, green_lower, green_upper)
        
        # Dark grapes (purple/red/black)
        dark_lower1 = np.array([0, 30, 30])    # Red range
        dark_upper1 = np.array([15, 255, 255])
        dark_lower2 = np.array([160, 30, 30])  # Purple range
        dark_upper2 = np.array([180, 255, 255])
        dark_mask1 = cv2.inRange(hsv, dark_lower1, dark_upper1)
        dark_mask2 = cv2.inRange(hsv, dark_lower2, dark_upper2)
        dark_mask = cv2.bitwise_or(dark_mask1, dark_mask2)
        
        # Combine masks
        grape_mask = cv2.bitwise_or(green_mask, dark_mask)
        
        # Clean up noise
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        grape_mask = cv2.morphologyEx(grape_mask, cv2.MORPH_OPEN, kernel)
        grape_mask = cv2.morphologyEx(grape_mask, cv2.MORPH_CLOSE, kernel)
        
        # Find connected components
        num_labels, labels_im = cv2.connectedComponents(grape_mask)
        
        # Place foreground markers in grape regions
        for label in range(1, min(num_labels, 8)):  # Up to 7 grape regions
            region_mask = (labels_im == label)
            if np.sum(region_mask) > 150:  # Minimum size threshold
                # Erode the region to place markers more in the center
                eroded = cv2.erode(region_mask.astype(np.uint8), 
                                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
                if np.sum(eroded) > 20:
                    markers[eroded > 0] = 2
        
        return markers
    
    def apply_random_walker(self, image_gray, image_rgb=None):
        """Apply random walker segmentation"""
        try:
            #markers = self.create_color_based_markers(image_rgb, image_gray)
            markers = self.create_automatic_markers(image_gray)
            labels = random_walker(image_gray, markers, beta=self.beta, mode=self.mode)
            mask = (labels == 2).astype(np.uint8) * 255
            return mask, markers
        except Exception as e:
            print(f"Error in random walker segmentation: {e}")
            return None, None

    def post_process_mask(self, mask, min_area=100):
        """Post-process the segmentation mask to remove noise"""
        try:
            # Find connected components
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
            
            # Create cleaned mask
            processed_mask = np.zeros_like(mask)
            
            # Keep only components larger than min_area
            for i in range(1, num_labels):
                if stats[i, cv2.CC_STAT_AREA] >= min_area:
                    processed_mask[labels == i] = 255
            
            # Morphological operations
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            processed_mask = cv2.morphologyEx(processed_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
            processed_mask = cv2.morphologyEx(processed_mask, cv2.MORPH_OPEN, kernel, iterations=1)
            
            return processed_mask
        except Exception as e:
            print(f"Error in post-processing: {e}")
            return mask
        
    def create_individual_grape_charts(self, df, output_path):
        """Create individual enhanced color charts for each grape (same as combined analysis)"""
        individual_dir = output_path / "individual_grape_charts"
        individual_dir.mkdir(exist_ok=True)
        
        unique_grapes = df['grape_id'].unique()
        
        for grape_id in unique_grapes:
            grape_data = df[df['grape_id'] == grape_id].sort_values('days_from_start')
            
            if len(grape_data) < 2:
                continue
            
            # Create the same 9-panel analysis as combined charts
            fig = plt.figure(figsize=(20, 16))
            
            # 1. Area Evolution
            ax1 = plt.subplot(3, 3, 1)
            ax1.plot(grape_data['days_from_start'], grape_data['total_grape_area_pixels'], 
                    'o-', linewidth=3, markersize=8, color='darkgreen', markerfacecolor='lightgreen')
            ax1.set_xlabel('Days from Start')
            ax1.set_ylabel('Area (pixels)')
            ax1.set_title(f'{grape_id} - Area Evolution')
            ax1.grid(True, alpha=0.3)
            self.add_date_axis_weekly(ax1, grape_data['days_from_start'], grape_data['timestamp'])
            
            # 2. RGB Evolution
            ax2 = plt.subplot(3, 3, 2)
            ax2.plot(grape_data['days_from_start'], grape_data['avg_red'], 
                    'o-', linewidth=3, markersize=6, color='red', alpha=0.8, label='Red')
            ax2.plot(grape_data['days_from_start'], grape_data['avg_green'], 
                    'o-', linewidth=3, markersize=6, color='green', alpha=0.8, label='Green')
            ax2.plot(grape_data['days_from_start'], grape_data['avg_blue'], 
                    'o-', linewidth=3, markersize=6, color='blue', alpha=0.8, label='Blue')
            ax2.set_xlabel('Days from Start')
            ax2.set_ylabel('RGB Values')
            ax2.set_title(f'{grape_id} - RGB Evolution')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            self.add_date_axis_weekly(ax2, grape_data['days_from_start'], grape_data['timestamp'])
            
            # 3. LAB A* Channel
            ax3 = plt.subplot(3, 3, 3)
            ax3.plot(grape_data['days_from_start'], grape_data['lab_a'], 
                    'o-', linewidth=3, markersize=8, color='darkred', markerfacecolor='lightcoral')
            ax3.set_xlabel('Days from Start')
            ax3.set_ylabel('LAB a* Value')
            ax3.set_title(f'{grape_id} - LAB a* (Green→Red)')
            ax3.grid(True, alpha=0.3)
            ax3.axhline(y=0, color='gray', linestyle='--', alpha=0.5, label='Green/Red Threshold')
            ax3.legend()
            self.add_date_axis_weekly(ax3, grape_data['days_from_start'], grape_data['timestamp'])
            
            # 4. Hue Evolution
            ax4 = plt.subplot(3, 3, 4)
            ax4.plot(grape_data['days_from_start'], grape_data['avg_hue'], 
                    'o-', linewidth=3, markersize=8, color='purple', markerfacecolor='plum')
            ax4.set_xlabel('Days from Start')
            ax4.set_ylabel('Hue (degrees)')
            ax4.set_title(f'{grape_id} - Hue Evolution')
            ax4.grid(True, alpha=0.3)
            ax4.axhline(y=60, color='lightgreen', linestyle='--', alpha=0.7, label='Green Start (60°)')
            ax4.axhline(y=150, color='darkgreen', linestyle='--', alpha=0.7, label='Green End (150°)')
            ax4.axhline(y=200, color='#b266ff', linestyle='--', alpha=0.7, label='Purple Start (200°)')
            ax4.legend(fontsize=8)
            self.add_date_axis_weekly(ax4, grape_data['days_from_start'], grape_data['timestamp'])
            
            # 5. Green vs Purple Hue Percentages
            ax5 = plt.subplot(3, 3, 5)
            ax5.plot(grape_data['days_from_start'], grape_data['hue_green_percentage'], 
                    'o-', label='Green Hue %', color='green', alpha=0.8, linewidth=3)
            ax5.plot(grape_data['days_from_start'], grape_data['hue_purple_percentage'], 
                    'o-', label='Purple Hue %', color='purple', alpha=0.8, linewidth=3)
            ax5.set_xlabel('Days from Start')
            ax5.set_ylabel('Hue Percentage')
            ax5.set_title(f'{grape_id} - Hue Distribution')
            ax5.legend()
            ax5.grid(True, alpha=0.3)
            self.add_date_axis_weekly(ax5, grape_data['days_from_start'], grape_data['timestamp'])
            
            # 6. Normalized Purple
            ax6 = plt.subplot(3, 3, 6)
            ax6.plot(grape_data['days_from_start'], grape_data['normalized_purple'], 
                    'o-', linewidth=3, markersize=8, color='darkviolet', markerfacecolor='violet')
            ax6.set_xlabel('Days from Start')
            ax6.set_ylabel('Normalized Purple')
            ax6.set_title(f'{grape_id} - Normalized Purple')
            ax6.grid(True, alpha=0.3)
            self.add_date_axis_weekly(ax6, grape_data['days_from_start'], grape_data['timestamp'])
            
            # 7. Purple Pixel Percentage
            ax7 = plt.subplot(3, 3, 7)
            ax7.plot(grape_data['days_from_start'], grape_data['purple_pixel_percentage'], 
                    'o-', linewidth=3, markersize=8, color='darkmagenta', markerfacecolor='magenta')
            ax7.set_xlabel('Days from Start')
            ax7.set_ylabel('Purple Pixels (%)')
            ax7.set_title(f'{grape_id} - Purple Pixel Distribution')
            ax7.grid(True, alpha=0.3)
            self.add_date_axis_weekly(ax7, grape_data['days_from_start'], grape_data['timestamp'])
            
            # 8. Veraison Score
            ax8 = plt.subplot(3, 3, 8)
            ax8.plot(grape_data['days_from_start'], grape_data['veraison_score'], 
                    'o-', linewidth=4, markersize=10, color='red', markerfacecolor='gold')
            ax8.set_xlabel('Days from Start')
            ax8.set_ylabel('Veraison Score (0-1)')
            ax8.set_title(f'{grape_id} - VERAISON PROGRESSION')
            ax8.grid(True, alpha=0.3)
            ax8.set_ylim(0, 1)
            self.add_date_axis_weekly(ax8, grape_data['days_from_start'], grape_data['timestamp'])
            
            # 9. Multi-metric Summary
            ax9 = plt.subplot(3, 3, 9)
            lab_a_min = grape_data['lab_a'].min()
            lab_a_max = grape_data['lab_a'].max()
            lab_a_range = lab_a_max - lab_a_min
            
            if lab_a_range > 0:
                lab_a_normalized = (grape_data['lab_a'] - lab_a_min) / lab_a_range
            else:
                lab_a_normalized = grape_data['lab_a'] * 0
            
            ax9.plot(grape_data['days_from_start'], 
                    lab_a_normalized, 'o-', label='LAB a* (norm)', alpha=0.8, linewidth=2)
            ax9.plot(grape_data['days_from_start'], 
                    grape_data['hue_purple_percentage'] / 100, 'o-', label='Purple Hue % (norm)', alpha=0.8, linewidth=2)
            ax9.plot(grape_data['days_from_start'], 
                    grape_data['normalized_purple'], 'o-', label='Normalized Purple', alpha=0.8, linewidth=2)
            ax9.plot(grape_data['days_from_start'], 
                    grape_data['veraison_score'], 'o-', label='Veraison Score', linewidth=3, color='red')
            ax9.set_xlabel('Days from Start')
            ax9.set_ylabel('Normalized Values (0-1)')
            ax9.set_title(f'{grape_id} - Metric Comparison')
            ax9.legend()
            ax9.grid(True, alpha=0.3)
            self.add_date_axis_weekly(ax9, grape_data['days_from_start'], grape_data['timestamp'])
            
            plt.tight_layout()
            plt.savefig(individual_dir / f"{grape_id}_enhanced_analysis.png", dpi=200, bbox_inches='tight')
            plt.close()
        
    def calculate_grape_metrics(self, mask, image_shape, pixel_to_mm_ratio=None):
        """
        Calculate comprehensive grape metrics for growth tracking
        
        Args:
            mask: Binary segmentation mask
            image_shape: Shape of the original image
            pixel_to_mm_ratio: Conversion factor from pixels to mm (optional)
        
        Returns:
            dict: Comprehensive metrics
        """
        total_pixels = image_shape[0] * image_shape[1]
        segmented_pixels = np.sum(mask > 0)
        
        # Basic area metrics
        surface_area_percentage = (segmented_pixels / total_pixels) * 100
        
        # Connected components analysis
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
        num_clusters = num_labels - 1
        
        if num_clusters > 0:
            cluster_areas = [stats[i, cv2.CC_STAT_AREA] for i in range(1, num_labels)]
            total_grape_area = sum(cluster_areas)
            avg_cluster_size = np.mean(cluster_areas)
            largest_cluster = np.max(cluster_areas)
            
            # Calculate bounding box dimensions for the largest cluster
            largest_idx = np.argmax(cluster_areas) + 1
            largest_stats = stats[largest_idx]
            bbox_width = largest_stats[cv2.CC_STAT_WIDTH]
            bbox_height = largest_stats[cv2.CC_STAT_HEIGHT]
            
            # Equivalent diameter (diameter of circle with same area)
            equiv_diameter = np.sqrt(4 * total_grape_area / np.pi)
            
        else:
            cluster_areas = []
            total_grape_area = 0
            avg_cluster_size = 0
            largest_cluster = 0
            bbox_width = 0
            bbox_height = 0
            equiv_diameter = 0
        
        # Convert to physical units if ratio provided
        if pixel_to_mm_ratio:
            physical_area = total_grape_area * (pixel_to_mm_ratio ** 2)
            physical_equiv_diameter = equiv_diameter * pixel_to_mm_ratio
            physical_bbox_width = bbox_width * pixel_to_mm_ratio
            physical_bbox_height = bbox_height * pixel_to_mm_ratio
        else:
            physical_area = None
            physical_equiv_diameter = None
            physical_bbox_width = None
            physical_bbox_height = None
        
        return {
            # Basic metrics
            'surface_area_percentage': surface_area_percentage,
            'total_pixels': int(total_pixels),
            'segmented_pixels': int(segmented_pixels),
            'total_grape_area_pixels': int(total_grape_area),
            
            # Shape metrics (removed num_clusters as requested)
            'avg_cluster_size': float(avg_cluster_size),
            'largest_cluster': int(largest_cluster),
            'equivalent_diameter_pixels': float(equiv_diameter),
            'bbox_width_pixels': int(bbox_width),
            'bbox_height_pixels': int(bbox_height),
            
            # Physical measurements (if conversion provided)
            'total_grape_area_mm2': physical_area,
            'equivalent_diameter_mm': physical_equiv_diameter,
            'bbox_width_mm': physical_bbox_width,
            'bbox_height_mm': physical_bbox_height,
        }
    
    def _get_empty_color_metrics(self):
        """Return empty metrics dictionary with all new fields"""
        return {
            # Original metrics
            'red_tendency': 0.0, 'avg_red': 0.0, 'avg_green': 0.0, 'avg_blue': 0.0,
            'red_green_ratio': 0.0, 'color_intensity': 0.0, 'color_uniformity': 0.0,
            'red_dominance_percentage': 0.0, 'red_std': 0.0, 'green_std': 0.0, 'blue_std': 0.0,
            'valid_pixels': 0,
            
            # New enhanced metrics
            'avg_hue': 0.0, 'hue_std': 0.0, 'avg_saturation': 0.0, 'avg_value': 0.0,
            'hue_green_percentage': 0.0, 'hue_purple_percentage': 0.0,
            'lab_l': 0.0, 'lab_a': 0.0, 'lab_b': 0.0, 'lab_a_std': 0.0, 'normalized_purple': 0.0,
            'color_variance': 0.0, 'purple_pixel_percentage': 0.0, 'veraison_score': 0.0
        }
    
    def analyze_color_in_mask_enhanced(self, image_rgb, mask):
        """
        Enhanced color analysis for veraison detection with multiple color spaces and metrics
        (Updated to remove purple_index from veraison_score calculation)
        
        Args:
            image_rgb: RGB image
            mask: Binary mask from segmentation
            
        Returns:
            dict: Comprehensive color analysis metrics for veraison detection
        """
        try:
            # Create boolean mask
            mask_bool = mask > 0
            
            if not np.any(mask_bool):
                return self._get_empty_color_metrics()
            
            # Extract RGB values from masked region
            red_values = image_rgb[:, :, 0][mask_bool].astype(np.float64)
            green_values = image_rgb[:, :, 1][mask_bool].astype(np.float64)
            blue_values = image_rgb[:, :, 2][mask_bool].astype(np.float64)
            
            # Basic RGB statistics
            avg_red = np.mean(red_values)
            avg_green = np.mean(green_values)
            avg_blue = np.mean(blue_values)
            
            # === HSV COLOR SPACE ANALYSIS ===
            # Convert to HSV for hue analysis
            rgb_pixels = np.stack([red_values, green_values, blue_values], axis=1)
            rgb_pixels_normalized = rgb_pixels / 255.0
            hsv_pixels = color.rgb2hsv(rgb_pixels_normalized.reshape(-1, 1, 3)).reshape(-1, 3)
            
            hue_values = hsv_pixels[:, 0] * 360  # Convert to degrees
            saturation_values = hsv_pixels[:, 1]
            value_values = hsv_pixels[:, 2]
            
            # Hue statistics
            avg_hue = np.mean(hue_values)
            hue_std = np.std(hue_values)
            
            # Veraison-specific hue analysis
            # Green grapes: 60-120°, Purple grapes: 240-300°
            green_hue_pixels = np.sum((hue_values >= 60) & (hue_values <= 150))
            purple_hue_pixels = np.sum((hue_values >= 200) & (hue_values <= 340))
            
            total_pixels = len(hue_values)
            hue_green_percentage = (green_hue_pixels / total_pixels) * 100
            hue_purple_percentage = (purple_hue_pixels / total_pixels) * 100
            
            # === LAB COLOR SPACE ANALYSIS ===
            # Convert RGB to LAB color space
            lab_pixels = color.rgb2lab(rgb_pixels_normalized.reshape(-1, 1, 3)).reshape(-1, 3)
            
            l_values = lab_pixels[:, 0]  # Lightness
            a_values = lab_pixels[:, 1]  # Green-Red axis
            b_values = lab_pixels[:, 2]  # Blue-Yellow axis
            
            avg_lab_l = np.mean(l_values)
            avg_lab_a = np.mean(a_values)  # KEY for veraison: negative=green, positive=red/purple
            min_lab_a = np.min(a_values)
            max_lab_a = np.max(a_values)
            avg_lab_b = np.mean(b_values)
            
            lab_a_std = np.std(a_values)
            
            # Normalized Purple: (R + B - 2G) / (R + G + B)
            total_intensity = avg_red + avg_green + avg_blue
            normalized_purple = (avg_red + avg_blue - 2*avg_green) / total_intensity if total_intensity > 0 else 0
            
            # === TEXTURE AND SPATIAL ANALYSIS ===
            # Color variance within grape (indicator of veraison heterogeneity)
            color_variance = np.var(red_values) + np.var(green_values) + np.var(blue_values)
            
            # Purple pixel clustering analysis
            # Define purple pixels as those where R > G and (R+B) > 1.2*G
            purple_mask = (red_values > green_values) & ((red_values + blue_values) > 1.2 * green_values)
            purple_pixel_percentage = (np.sum(purple_mask) / total_pixels) * 100
            
            # Color uniformity (inverse of standard deviation)
            color_std_avg = (np.std(red_values) + np.std(green_values) + np.std(blue_values)) / 3.0
            color_uniformity = 1.0 / (1.0 + color_std_avg)
            
            # === VERAISON PROGRESSION INDICATOR (Updated without purple_index) ===
            # Combine multiple metrics for overall veraison score
            lab_a_normalized = (avg_lab_a + 128) / 256.0  # Normalize to [0,1]
            lab_a_normalized = max(0, min(1, lab_a_normalized))  # Clamp to [0,1]

            veraison_score = (
                0.4 * lab_a_normalized +  # LAB a* progression (green→red)
                0.3 * (hue_purple_percentage / 100.0) +  # Hue-based purple percentage
                0.3 * (purple_pixel_percentage / 100.0)  # Spatial purple distribution
            )
            veraison_score = max(0, min(1, veraison_score))  # Clamp to [0,1]
            
            return {
                # Original metrics (keeping for compatibility)
                'avg_red': float(avg_red),
                'avg_green': float(avg_green),
                'avg_blue': float(avg_blue),
                'red_green_ratio': float(avg_red / avg_green if avg_green > 0 else 0),
                'color_intensity': float(total_intensity / 3.0),
                'color_uniformity': float(color_uniformity),
                'red_dominance_percentage': float(np.sum(red_values > np.maximum(green_values, blue_values)) / total_pixels * 100),
                'red_std': float(np.std(red_values)),
                'green_std': float(np.std(green_values)),
                'blue_std': float(np.std(blue_values)),
                'valid_pixels': int(total_pixels),
                
                # === NEW ENHANCED METRICS FOR VERAISON ===
                # HSV Color Space
                'avg_hue': float(avg_hue),
                'hue_std': float(hue_std),
                'avg_saturation': float(np.mean(saturation_values)),
                'avg_value': float(np.mean(value_values)),
                'hue_green_percentage': float(hue_green_percentage),
                'hue_purple_percentage': float(hue_purple_percentage),
                
                # LAB Color Space
                'lab_l': float(avg_lab_l),
                'lab_a': float(avg_lab_a),  
                'lab_b': float(avg_lab_b),
                'lab_a_std': float(lab_a_std),
                
                # Enhanced Color Ratios
                'normalized_purple': float(normalized_purple),  # (R+B-2G)/(R+G+B)
                
                # Spatial and Texture Analysis
                'color_variance': float(color_variance),
                'purple_pixel_percentage': float(purple_pixel_percentage),
                
                # Veraison Summary Score (Updated calculation)
                'veraison_score': float(veraison_score)  # Overall veraison progression [0-1]
            }
            
        except Exception as e:
            print(f"Error in enhanced color analysis: {e}")
            return self._get_empty_color_metrics()
    
    def process_single_grape_folder(self, output_folder, grape_folder, grape_id, pixel_to_mm_ratio=None, save_masks=False):
        """
        Process a single grape's time series folder
        
        Args:
            grape_folder (Path): Path to single grape folder
            grape_id (str): Identifier for this grape
            pixel_to_mm_ratio (float): Optional conversion from pixels to mm
            
        Returns:
            list: Time series data for this grape
        """
        # Find all image files
        image_files = []
        for ext in self.supported_extensions:
            image_files.extend(grape_folder.glob(f"*{ext}"))
            image_files.extend(grape_folder.glob(f"*{ext.upper()}"))
        
        if not image_files:
            print(f"  No supported image files found in {grape_folder}")
            return []
        
        # Remove duplicates from image files list
        unique_image_files = list(set(image_files))
        print(f"  Found {len(unique_image_files)} unique images for {grape_id}")
        
        # Process images and extract timestamps
        grape_time_series = []
        processed_files = set()  # Track processed files to avoid duplicates
        
        for image_file in unique_image_files:
            # Skip if already processed
            if str(image_file) in processed_files:
                print(f"  ! Skipping duplicate file: {image_file.name}")
                continue
            
            processed_files.add(str(image_file))
            
            # Extract timestamp
            timestamp = self.extract_timestamp_from_filename(image_file.name)
            if timestamp is None:
                # Use file modification time as fallback
                timestamp = datetime.fromtimestamp(image_file.stat().st_mtime)
            
            # Load and process image
            image, image_rgb, image_gray = self.load_image(image_file)
            if image is None:
                continue
            
            # Apply segmentation
            mask, markers = self.apply_random_walker(image_gray, image_rgb)
            if mask is None:
                continue
            
            # Post-process
            mask = self.post_process_mask(mask)

            if save_masks:
                output_path = Path(output_folder)
                masks_dir = output_path / "masks" / grape_id
                self.save_mask_overlay(image_rgb, mask, masks_dir, Path(image_file).stem)

            # Calculate metrics
            metrics = self.calculate_grape_metrics(mask, image_gray.shape, pixel_to_mm_ratio)
            
            # Add color analysis
            color_metrics = self.analyze_color_in_mask_enhanced(image_rgb, mask)
            metrics.update(color_metrics)
            
            # Add metadata
            metrics['timestamp'] = timestamp
            metrics['filename'] = image_file.name
            metrics['grape_id'] = grape_id
            metrics['days_from_start'] = 0  # Will be calculated later
            
            grape_time_series.append(metrics)

        # Sort by timestamp
        grape_time_series.sort(key=lambda x: x['timestamp'])
        
        print(f"  Successfully processed {len(grape_time_series)} measurements")
        return grape_time_series
    
    def process_multi_grape_folders(self, input_folder, output_folder, pixel_to_mm_ratio=None, save_masks=False):
        """
        Process multiple grape folders for comparative analysis
        
        Args:
            input_folder (str): Path to folder containing grape subfolders
            output_folder (str): Path to output folder
            pixel_to_mm_ratio (float): Optional conversion from pixels to mm
        """
        input_path = Path(input_folder)
        output_path = Path(output_folder)
        output_path.mkdir(parents=True, exist_ok=True)
        masks_output_path = output_path / "masks"
        masks_output_path.mkdir(parents=True, exist_ok=True)

        # Find all grape subfolders
        grape_folders = [f for f in input_path.iterdir() if f.is_dir()]
        
        if not grape_folders:
            print(f"No subfolders found in {input_folder}")
            return
        
        print(f"Found {len(grape_folders)} grape folders for analysis")
        print("-" * 60)
        
        # Process each grape folder
        all_grape_data = []
        
        for grape_folder in grape_folders:
            grape_id = grape_folder.name
            print(f"Processing grape: {grape_id}")
            
            grape_data = self.process_single_grape_folder(
                output_folder, grape_folder, grape_id, pixel_to_mm_ratio, save_masks 
            )
            
            if grape_data:
                all_grape_data.extend(grape_data)
                print(f"  ✓ Processed {len(grape_data)} time points")
            else:
                print(f"  ✗ No valid data found")
        
        if not all_grape_data:
            print("No valid data found across all grapes")
            return
        
        # Remove duplicates and filter zero areas
        print(f"Total measurements before filtering: {len(all_grape_data)}")
        
        # Create a set to track unique combinations
        seen_combinations = set()
        deduplicated_data = []
        
        for data_point in all_grape_data:
            unique_key = (data_point['grape_id'], data_point['filename'])
            
            if unique_key not in seen_combinations:
                seen_combinations.add(unique_key)
                deduplicated_data.append(data_point)
            else:
                print(f"  ! Removed duplicate: {data_point['grape_id']} - {data_point['filename']}")
        
        print(f"Total measurements after deduplication: {len(deduplicated_data)}")
        
        # Filter out zero area measurements
        filtered_data = []
        zero_area_count = 0
        
        for data_point in deduplicated_data:
            if data_point['total_grape_area_pixels'] > 0:
                filtered_data.append(data_point)
            else:
                zero_area_count += 1
                print(f"  ! Filtered out zero area: {data_point['grape_id']} - {data_point['filename']}")
        
        print(f"Removed {zero_area_count} measurements with zero area")
        print(f"Final measurements for analysis: {len(filtered_data)}")
        
        if not filtered_data:
            print("No valid measurements found after filtering zero areas")
            return
        
        # Calculate days from start for each grape individually
        self.calculate_days_from_start_per_grape(filtered_data)
        
        # Use filtered data
        self.multi_grape_data = filtered_data
        
        # Generate comprehensive analysis
        self.generate_multi_grape_analysis_enhanced(output_path)
        
        print(f"\nMulti-grape analysis complete! Results saved to: {output_path}")

    def create_boxplot_analysis_enhanced(self, df, output_path):
        """Create enhanced boxplot analysis for grape metrics over time"""
        boxplot_dir = output_path / "boxplot_analysis"
        boxplot_dir.mkdir(exist_ok=True)
        
        # Set up the plotting style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Enhanced metrics to analyze over time
        metrics = [
            ('total_grape_area_pixels', 'Total Area (pixels)', 'Area Distribution Over Time'),
            ('surface_area_percentage', 'Surface Coverage (%)', 'Coverage Distribution Over Time'),
            ('equivalent_diameter_pixels', 'Equivalent Diameter (pixels)', 'Diameter Distribution Over Time'),
            ('color_intensity', 'Color Intensity', 'Color Intensity Distribution Over Time'),
            ('lab_a', 'LAB a* Value', 'LAB a* Distribution Over Time (Green→Red)'),
            ('normalized_purple', 'Normalized Purple', 'Normalized Purple Distribution Over Time'),
            ('hue_purple_percentage', 'Purple Hue (%)', 'Purple Hue Percentage Distribution Over Time'),
            ('purple_pixel_percentage', 'Purple Pixels (%)', 'Purple Pixel Percentage Distribution Over Time'),
            ('veraison_score', 'Veraison Score', 'Veraison Score Distribution Over Time')
        ]
        earliest_date = df['timestamp'].min()
        df['days_from_start'] = (df['timestamp'] - earliest_date).dt.days


        for metric, ylabel, title in metrics:
            fig, ax1 = plt.subplots(figsize=(16, 8))
            
            # Get unique days sorted
            unique_days = sorted(df['days_from_start'].unique())
            
            # Prepare data for boxplot - one box per day
            boxplot_data = []
            valid_days = []
            timestamps_for_days = []
            sample_counts = []
            
            for day in unique_days:
                day_data = df[df['days_from_start'] == day][metric]
                if len(day_data) > 0:  # Only include days with data
                    boxplot_data.append(day_data.values)
                    valid_days.append(day)
                    sample_counts.append(len(day_data))
                    # Get a representative timestamp for this day
                    day_timestamp = df[df['days_from_start'] == day]['timestamp'].iloc[0]
                    timestamps_for_days.append(day_timestamp)
            
            # Debug information
            print(f"\nMetric: {metric}")

            
            # Calculate adaptive widths based on sample count
            # More samples = wider box, fewer samples = narrower box
            max_samples = max(sample_counts) if sample_counts else 1
            min_width = 0.2  # Minimum width for single samples
            max_width = 0.8  # Maximum width for many samples
            
            adaptive_widths = []
            for count in sample_counts:
                if count == 1:
                    # Single sample: use minimum width
                    width = min_width
                else:
                    # Scale width based on sample count
                    width_factor = min(count / max_samples, 1.0)
                    width = min_width + (max_width - min_width) * width_factor
                adaptive_widths.append(width)
            
            # Create boxplot with adaptive widths
            if len(boxplot_data) > 0:
                box_plot = ax1.boxplot(boxplot_data, 
                                    positions=valid_days, 
                                    patch_artist=True, 
                                    widths=adaptive_widths)  # Use adaptive widths
                
                # Color the boxes with a gradient
                colors = plt.cm.viridis(np.linspace(0, 1, len(box_plot['boxes'])))
                for patch, color in zip(box_plot['boxes'], colors):
                    patch.set_facecolor(color)
                    patch.set_alpha(0.7)
            
            ax1.set_xlabel('Days from Start', fontsize=12)
            ax1.set_ylabel(ylabel, fontsize=12)
            ax1.set_title(f'{title} - Distribution Across All Grapes Per Day', fontsize=14, fontweight='bold')
            ax1.grid(True, alpha=0.3)
            
            # Set x-axis limits to show all days properly
            if valid_days:
                ax1.set_xlim(min(valid_days) - 1, max(valid_days) + 1)
            
            # Add reference lines for specific metrics
            if metric == 'lab_a':
                ax1.axhline(y=0, color='red', linestyle='--', alpha=0.5, label='Green/Red Threshold')
                ax1.legend()
            elif metric == 'veraison_score':
                ax1.set_ylim(0, 1)
            
            # Create second x-axis for dates with proper alignment
            ax2 = ax1.twiny()
            ax2.set_xlim(ax1.get_xlim())  # Match the main axis limits
            
            # Set date ticks only at positions where we have data
            if timestamps_for_days and len(timestamps_for_days) > 0:
                # Use the same positions as the main axis
                ax2.set_xticks(valid_days)
                ax2.set_xticklabels([ts.strftime('%d-%m') for ts in timestamps_for_days], 
                                rotation=45, ha='right')
            
            ax2.set_xlabel('Date', fontsize=12)
            
            plt.tight_layout()
            plt.savefig(boxplot_dir / f"{metric.replace('/', '_').replace(' ', '_').lower()}_daily_distribution.png", 
                    dpi=200, bbox_inches='tight')
            plt.close()
    
    def create_combined_enhanced_color_charts(self, df, output_path):
            """Create combined enhanced color analysis charts with area and RGB evolution"""
            color_dir = output_path / "combined_enhanced_color_analysis"
            color_dir.mkdir(exist_ok=True)
            
            # === COMBINED VERAISON-FOCUSED ANALYSIS WITH AREA AND RGB ===
            fig = plt.figure(figsize=(20, 16))
            earliest_date = df['timestamp'].min()
            df['days_from_start'] = (df['timestamp'] - earliest_date).dt.days        
            
            # Calculate average values across all grapes for each day
            grouped_data = df.groupby('days_from_start').agg({
                'lab_a': 'mean',
                'avg_hue': 'mean',
                'hue_green_percentage': 'mean',
                'hue_purple_percentage': 'mean',
                'normalized_purple': 'mean',
                'purple_pixel_percentage': 'mean',
                'veraison_score': 'mean',
                'total_grape_area_pixels': 'mean',
                'avg_red': 'mean',
                'avg_green': 'mean',
                'avg_blue': 'mean',
                'timestamp': 'first'
            }).reset_index()
            
            # 1. Area Evolution
            ax1 = plt.subplot(3, 3, 1)
            ax1.plot(grouped_data['days_from_start'], grouped_data['total_grape_area_pixels'], 
                    'o-', linewidth=3, markersize=8, color='darkgreen', markerfacecolor='lightgreen')
            ax1.set_xlabel('Days from Start')
            ax1.set_ylabel('Average Area (pixels)')
            ax1.set_title('Average Area Evolution')
            ax1.grid(True, alpha=0.3)
            self.add_date_axis_weekly(ax1, grouped_data['days_from_start'], grouped_data['timestamp'])
            
            # 2. RGB Evolution
            ax2 = plt.subplot(3, 3, 2)
            ax2.plot(grouped_data['days_from_start'], grouped_data['avg_red'], 
                    'o-', linewidth=3, markersize=6, color='red', alpha=0.8, label='Red')
            ax2.plot(grouped_data['days_from_start'], grouped_data['avg_green'], 
                    'o-', linewidth=3, markersize=6, color='green', alpha=0.8, label='Green')
            ax2.plot(grouped_data['days_from_start'], grouped_data['avg_blue'], 
                    'o-', linewidth=3, markersize=6, color='blue', alpha=0.8, label='Blue')
            ax2.set_xlabel('Days from Start')
            ax2.set_ylabel('Average RGB Values')
            ax2.set_title('Average RGB Evolution')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            self.add_date_axis_weekly(ax2, grouped_data['days_from_start'], grouped_data['timestamp'])
            
            # 3. LAB A* Channel (KEY for veraison)
            ax3 = plt.subplot(3, 3, 3)
            ax3.plot(grouped_data['days_from_start'], grouped_data['lab_a'], 
                    'o-', linewidth=3, markersize=8, color='darkred', markerfacecolor='lightcoral')
            ax3.set_xlabel('Days from Start')
            ax3.set_ylabel('Average LAB a* Value')
            ax3.set_title('Average LAB a* (Green→Red Transition)')
            ax3.grid(True, alpha=0.3)
            ax3.axhline(y=0, color='gray', linestyle='--', alpha=0.5, label='Green/Red Threshold')
            ax3.legend()
            self.add_date_axis_weekly(ax3, grouped_data['days_from_start'], grouped_data['timestamp'])
            
            # 4. Hue Evolution
            ax4 = plt.subplot(3, 3, 4)
            ax4.plot(grouped_data['days_from_start'], grouped_data['avg_hue'], 
                    'o-', linewidth=3, markersize=8, color='purple', markerfacecolor='plum')
            ax4.set_xlabel('Days from Start')
            ax4.set_ylabel('Average Hue (degrees)')
            ax4.set_title('Average Hue Evolution')
            ax4.grid(True, alpha=0.3)
            ax4.axhline(y=60, color='lightgreen', linestyle='--', alpha=0.7, label='Green Start (60°)')
            ax4.axhline(y=150, color='darkgreen', linestyle='--', alpha=0.7, label='Green End (150°)')
            ax4.axhline(y=200, color='#b266ff', linestyle='--', alpha=0.7, label='Purple Start (200°)')
            ax4.legend(fontsize=8)
            self.add_date_axis_weekly(ax4, grouped_data['days_from_start'], grouped_data['timestamp'])
            
            # 5. Green vs Purple Hue Percentages
            ax5 = plt.subplot(3, 3, 5)
            ax5.plot(grouped_data['days_from_start'], grouped_data['hue_green_percentage'], 
                    'o-', label='Green Hue %', color='green', alpha=0.8, linewidth=3)
            ax5.plot(grouped_data['days_from_start'], grouped_data['hue_purple_percentage'], 
                    'o-', label='Purple Hue %', color='purple', alpha=0.8, linewidth=3)
            ax5.set_xlabel('Days from Start')
            ax5.set_ylabel('Average Hue Percentage')
            ax5.set_title('Average Hue Distribution')
            ax5.legend()
            ax5.grid(True, alpha=0.3)
            self.add_date_axis_weekly(ax5, grouped_data['days_from_start'], grouped_data['timestamp'])
            
            # 6. Normalized Purple
            ax6 = plt.subplot(3, 3, 6)
            ax6.plot(grouped_data['days_from_start'], grouped_data['normalized_purple'], 
                    'o-', linewidth=3, markersize=8, color='darkviolet', markerfacecolor='violet')
            ax6.set_xlabel('Days from Start')
            ax6.set_ylabel('Average Normalized Purple')
            ax6.set_title('Average Normalized Purple (R+B-2G)/(R+G+B)')
            ax6.grid(True, alpha=0.3)
            self.add_date_axis_weekly(ax6, grouped_data['days_from_start'], grouped_data['timestamp'])
            
            # 7. Purple Pixel Percentage
            ax7 = plt.subplot(3, 3, 7)
            ax7.plot(grouped_data['days_from_start'], grouped_data['purple_pixel_percentage'], 
                    'o-', linewidth=3, markersize=8, color='darkmagenta', markerfacecolor='magenta')
            ax7.set_xlabel('Days from Start')
            ax7.set_ylabel('Average Purple Pixels (%)')
            ax7.set_title('Average Purple Pixel Distribution')
            ax7.grid(True, alpha=0.3)
            self.add_date_axis_weekly(ax7, grouped_data['days_from_start'], grouped_data['timestamp'])
            
            # 8. Veraison Score
            ax8 = plt.subplot(3, 3, 8)
            ax8.plot(grouped_data['days_from_start'], grouped_data['veraison_score'], 
                    'o-', linewidth=4, markersize=10, color='red', markerfacecolor='gold')
            ax8.set_xlabel('Days from Start')
            ax8.set_ylabel('Average Veraison Score (0-1)')
            ax8.set_title('AVERAGE VERAISON PROGRESSION SCORE')
            ax8.grid(True, alpha=0.3)
            ax8.set_ylim(0, 1)
            self.add_date_axis_weekly(ax8, grouped_data['days_from_start'], grouped_data['timestamp'])
            
            # 9. Multi-metric Summary Comparison
            ax9 = plt.subplot(3, 3, 9)
            lab_a_min = grouped_data['lab_a'].min()
            lab_a_max = grouped_data['lab_a'].max()
            lab_a_range = lab_a_max - lab_a_min
            
            if lab_a_range > 0:
                lab_a_normalized = (grouped_data['lab_a'] - lab_a_min) / lab_a_range
            else:
                lab_a_normalized = grouped_data['lab_a'] * 0
            
            ax9.plot(grouped_data['days_from_start'], 
                    lab_a_normalized, 'o-', label='LAB a* (norm)', alpha=0.8, linewidth=2)
            ax9.plot(grouped_data['days_from_start'], 
                    grouped_data['hue_purple_percentage'] / 100, 'o-', label='Purple Hue % (norm)', alpha=0.8, linewidth=2)
            ax9.plot(grouped_data['days_from_start'], 
                    grouped_data['normalized_purple'], 'o-', label='Normalized Purple', alpha=0.8, linewidth=2)
            ax9.plot(grouped_data['days_from_start'], 
                    grouped_data['veraison_score'], 'o-', label='Veraison Score', linewidth=3, color='red')
            ax9.set_xlabel('Days from Start')
            ax9.set_ylabel('Normalized Values (0-1)')
            ax9.set_title('Average Metric Comparison')
            ax9.legend()
            ax9.grid(True, alpha=0.3)
            self.add_date_axis_weekly(ax9, grouped_data['days_from_start'], grouped_data['timestamp'])
            
            plt.tight_layout()
            plt.savefig(color_dir / "combined_enhanced_veraison_analysis.png", dpi=200, bbox_inches='tight')
            plt.close()
        
    def save_mask_overlay(self, image_rgb, mask, output_path, filename):
        """Save mask overlay on original image"""
        try:
            # Create output directory if it doesn't exist
            output_path = Path(output_path)
            output_path.mkdir(parents=True, exist_ok=True)
            
            # Create overlay
            overlay = image_rgb.copy().astype(np.float32)
            
            # Create colored mask (red overlay)
            mask_colored = np.zeros_like(overlay)
            mask_colored[:, :, 0] = mask.astype(np.float32)  # Red channel
            
            # Apply mask with transparency
            alpha = 0.4
            mask_bool = mask > 0
            overlay[mask_bool] = (1 - alpha) * overlay[mask_bool] + alpha * mask_colored[mask_bool]
            
            # Convert back to uint8
            overlay = np.clip(overlay, 0, 255).astype(np.uint8)
            
            # Save using cv2 instead of matplotlib for better performance
            output_file = output_path / f"{filename}_mask.png"
            overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(output_file), overlay_bgr)
            
            print(f"    Saved mask overlay: {output_file}")
            
        except Exception as e:
            print(f"Error saving mask overlay: {e}")
            import traceback
            traceback.print_exc()  # This will help debug any remaining issues
            
    def generate_multi_grape_analysis_enhanced(self, output_path):
        """Generate enhanced multi-grape analysis with combined charts"""
        if not self.multi_grape_data:
            print("No multi-grape data available for analysis")
            return
        
        # Create DataFrame
        df = pd.DataFrame(self.multi_grape_data)
        
        # Save detailed CSV
        csv_path = output_path / "multi_grape_time_series_data.csv"
        df.to_csv(csv_path, index=False)
        
        # Create visualizations
        self.create_boxplot_analysis_enhanced(df, output_path)  # Use enhanced version
        self.create_individual_grape_charts(df, output_path)
        self.create_combined_enhanced_color_charts(df, output_path)  # NEW: Combined charts
        
        print(f"Enhanced multi-grape data saved to: {csv_path}")


# Main execution function
def main():
    parser = argparse.ArgumentParser(description='Multi-Grape Growth Monitoring System')
    parser.add_argument('input_folder', help='Path to folder containing grape subfolders')
    parser.add_argument('output_folder', help='Path to output folder for results')
    parser.add_argument('--beta', type=int, default=130, help='Random walker beta parameter')
    parser.add_argument('--mode', type=str, default='bf', choices=['bf', 'cg_mg'], 
                       help='Random walker solver mode')
    parser.add_argument('--pixel-to-mm', type=float, help='Pixel to mm conversion ratio')
    parser.add_argument('--save_masks', action='store_true', help='Save segmentation mask overlays')
    args = parser.parse_args()
    
    # Initialize monitor
    monitor = MultiGrapeTimeSeriesMonitor(beta=args.beta, mode=args.mode)
    
    # Process multi-grape folders
    monitor.process_multi_grape_folders(
        args.input_folder, 
        args.output_folder, 
        args.pixel_to_mm,
        args.save_masks  # Add this line
    )


if __name__ == "__main__":
    main()
