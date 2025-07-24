import cv2
import numpy as np
from ultralytics import YOLO
import os
from pathlib import Path
import json
import argparse
import re
import pandas as pd
from datetime import datetime
import shutil
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from sahi.utils.cv import read_image

class TwoStageGrapeDetector:
    def __init__(self, main_model_path, bbox_model_path=None, expansion_factor=1.3, target_size=640,memory_window = 3, main_model_conf=0.2, main_model_iou=0.1, main_model_size=640, bbox_model_size=640, bbox_model_iou=0.1, bbox_model_conf = 0.2, use_sahi=True, slice_height=640, slice_width=640, overlap_height_ratio=0.2, overlap_width_ratio=0.2, global_iou_threshold=0.3, debug=False):
        """
        Initialize the two-stage grape detector
        
        Args:
            main_model_path (str): Path to main YOLO11 model
            bbox_model_path (str): Path to bbox YOLO11 model (optional)
            target_size (int): Target size for resizing crops
            main_model_conf (float): Confidence threshold for main model
            main_model_iou (float): IoU threshold for main model
            main_model_size (int): Input size for main model inference
            debug (bool): Enable debug output
        """
        self.main_model = YOLO(main_model_path)
        self.bbox_model = YOLO(bbox_model_path) if bbox_model_path else None
        self.target_size = target_size
        self.memory_window = memory_window
        self.main_model_size = main_model_size
        self.main_model_conf = main_model_conf
        self.main_model_iou = main_model_iou
        self.expansion_factor = expansion_factor
        self.global_iou_threshold = global_iou_threshold
        self.previous_detections = {}  # {image_name: [detection_list]}
        self.debug = debug
        self.bbox_model_iou = bbox_model_iou
        self.bbox_model_size = bbox_model_size
        self.bbox_model_conf = bbox_model_conf
        # SAHI configuration
        self.use_sahi = use_sahi
        self.slice_height = slice_height
        self.slice_width = slice_width
        self.overlap_height_ratio = overlap_height_ratio
        self.overlap_width_ratio = overlap_width_ratio
        if self.use_sahi:
            self.sahi_detection_model = AutoDetectionModel.from_pretrained(
                model_type='ultralytics',
                model_path=main_model_path,
                confidence_threshold=main_model_conf,
                device='cpu'  # Will auto-detect GPU if available
            )

        self.debug_stats = {
            'regions_investigated': 0,
            'regions_skipped_invalid_crop': 0,
            'missed_detections_found': 0,
            'source_images_used': {},
            'investigation_history': [],
            'missed_investigations': []
        }
      
    def expand_bbox(self, bbox, img_shape, expansion_factor=None):
        """
        Expand bbox to include more context
        
        Args:
            bbox (list): [x, y, w, h] format
            img_shape (tuple): Image shape (h, w, c)
            expansion_factor (float): 1.5 means 50% larger bbox
        """
        

        if expansion_factor is None:
            expansion_factor = self.expansion_factor
        
        print(f"Expanding bbox {bbox} with factor {expansion_factor}")

        h, w = img_shape[:2]
        x, y, w_box, h_box = bbox
        

        # Expand dimensions
        new_w = w_box * expansion_factor
        new_h = h_box * expansion_factor
        
        # Calculate new top-left corner
        new_x = max(0, x - new_w/2)
        new_y = max(0, y - new_h/2)
        
        # Ensure we don't go outside image bounds
        new_x2 = min(w, new_x + new_w)
        new_y2 = min(h, new_y + new_h)
        
        # Adjust if we hit boundaries
        new_x = max(0, new_x2 - new_w)
        new_y = max(0, new_y2 - new_h)
        
        return [int(x), int(y), int(new_w), int(new_h)]
    
    def crop_and_resize(self, image, bbox, target_size=None):
        """
        Crop bbox region and resize to target size
        
        Args:
            image (np.array): Input image
            bbox (list): [x, y, w, h] format
            target_size (int): Target size for resizing
        """
        if target_size is None:
            target_size = self.target_size
        
        print(f"Cropping and resizing bbox {bbox} to {target_size}x{target_size}")

        x, y, w, h = bbox
        
        # Better bounds checking
        img_h, img_w = image.shape[:2]
        
        # Ensure coordinates are within image bounds
        x = max(0, min(x, img_w - 1))
        y = max(0, min(y, img_h - 1))
        x2 = max(x + 1, min(x + w, img_w))
        y2 = max(y + 1, min(y + h, img_h))
        
        # Check if we have a valid crop area
        if x2 <= x or y2 <= y:
            return None
            
        crop = image[y:y2, x:x2]
        print(f"  Crop shape: {crop.shape}, x={x}, y={y}, w={w}, h={h}")
        
        if crop.size == 0 or crop.shape[0] == 0 or crop.shape[1] == 0:
            return None
            
        # Resize while maintaining aspect ratio
        resized = cv2.resize(crop, (target_size, target_size))
        print(f"  Crop coordinates after resized: ({resized.shape[1]}, {resized.shape[0]})")
        print(f"  Resized crop shape: {resized.shape}")
        return resized
    
    def detect_full_image_sahi(self, image, image_path, image_name):
        """
        Run main detection on full image using SAHI
        
        Args:
            image (np.array): Input image
            image_name (str): Name/ID of the image
        """
        if self.debug:
            print(f"  [SAHI MODEL] Running SAHI detection on {image_name}")
            print(f"    Slice size: {self.slice_width}x{self.slice_height}")
            print(f"    Overlap: {self.overlap_width_ratio}x{self.overlap_height_ratio}")
            print(f"    Confidence: {self.main_model_conf}")
        
        result = get_sliced_prediction(
            read_image(image_path),
            self.sahi_detection_model,
            slice_height=self.slice_height,
            slice_width=self.slice_width,
            overlap_height_ratio=self.overlap_height_ratio,
            overlap_width_ratio=self.overlap_width_ratio,
            verbose=0
        )
        
        detections = []
        if result.object_prediction_list:
            for prediction in result.object_prediction_list:
                bbox_data = prediction.bbox
                
                # Convert to [x, y, w, h] format
                bbox = [
                    int(bbox_data.minx), 
                    int(bbox_data.miny),
                    int(bbox_data.maxx - bbox_data.minx),
                    int(bbox_data.maxy - bbox_data.miny)
                ]
                
                detections.append({
                    'bbox': bbox,
                    'confidence': float(prediction.score.value),
                    'class': int(prediction.category.id),
                    'source': 'sahi_main_model'
                })
        
        # Update debug stats
        if hasattr(result, 'durations_in_seconds'):
            if 'slice_count' in result.durations_in_seconds:
                self.debug_stats['sahi_tiles_processed'] += result.durations_in_seconds.get('slice_count', 0)
        
        if self.debug:
            print(f"    [SAHI MODEL] Found {len(detections)} detections in {image_name}")
            if hasattr(result, 'durations_in_seconds'):
                print(f"    Processing time: {result.durations_in_seconds}")
        
        return detections

    def detect_full_image_regular(self, image, image_name):
        """
        Run main detection on full image using regular YOLO inference
        
        Args:
            image (np.array): Input image
            image_name (str): Name/ID of the image
        """
        if self.debug:
            print(f"  [REGULAR MODEL] Running detection on {image_name} (input_size={self.main_model_size}, conf={self.main_model_conf}, iou={self.main_model_iou})")
        
        results = self.main_model(image, imgsz=self.main_model_size, conf=self.main_model_conf, iou=self.main_model_iou, verbose=False)
        
        detections = []
        if len(results[0].boxes) > 0:
            boxes = results[0].boxes
            for i in range(len(boxes)):
                x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy()
                conf = boxes.conf[i].cpu().numpy()
                cls = boxes.cls[i].cpu().numpy()
                
                bbox = [int(x1), int(y1), int(x2-x1), int(y2-y1)]
                detections.append({
                    'bbox': bbox,
                    'confidence': float(conf),
                    'class': int(cls),
                    'source': 'regular_main_model'
                })
        
        if self.debug:
            print(f"    [REGULAR MODEL] Found {len(detections)} detections in {image_name}")
        
        return detections
    
    def detect_full_image(self, image, image_path, image_name):
        """
        Run main detection on full image (SAHI or regular based on configuration)
        
        Args:
            image (np.array): Input image
            image_name (str): Name/ID of the image
        """
        if self.use_sahi:
            sahi_detections = self.detect_full_image_sahi(image, image_path, image_name)
            
            # Optionally compare with regular detection for debugging
            if self.debug:
                regular_detections = self.detect_full_image_regular(image, image_name)
                self.debug_stats['sahi_vs_regular_comparison'].append({
                    'image_name': image_name,
                    'sahi_count': len(sahi_detections),
                    'regular_count': len(regular_detections),
                    'sahi_avg_conf': np.mean([d['confidence'] for d in sahi_detections]) if sahi_detections else 0,
                    'regular_avg_conf': np.mean([d['confidence'] for d in regular_detections]) if regular_detections else 0
                })
                print(f"    [COMPARISON] SAHI: {len(sahi_detections)} detections, Regular: {len(regular_detections)} detections")
            
            return sahi_detections
        else:
            return self.detect_full_image_regular(image, image_name)
    
    def detect_missed_regions(self, image, image_name, prev_image_names, current_detections, 
                             global_iou_threshold=0.3, min_confidence=0.3):
        """
        Detect in regions where objects were previously found but not detected today
        
        Args:
            image (np.array): Current image
            image_name (str): Current image name
            prev_image_names (list): List of previous image names to check
            current_detections (list): Current image detections from main model
            global_iou_threshold (float): IoU threshold for matching detections
            min_confidence (float): Minimum confidence for bbox model detections
        """
        print(f"  [BBOX MODEL] Detecting missed regions in {image_name} using previous detections")
        if not prev_image_names or self.bbox_model is None:
            if self.debug:
                print(f"    [BBOX MODEL] Skipping - no previous images or bbox model not available")
            return []
        
        # Collect all previous detections from memory window (EXCLUDING current image)
        all_previous_detections = []
        source_tracking = {}
        
        for prev_name in prev_image_names:
            if prev_name == image_name:
                continue
                
            if prev_name in self.previous_detections:
                prev_dets = self.previous_detections[prev_name]
                for det in prev_dets:
                    det_id = len(all_previous_detections)
                    all_previous_detections.append(det)
                    source_tracking[det_id] = prev_name
        
        if self.debug:
            print(f"    [BBOX MODEL] Analyzing {len(all_previous_detections)} previous detections from {len([p for p in prev_image_names if p != image_name])} images")
        
        missed_detections = []
        
        # Remove duplicates from previous detections
        unique_previous_detections = self.remove_duplicate_detections(all_previous_detections, global_iou_threshold)
        
        if self.debug:
            print(f"    [BBOX MODEL] After deduplication: {len(unique_previous_detections)} unique regions to investigate")
        
        regions_investigated = 0
        regions_with_detections = 0
        regions_skipped = 0
        
        for idx, prev_det in enumerate(unique_previous_detections):
            prev_bbox = prev_det['bbox']
            
            # Find which image this detection originally came from
            source_image = "unknown"
            for i, orig_det in enumerate(all_previous_detections):
                if (orig_det['bbox'] == prev_det['bbox'] and 
                    abs(orig_det['confidence'] - prev_det['confidence']) < 0.001):
                    source_image = source_tracking.get(i, "unknown")
                    break
            
            # Check if this region has a detection today
            has_current_detection = False
            best_iou = 0.0
            for curr_det in current_detections:
                curr_bbox = curr_det['bbox']
                iou = calculate_iou_boxes(prev_bbox, curr_bbox)
                best_iou = max(best_iou, iou)
                if iou > global_iou_threshold:
                    has_current_detection = True
                    break
            
            # If no detection in this region, run bbox model
            if not has_current_detection:
                regions_investigated += 1
                self.debug_stats['regions_investigated'] += 1
                
                if self.debug:
                    print(f"      [INVESTIGATING] Region {regions_investigated}: bbox={prev_bbox}, "
                          f"source_image='{source_image}', best_iou_with_current={best_iou:.3f}")
                
                expanded_bbox = self.expand_bbox(prev_bbox, image.shape, expansion_factor=self.expansion_factor)
                crop = self.crop_and_resize(image, expanded_bbox, self.target_size)
                
                if crop is not None:
                    # Run bbox model on cropped region
                    print(f"        [CROP] Running bbox model on expanded region {expanded_bbox} (size={self.target_size}, iou={self.bbox_model_iou}, conf={self.bbox_model_conf})")
                    crop_results = self.bbox_model(crop, imgsz=self.bbox_model_size, conf=self.bbox_model_conf, iou=self.bbox_model_iou, verbose=False)

                    if len(crop_results[0].boxes) > 0:
                        regions_with_detections += 1
                        
                        # Convert crop coordinates back to original image coordinates
                        crop_boxes = crop_results[0].boxes
                        region_detections = []
                        
                        for i in range(len(crop_boxes)):
                            crop_x1, crop_y1, crop_x2, crop_y2 = crop_boxes.xyxy[i].cpu().numpy()
                            crop_conf = crop_boxes.conf[i].cpu().numpy()
                            crop_cls = crop_boxes.cls[i].cpu().numpy()
                            
                            # Only keep detections above minimum confidence
                            if crop_conf < min_confidence:
                                continue
                            
                            # Scale back to expanded bbox coordinates
                            scale_x = expanded_bbox[2] / self.target_size
                            scale_y = expanded_bbox[3] / self.target_size
                            
                            orig_x1 = expanded_bbox[0] + crop_x1 * scale_x
                            orig_y1 = expanded_bbox[1] + crop_y1 * scale_y
                            orig_x2 = expanded_bbox[0] + crop_x2 * scale_x
                            orig_y2 = expanded_bbox[1] + crop_y2 * scale_y
                            
                            detection = {
                                'bbox': [int(orig_x1), int(orig_y1), 
                                        int(orig_x2-orig_x1), int(orig_y2-orig_y1)],
                                'confidence': float(crop_conf),
                                'class': int(crop_cls),  # Keep the actual predicted class
                                'source': 'bbox_model',
                                'source_image': source_image,
                                'original_bbox': prev_bbox
                            }
                            
                            region_detections.append(detection)
                            missed_detections.append(detection)
                        
                        if region_detections:
                            self.debug_stats['missed_detections_found'] += len(region_detections)
                            if source_image not in self.debug_stats['source_images_used']:
                                self.debug_stats['source_images_used'][source_image] = 0
                            self.debug_stats['source_images_used'][source_image] += len(region_detections)
                            
                            if self.debug:
                                print(f"        [FOUND!] {len(region_detections)} detections in this region, "
                                      f"confidences: {[format(d['confidence'], '.3f') for d in region_detections]}")
                                
                            # Store investigation history
                            self.debug_stats['investigation_history'].append({
                                'current_image': image_name,
                                'source_image': source_image,
                                'original_bbox': prev_bbox,
                                'expanded_bbox': expanded_bbox,
                                'detections_found': len(region_detections),
                                'confidences': [d['confidence'] for d in region_detections]
                            })
                    else:
                        if self.debug:
                            print(f"        [NO DETECTION] No objects found in this region")
                        
                        self.debug_stats['missed_investigations'].append({
                            'current_image': image_name,
                            'source_image': source_image,
                            'original_bbox': prev_bbox,
                            'expanded_bbox': expanded_bbox,
                            'best_iou_with_current': best_iou
                        })
                else:
                    regions_skipped += 1
                    self.debug_stats['regions_skipped_invalid_crop'] += 1
                    
                    if self.debug:
                        print(f"        [SKIPPED] Invalid crop for region {regions_investigated}, "
                              f"bbox={prev_bbox}, expanded_bbox={expanded_bbox}")
        
        if self.debug:
            total_regions = len(unique_previous_detections)
            regions_with_current = total_regions - regions_investigated - regions_skipped
            print(f"    [BBOX MODEL] Summary: {total_regions} total regions, "
                  f"{regions_with_current} had current detections, "
                  f"{regions_investigated} investigated, {regions_skipped} skipped (invalid crop), "
                  f"found detections in {regions_with_detections} regions, "
                  f"total new detections: {len(missed_detections)}")
        
        return missed_detections
    
    def unified_deduplication(self, detections, global_iou_threshold=0.3):
        """
        Unified deduplication that handles all cases with a single parameter.
        Priority order: main_model > sahi_main_model > bbox_model
        
        Args:
            detections (list): All detections from all models
            global_iou_threshold (float): Single IoU threshold for all deduplication
        
        Returns:
            list: Deduplicated detections
        """
        if len(detections) == 0:
            return detections
        
        # Define model priority (higher number = higher priority)
        model_priority = {
            'regular_main_model': 3,
            'sahi_main_model': 2,  
            'bbox_model': 1
        }
        
        # Sort by priority first, then by confidence within same priority
        detections.sort(key=lambda x: (model_priority.get(x['source'], 0), x['confidence']), reverse=True)
        
        kept_detections = []
        removed_count = 0
        
        for detection in detections:
            should_keep = True
            
            # Check against all already kept detections
            for kept_detection in kept_detections:
                iou = calculate_iou_boxes(detection['bbox'], kept_detection['bbox'])
                
                if iou > global_iou_threshold:
                    should_keep = False
                    removed_count += 1
                    
                    if self.debug:
                        print(f"      [UNIFIED_DEDUP] Removed {detection['source']} detection "
                            f"(conf={detection['confidence']:.3f}, IoU={iou:.3f}) "
                            f"due to overlap with {kept_detection['source']} "
                            f"(conf={kept_detection['confidence']:.3f})")
                    break
            
            if should_keep:
                kept_detections.append(detection)
        
        if self.debug and removed_count > 0:
            print(f"      [UNIFIED_DEDUP] Removed {removed_count} duplicates, kept {len(kept_detections)}")
        
        return kept_detections

    def detect_two_stage(self, image, image_path, image_name, prev_image_names=None, 
                            memory_window=3, global_iou_threshold=0.3, 
                            main_model_conf=0.3, bbox_model_conf=0.3):
        """
        Simplified two-stage detection with unified deduplication
        
        Args:
            global_iou_threshold (float): Single IoU threshold for all deduplication
            Other args same as before
        """
        if self.debug:
            detection_method = "SAHI" if self.use_sahi else "Regular"
            print(f"  [TWO-STAGE-UNIFIED] Processing {image_name} using {detection_method}")
        
        # Stage 1: Main model detection
        main_detections = self.detect_full_image(image, image_path, image_name)
        main_detections = [d for d in main_detections if d['confidence'] >= main_model_conf]
        
        # Stage 2: Bbox model detection in missed regions
        missed_detections = []
        if prev_image_names and len(prev_image_names) > 0:
            recent_prev_names = [name for name in prev_image_names[-memory_window:] if name != image_name]
            missed_detections = self.detect_missed_regions(
                image, image_name, recent_prev_names, main_detections,
                global_iou_threshold=global_iou_threshold,  
                min_confidence=bbox_model_conf
            )
        
        # UNIFIED DEDUPLICATION: One step handles everything
        all_detections = main_detections + missed_detections
        final_detections = self.unified_deduplication(all_detections, global_iou_threshold)
        
        # Store filtered detections
        self.previous_detections[image_name] = [d for d in final_detections if d['source'] in ['regular_main_model', 'sahi_main_model']]
        
        if self.debug:
            main_count = len([d for d in final_detections if d['source'] in ['regular_main_model', 'sahi_main_model']])
            bbox_count = len([d for d in final_detections if d['source'] == 'bbox_model'])
            print(f"  [UNIFIED_FINAL] {len(final_detections)} total detections "
                f"(main: {main_count}, bbox: {bbox_count})")
        
        return final_detections

    def remove_duplicate_detections(self, detections, global_iou_threshold=0.5):
        """Remove duplicate detections using NMS-like approach"""
        if len(detections) == 0:
            return detections
        
        # Sort by confidence
        detections.sort(key=lambda x: x['confidence'], reverse=True)
        
        keep = []
        for i, det in enumerate(detections):
            should_keep = True
            for kept_det in keep:
            
                iou = calculate_iou_boxes(det['bbox'], kept_det['bbox'])
                if iou > global_iou_threshold:
                    should_keep = False
                    if self.debug and 'source_image' in det:
                        print(f"      [DUPLICATE] Removing detection (IoU={iou:.3f}) from {det.get('source_image', 'unknown')}")
                    break
            
            if should_keep:
                keep.append(det)
        
        return keep
    
    def print_debug_summary(self):
        """Print comprehensive debug summary"""
        if not self.debug:
            return
            
        print("\n" + "="*60)
        print("DEBUG SUMMARY")
        print("="*60)
        print(f"Total regions investigated: {self.debug_stats['regions_investigated']}")
        print(f"Total regions skipped (invalid crop): {self.debug_stats['regions_skipped_invalid_crop']}")
        print(f"Total missed detections found: {self.debug_stats['missed_detections_found']}")
        print(f"Total missed investigations (no detection): {len(self.debug_stats['missed_investigations'])}")
        
        if self.debug_stats['source_images_used']:
            print("\nSource images that contributed to missed detections:")
            for img, count in sorted(self.debug_stats['source_images_used'].items()):
                print(f"  {img}: {count} detection(s)")
        
        if self.debug_stats['investigation_history']:
            print(f"\nSuccessful investigations ({len(self.debug_stats['investigation_history'])}):")
            for i, investigation in enumerate(self.debug_stats['investigation_history'][-10:], 1):
                print(f"  {i}. Current: {investigation['current_image']}")
                print(f"     Source: {investigation['source_image']}")
                print(f"     Found: {investigation['detections_found']} detection(s)")
                print(f"     Confidences: {[f'{c:.3f}' for c in investigation['confidences']]}")
        
        if self.debug_stats['missed_investigations']:
            print(f"\nMissed investigations (regions investigated but no detection found, last 10):")
            for i, investigation in enumerate(self.debug_stats['missed_investigations'][-10:], 1):
                print(f"  {i}. Current: {investigation['current_image']}")
                print(f"     Source: {investigation['source_image']}")
                print(f"     Original bbox: {investigation['original_bbox']}")
                print(f"     Best IoU with current detections: {investigation['best_iou_with_current']:.3f}")
        
        print(f"Detection method: {'SAHI' if self.use_sahi else 'Regular YOLO'}")
        
        if self.use_sahi:
            print(f"SAHI Configuration:")
            print(f"  Slice size: {self.slice_width}x{self.slice_height}")
            print(f"  Overlap ratios: {self.overlap_width_ratio}x{self.overlap_height_ratio}")
            print(f"  Post-processing: {self.postprocess_type}")
            print(f"  Total tiles processed: {self.debug_stats['sahi_tiles_processed']}")
            
            if self.debug_stats['sahi_vs_regular_comparison']:
                print(f"\nSAHI vs Regular Comparison (last 5 images):")
                for comp in self.debug_stats['sahi_vs_regular_comparison'][-5:]:
                    print(f"  {comp['image_name']}: SAHI={comp['sahi_count']} (avg_conf={comp['sahi_avg_conf']:.3f}), "
                          f"Regular={comp['regular_count']} (avg_conf={comp['regular_avg_conf']:.3f})")
        
        print(f"\nSecond Stage Statistics:")
        print(f"Total regions investigated: {self.debug_stats['regions_investigated']}")
        print(f"Total regions skipped (invalid crop): {self.debug_stats['regions_skipped_invalid_crop']}")
        print(f"Total missed detections found: {self.debug_stats['missed_detections_found']}")
        print(f"Total missed investigations (no detection): {len(self.debug_stats['missed_investigations'])}")
        
        if self.debug_stats['source_images_used']:
            print("\nSource images that contributed to missed detections:")
            for img, count in sorted(self.debug_stats['source_images_used'].items()):
                print(f"  {img}: {count} detection(s)")
        
        if self.debug_stats['investigation_history']:
            print(f"\nSuccessful investigations ({len(self.debug_stats['investigation_history'])}):")
            for i, investigation in enumerate(self.debug_stats['investigation_history'][-10:], 1):
                print(f"  {i}. Current: {investigation['current_image']}")
                print(f"     Source: {investigation['source_image']}")
                print(f"     Found: {investigation['detections_found']} detection(s)")
                print(f"     Confidences: {[f'{c:.3f}' for c in investigation['confidences']]}")
        
        if self.debug_stats['missed_investigations']:
            print(f"\nMissed investigations (regions investigated but no detection found, last 10):")
            for i, investigation in enumerate(self.debug_stats['missed_investigations'][-10:], 1):
                print(f"  {i}. Current: {investigation['current_image']}")
                print(f"     Source: {investigation['source_image']}")
                print(f"     Original bbox: {investigation['original_bbox']}")
                print(f"     Best IoU with current detections: {investigation['best_iou_with_current']:.3f}")

def save_yolo_format_txt(detections, img_width, img_height, save_path):
    """
    Save detections in YOLO format txt file
    
    Args:
        detections (list): List of detection dictionaries
        img_width (int): Image width
        img_height (int): Image height
        save_path (str): Path to save txt file
    """
    with open(save_path, 'w') as f:
        for detection in detections:
            # Extract bbox coordinates [x, y, w, h]
            x, y, w, h = detection['bbox']
            
            # Convert to YOLO format (normalized center coordinates + width/height)
            center_x = (x + w/2.0) / img_width
            center_y = (y + h/2.0) / img_height
            width = w / img_width
            height = h / img_height
            
            # Use the actual predicted class instead of hardcoded 0
            class_id = detection['class']
            confidence = detection['confidence']
            
            # Write in YOLO format: class_id center_x center_y width height confidence
            f.write(f"{class_id} {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f} {confidence:.6f}\n")

def get_image_files(folder_path, extensions=['.jpg', '.jpeg', '.png', '.bmp', '.tiff']):
    """
    Get sorted list of image files from folder (no duplicates)
    
    Args:
        folder_path (str): Path to folder containing images
        extensions (list): List of valid image extensions
    """
    folder = Path(folder_path)
    image_files = set()  # Use set to prevent duplicates
    
    # Convert extensions to lowercase for comparison
    extensions_lower = [ext.lower() for ext in extensions]
    
    # Get all files and filter by extension (case-insensitive)
    for file_path in folder.iterdir():
        if file_path.is_file():
            file_ext = file_path.suffix.lower()
            if file_ext in extensions_lower:
                image_files.add(file_path)
    
    # Convert back to list and sort
    image_files = list(image_files)
    
    # Sort files naturally (handles numeric sorting correctly)
    def natural_sort_key(text):
        return [int(c) if c.isdigit() else c.lower() for c in re.split('([0-9]+)', str(text))]
    
    image_files.sort(key=natural_sort_key)
    return image_files

def draw_enhanced_detections(image, detections, ground_truth=None, save_path=None, 
                           main_eval_iou=0.5, bbox_eval_iou=0.5, ignore_class_mismatch=False):
    """
    Draw detections with:
    - Green for correct predictions, Red for wrong predictions
    - Solid lines for main model, dotted lines for crop model
    """
    img_vis = image.copy()
    
    # Evaluate predictions if ground truth is available
    if ground_truth is not None:
        evaluated_detections = evaluate_predictions(
            predictions=detections, 
            ground_truth=ground_truth, 
            main_eval_iou=main_eval_iou, 
            bbox_eval_iou=bbox_eval_iou, 
            ignore_class_mismatch=ignore_class_mismatch
        )
    else:
        evaluated_detections = []
        for det in detections:
            det_copy = det.copy()
            det_copy['is_correct'] = True
            evaluated_detections.append(det_copy)
    
    # Colors: Green for correct, Red for wrong
    correct_color = (0, 255, 0)  # Green
    wrong_color = (0, 0, 255)    # Red
    
    for det in evaluated_detections:
        bbox = det['bbox']
        conf = det['confidence']
        cls = det['class']
        source = det.get('source', 'main_model')
        is_correct = det.get('is_correct', True)
        
        x, y, w, h = bbox
        
        # Choose color based on correctness
        color = correct_color if is_correct else wrong_color
        
        # Draw based on model source
        if source in ['regular_main_model', 'sahi_main_model']:
            # Solid line for main model
            thickness = 2 
            cv2.rectangle(img_vis, (x, y), (x + w, y + h), color, thickness)
        else:
            # Dotted line for crop model
            thickness = 2 
            draw_dotted_rectangle(img_vis, (x, y), (x + w, y + h), color, thickness)
        
        # Create label
        if source == 'sahi_main_model':
            model_short = "SAHI"
        elif source == 'bbox_model':
            model_short = "CROP"
        else:   
            model_short = "MAIN" 

        label = f"{model_short} C{cls} {conf:.2f}"
        
        if 'matched_gt_iou' in det:
            label += f" IoU:{det['matched_gt_iou']:.2f}"
        elif 'iou' in det:
            label += f" IoU:{det['iou']:.2f}"
        
        # Draw text with background
        label_parts = [label[i:i+35] for i in range(0, len(label), 35)]
        
        for i, part in enumerate(label_parts):
            text_y = y - 10 - (i * 20)
            if text_y < 20:
                text_y = y + h + 20 + (i * 20)
            
            (text_width, text_height), _ = cv2.getTextSize(part, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(img_vis, (x, text_y - text_height - 2), 
                         (x + text_width, text_y + 2), (0, 0, 0), -1)
            cv2.putText(img_vis, part, (x, text_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    
    # Draw ground truth in white dashed lines 
    if ground_truth is not None:
        for gt in ground_truth:
            gt_bbox = gt['bbox']
            gt_x, gt_y, gt_w, gt_h = gt_bbox
            
            dash_length = 10
            # Draw dashed rectangle 
            for i in range(gt_x, gt_x + gt_w, dash_length * 2):
                cv2.line(img_vis, (i, gt_y), (min(i + dash_length, gt_x + gt_w), gt_y), (255, 255, 255), 1)
            for i in range(gt_x, gt_x + gt_w, dash_length * 2):
                cv2.line(img_vis, (i, gt_y + gt_h), (min(i + dash_length, gt_x + gt_w), gt_y + gt_h), (255, 255, 255), 1)
            for i in range(gt_y, gt_y + gt_h, dash_length * 2):
                cv2.line(img_vis, (gt_x, i), (gt_x, min(i + dash_length, gt_y + gt_h)), (255, 255, 255), 1)
            for i in range(gt_y, gt_y + gt_h, dash_length * 2):
                cv2.line(img_vis, (gt_x + gt_w, i), (gt_x + gt_w, min(i + dash_length, gt_y + gt_h)), (255, 255, 255), 1)
            
            cv2.putText(img_vis, f"GT C{gt['class']}", (gt_x, gt_y - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    if save_path:
        cv2.imwrite(save_path, img_vis)
    
    return img_vis

def draw_dotted_rectangle(img, pt1, pt2, color, thickness=1, gap=5):
    """Draw a dotted rectangle"""
    x1, y1 = pt1
    x2, y2 = pt2
    
    # Top line
    for x in range(x1, x2, gap * 2):
        cv2.line(img, (x, y1), (min(x + gap, x2), y1), color, thickness)
    # Bottom line  
    for x in range(x1, x2, gap * 2):
        cv2.line(img, (x, y2), (min(x + gap, x2), y2), color, thickness)
    # Left line
    for y in range(y1, y2, gap * 2):
        cv2.line(img, (x1, y), (x1, min(y + gap, y2)), color, thickness)
    # Right line
    for y in range(y1, y2, gap * 2):
        cv2.line(img, (x2, y), (x2, min(y + gap, y2)), color, thickness)

def load_ground_truth_labels(label_path, img_width, img_height):
    """
    Load ground truth labels from YOLO format txt file
    
    Args:
        label_path (str): Path to ground truth label file
        img_width (int): Image width
        img_height (int): Image height
    
    Returns:
        list: List of ground truth bounding boxes in [x, y, w, h] format
    """
    ground_truth = []
    
    if not os.path.exists(label_path):
        return ground_truth
    
    try:
        with open(label_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                parts = line.split()
                if len(parts) >= 5:
                    class_id = int(parts[0])
                    center_x = float(parts[1]) * img_width
                    center_y = float(parts[2]) * img_height
                    width = float(parts[3]) * img_width
                    height = float(parts[4]) * img_height
                    
                    # Convert to [x, y, w, h] format
                    x = center_x - width / 2
                    y = center_y - height / 2
                    
                    ground_truth.append({
                        'bbox': [int(x), int(y), int(width), int(height)],
                        'class': class_id
                    })
    except Exception as e:
        print(f"Error loading ground truth from {label_path}: {e}")
    
    return ground_truth

def evaluate_predictions(predictions, ground_truth, main_eval_iou=0.5, bbox_eval_iou=0.5,
                        ignore_class_mismatch=False):
    """
    Evaluate predictions against ground truth to determine correctness
   
    Args:
        predictions (list): List of prediction dictionaries
        ground_truth (list): List of ground truth dictionaries
        main_eval_iou (float): IoU threshold for main model evaluation
        bbox_eval_iou (float): IoU threshold for bbox model evaluation
        ignore_class_mismatch (bool): If True, ignore class mismatches when evaluating IoU
   
    Returns:
        list: List of predictions with 'is_correct' field added
    """
    evaluated_predictions = []
    gt_matched = [False] * len(ground_truth)
   
    sorted_predictions = sorted(predictions, key=lambda x: x['confidence'], reverse=True)
   
    for pred in sorted_predictions:
        pred_copy = pred.copy()
        pred_copy['is_correct'] = False
        best_iou = 0.0
        best_gt_idx = -1
        best_match_gt_idx = -1  # For tracking matches that meet threshold
       
        # Choose IoU threshold based on model source
        eval_iou_threshold = main_eval_iou if pred['source'] in ['regular_main_model', 'sahi_main_model'] else bbox_eval_iou
       
        for gt_idx, gt in enumerate(ground_truth):
            if gt_matched[gt_idx]:
                continue
           
            iou = calculate_iou_boxes(pred['bbox'], gt['bbox'])
           
            # Always track the best IoU regardless of threshold
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx
           
            # Check if this is a valid match (meets threshold and class requirements)
            if iou >= eval_iou_threshold:
                class_match = ignore_class_mismatch or (pred['class'] == gt['class'])
                if class_match and iou > (0.0 if best_match_gt_idx == -1 else calculate_iou_boxes(pred['bbox'], ground_truth[best_match_gt_idx]['bbox'])):
                    best_match_gt_idx = gt_idx

        # Always save the best IoU found
        pred_copy['matched_gt_iou'] = best_iou
        
        # Mark as correct only if we found a valid match
        if best_match_gt_idx >= 0:
            pred_copy['is_correct'] = True
            gt_matched[best_match_gt_idx] = True
        else:
            pred_copy['is_correct'] = False
       
        evaluated_predictions.append(pred_copy)
   
    return evaluated_predictions
def evaluate_predictions_bis(predictions, ground_truth, main_eval_iou=0.5, bbox_eval_iou=0.5, 
                        ignore_class_mismatch=False):
    """
    Evaluate predictions against ground truth to determine correctness
    
    Args:
        predictions (list): List of prediction dictionaries
        ground_truth (list): List of ground truth dictionaries
        main_eval_iou (float): IoU threshold for main model evaluation
        bbox_eval_iou (float): IoU threshold for bbox model evaluation
        ignore_class_mismatch (bool): If True, ignore class mismatches when evaluating IoU
    
    Returns:
        list: List of predictions with 'is_correct' field added
    """
    evaluated_predictions = []
    gt_matched = [False] * len(ground_truth)
    
    sorted_predictions = sorted(predictions, key=lambda x: x['confidence'], reverse=True)
    
    for pred in sorted_predictions:
        pred_copy = pred.copy()
        pred_copy['is_correct'] = False
        best_iou = 0.0
        best_gt_idx = -1
        
        # Choose IoU threshold based on model source
        eval_iou_threshold = main_eval_iou if pred['source'] in ['regular_main_model', 'sahi_main_model'] else bbox_eval_iou
        
        for gt_idx, gt in enumerate(ground_truth):
            if gt_matched[gt_idx]:
                continue
            
            iou = calculate_iou_boxes(pred['bbox'], gt['bbox'])
            
            if iou > best_iou and iou >= eval_iou_threshold:
                # Check class match unless ignoring class mismatch
                class_match = ignore_class_mismatch or (pred['class'] == gt['class'])
                if class_match:
                    best_iou = iou
                    best_gt_idx = gt_idx

            elif iou>best_iou and iou < eval_iou_threshold:
                # If we found a match but below threshold, still consider it
                if best_gt_idx == -1 or iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx

        if best_gt_idx >= 0:
            pred_copy['is_correct'] = True
            pred_copy['matched_gt_iou'] = best_iou
            gt_matched[best_gt_idx] = True
        else:
            pred_copy['is_correct'] = False
            pred_copy['matched_gt_iou'] = best_iou
        
        evaluated_predictions.append(pred_copy)
    
    return evaluated_predictions
""
def calculate_iou_boxes(box1, box2):
    """
    Calculate IoU between two bounding boxes [x, y, w, h]
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[0] + box1[2], box2[0] + box2[2])
    y2 = min(box1[1] + box1[3], box2[1] + box2[3])
    
    if x2 <= x1 or y2 <= y1:
        return 0.0
        
    intersection = (x2 - x1) * (y2 - y1)
    area1 = box1[2] * box1[3]
    area2 = box2[2] * box2[3]
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0

def extract_date_from_filename(filename):
    """
    Extract date from filename like '7s3a5abm_2024-06-25_130143_8_10.jpg'
    Returns date string in format 'YYYY-MM-DD' or None if not found
    """
    # Pattern to match date in format YYYY-MM-DD
    date_pattern = r'(\d{4}-\d{2}-\d{2})'
    match = re.search(date_pattern, filename)
    if match:
        return match.group(1)
    return None

def save_detection_crop(image, detection, crop_save_dir, image_name, detection_idx):
    """
    Save a crop of the detection region
    
    Args:
        image: Original image
        detection: Detection dictionary with bbox info
        crop_save_dir: Directory to save crops
        image_name: Name of the source image
        detection_idx: Index of this detection in the image
    
    Returns:
        str: Filename of saved crop
    """
    bbox = detection['bbox']
    x, y, w, h = bbox
    
    # Ensure crop coordinates are within image bounds
    img_h, img_w = image.shape[:2]
    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(img_w, x + w)
    y2 = min(img_h, y + h)
    
    # Extract crop
    crop = image[y1:y2, x1:x2]
    
    if crop.size == 0:
        return None
    
    # Generate crop filename
    source = detection['source']
    confidence = detection['confidence']
    crop_filename = f"{image_name}_det{detection_idx:03d}_{source}_conf{confidence:.3f}.jpg"
    crop_path = Path(crop_save_dir) / crop_filename
    
    # Save crop
    cv2.imwrite(str(crop_path), crop)
    
    return crop_filename

def process_image_folder_with_crops_and_csv(folder_path, main_model_path, bbox_model_path=None, 
                                          ground_truth_folder=None, output_dir=None, 
                                          save_visualizations=True, save_txt_files=True, 
                                          save_crops=True, memory_window=3, bbox_model_conf=0.3, expansion_factor=1.3, 
                                          main_model_size=640, main_model_conf=0.2, main_model_iou=0.1, 
                                          bbox_model_size=640, bbox_model_iou=0.1, global_iou_threshold=0.3,
                                          main_eval_iou=0.5, bbox_eval_iou=0.5, 
                                          use_sahi=False, slice_height=640, slice_width=640, 
                                          overlap_height_ratio=0.2, overlap_width_ratio=0.2,
                                          ignore_class_mismatch=False, target_size=640, debug=False):
    """
    Enhanced version that saves crops and generates CSV with detection information
    """
 
    # Initialize detector
    detector = TwoStageGrapeDetector(
        main_model_path=main_model_path,
        bbox_model_path=bbox_model_path,
        debug=debug,
        main_model_size=main_model_size,
        main_model_conf=main_model_conf,
        bbox_model_size=bbox_model_size,
        bbox_model_iou=bbox_model_iou,
        main_model_iou=main_model_iou,
        memory_window=memory_window,
        global_iou_threshold=global_iou_threshold,
        use_sahi=use_sahi,
        slice_height=slice_height,
        slice_width=slice_width,
        overlap_height_ratio=overlap_height_ratio,
        overlap_width_ratio=overlap_width_ratio,
        target_size=target_size,
        expansion_factor=expansion_factor
    )

    # Get sorted image files
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    image_files = sorted([f for f in Path(folder_path).iterdir() 
                         if f.suffix.lower() in image_extensions])
    
    if len(image_files) == 0:
        print(f"No image files found in {folder_path}")
        return
    
    print(f"Found {len(image_files)} images to process")
    
    # Setup output directory
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        if save_visualizations:
            vis_path = output_path / 'visualizations'
            vis_path.mkdir(exist_ok=True)
            
        if save_txt_files:
            txt_path = output_path / 'labels'
            txt_path.mkdir(exist_ok=True)
            
        if save_crops:
            crops_path = output_path / 'crops'
            crops_path.mkdir(exist_ok=True)
    
    # Initialize CSV data collection
    csv_data = []
    
    # Process images
    all_results = {}
    processed_image_names = []
    
    for i, img_file in enumerate(image_files):
        print(f"\nProcessing {i+1}/{len(image_files)}: {img_file.name}")
        
        # Load image
        image = cv2.imread(str(img_file))
        image_path = os.path.join(img_file)
        if image is None:
            print(f"Warning: Could not load {img_file}")
            continue
        
        img_height, img_width = image.shape[:2]
        image_name = f"{img_file.stem}_{i:04d}"
        
        # Extract date from filename
        date_str = extract_date_from_filename(img_file.name)
        if not date_str:
            print(f"Warning: Could not extract date from {img_file.name}")
            date_str = "unknown"
        
        
        # Load ground truth if available
        ground_truth = None
        if ground_truth_folder:
            gt_file = Path(ground_truth_folder) / f"{img_file.stem}.txt"
            if gt_file.exists():
                ground_truth = load_ground_truth_labels(gt_file, img_width, img_height)
                if ground_truth:
                    print(f"  Loaded {len(ground_truth)} ground truth boxes from {gt_file}")
                else:
                    print(f"  No valid ground truth found in {gt_file}")
                
        
        # Run detection
        detections = detector.detect_two_stage(
            image, image_path, image_name, processed_image_names,
            memory_window=memory_window,
            #iou_threshold=iou_threshold,
            global_iou_threshold=global_iou_threshold,
            bbox_model_conf=bbox_model_conf, 
            main_model_conf=main_model_conf
        )
        
        processed_image_names.append(image_name)
        
        # Process each detection
        for det_idx, detection in enumerate(detections):
            bbox = detection['bbox']
            x, y, w, h = bbox
            
            # Calculate center coordinates (normalized)
            x_center = (x + w/2) / img_width
            y_center = (y + h/2) / img_height
            
            # Normalize width and height
            norm_width = w / img_width
            norm_height = h / img_height
            
            # Save crop if enabled
            crop_filename = None
            if save_crops and output_dir:
                crop_filename = save_detection_crop(
                    image, detection, crops_path, image_name, det_idx
                )
            
            # Map source to model name
            print(detection['source'])
            if detection['source'] == 'regular_main_model':
                model_used = "main"
            elif detection['source'] == 'sahi_main_model':
                model_used = "main_sahi"
            elif detection['source'] == 'bbox_model':
                model_used = "crop"
            
            # Add to CSV data
            csv_row = {
                'date': date_str,
                'class_predicted': detection['class'],
                'model_used': model_used, 
                'confidence': detection['confidence'],
                'height': norm_height,
                'width': norm_width,
                'x_center': x_center,
                'y_center': y_center,
                'crop_image_name': crop_filename if crop_filename else "not_saved"
            }
            csv_data.append(csv_row)
        
        # Count detections by source
        main_count = len([d for d in detections if d['source'] == 'regular_main_model'])
        sahi_count = len([d for d in detections if d['source'] == 'sahi_main_model'])
        bbox_count = len([d for d in detections if d['source'] == 'bbox_model'])
        
        print(f"  Found {len(detections)} detections (main: {main_count}, bbox: {bbox_count})")
        
        # Store results
        all_results[img_file.name] = {
            'file_path': str(img_file),
            'internal_name': image_name,
            'date': date_str,
            'total_detections': len(detections),
            'main_model_detections': main_count,
            'bbox_model_detections': bbox_count,
            'detections': detections,
            'has_ground_truth': ground_truth is not None,
            'ground_truth_count': len(ground_truth) if ground_truth else 0
        }
        
        # Save visualization
        if save_visualizations and output_dir :
            vis_save_path = output_path / 'visualizations' / img_file.name
            vis_img = draw_enhanced_detections(
                image, detections, ground_truth=ground_truth, save_path=str(vis_save_path),
                main_eval_iou=main_eval_iou, bbox_eval_iou=bbox_eval_iou,
                ignore_class_mismatch=ignore_class_mismatch
            )

        # Save txt file with predictions in YOLO format
        if save_txt_files and output_dir:
            txt_filename = img_file.stem + '.txt'
            txt_save_path = output_path / 'labels' / txt_filename
            save_yolo_format_txt(detections, img_width, img_height, str(txt_save_path))
    
    # Save CSV file
    if output_dir and csv_data:
        csv_df = pd.DataFrame(csv_data)
        csv_path = output_path / 'detections_summary.csv'
        csv_df.to_csv(csv_path, index=False)
        print(f"\nCSV summary saved to {csv_path}")
        print(f"Total detections recorded: {len(csv_data)}")
        
        # Print CSV summary
        if len(csv_data) > 0:
            print(f"\nCSV Summary:")
            print(f"Date range: {csv_df['date'].min()} to {csv_df['date'].max()}")
            print(f"Model usage: Main={len(csv_df[csv_df['model_used'] == 'main'])}, Crop={len(csv_df[csv_df['model_used'] == 'crop'])}")
            print(f"Classes detected: {sorted(csv_df['class_predicted'].unique())}")
            print(f"Confidence range: {csv_df['confidence'].min():.3f} to {csv_df['confidence'].max():.3f}")
    
    # Print debug summary
    if debug:
        detector.print_debug_summary()
    
    # Save results to JSON
    if output_dir:
        results_path = output_path / 'detection_results.json'
        with open(results_path, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f"Results saved to {results_path}")
        
        if debug:
            debug_path = output_path / 'debug_stats.json'
            with open(debug_path, 'w') as f:
                json.dump(detector.debug_stats, f, indent=2)
            print(f"Debug stats saved to {debug_path}")
    
    return all_results

def save_yolo_format_txt(detections, img_width, img_height, txt_path):
    """
    Save detections in YOLO format
    Implement this based on your existing code
    """
    with open(txt_path, 'w') as f:
        for detection in detections:
            bbox = detection['bbox']
            x, y, w, h = bbox
            class_id = detection['class']
            
            # Convert to YOLO format (normalized center coordinates)
            x_center = (x + w/2) / img_width
            y_center = (y + h/2) / img_height
            norm_width = w / img_width
            norm_height = h / img_height
            
            f.write(f"{class_id} {x_center} {y_center} {norm_width} {norm_height}\n")

def get_image_files(folder_path):
    """
    Get sorted list of image files
    Implement this based on your existing code
    """
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    folder = Path(folder_path)
    return sorted([f for f in folder.iterdir() if f.suffix.lower() in image_extensions])

def is_center_inside_box(center_x, center_y, box_x, box_y, box_width, box_height):
    """
    Check if a center point is inside a bounding box.
    All coordinates are normalized (0-1).
    """
    # Calculate box boundaries
    left = box_x - box_width / 2
    right = box_x + box_width / 2
    top = box_y - box_height / 2
    bottom = box_y + box_height / 2
    
    # Check if center is inside the box
    return (left <= center_x <= right) and (top <= center_y <= bottom)

def assign_tracking_ids(csv_file_path, save_to_original=True):
    """
    Assign tracking IDs to predictions based on spatial overlap across dates.
    
    Parameters:
    csv_file_path (str): Path to the input CSV file
    save_to_original (bool): If True, saves the result back to the original file
    
    Returns:
    pandas.DataFrame: DataFrame with tracking IDs assigned
    """
    # Read the CSV file
    df = pd.read_csv(csv_file_path)
    
    # Convert date column to datetime for proper sorting
    df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')
    
    # Sort by date to process chronologically
    df = df.sort_values('date').reset_index(drop=True)
    
    # Initialize tracking ID column
    df['tracking_id'] = -1
    
    # Keep track of existing objects and their last known positions
    active_objects = {}  # {tracking_id: {'x': x_center, 'y': y_center, 'width': width, 'height': height, 'last_date': date}}
    next_id = 1
    
    # Get unique dates
    unique_dates = df['date'].unique()
    
    for date in unique_dates:
        # Get all predictions for current date
        current_predictions = df[df['date'] == date].copy()
        
        # If this is the first date, assign new IDs to all predictions
        if len(active_objects) == 0:
            for idx in current_predictions.index:
                df.loc[idx, 'tracking_id'] = next_id
                active_objects[next_id] = {
                    'x': df.loc[idx, 'x_center'],
                    'y': df.loc[idx, 'y_center'],
                    'width': df.loc[idx, 'width'],
                    'height': df.loc[idx, 'height'],
                    'last_date': date
                }
                next_id += 1
        else:
            # For subsequent dates, try to match with existing objects
            assigned_ids = set()
            
            for idx in current_predictions.index:
                current_x = df.loc[idx, 'x_center']
                current_y = df.loc[idx, 'y_center']
                
                # Check if current center is inside any existing object's bounding box
                matched_id = None
                for obj_id, obj_data in active_objects.items():
                    if obj_id not in assigned_ids:  # Avoid double assignment
                        if is_center_inside_box(current_x, current_y, 
                                              obj_data['x'], obj_data['y'], 
                                              obj_data['width'], obj_data['height']):
                            matched_id = obj_id
                            break
                
                if matched_id is not None:
                    # Assign existing ID
                    df.loc[idx, 'tracking_id'] = matched_id
                    assigned_ids.add(matched_id)
                    
                    # Update object's position
                    active_objects[matched_id] = {
                        'x': current_x,
                        'y': current_y,
                        'width': df.loc[idx, 'width'],
                        'height': df.loc[idx, 'height'],
                        'last_date': date
                    }
                else:
                    # Create new ID for unmatched prediction
                    df.loc[idx, 'tracking_id'] = next_id
                    active_objects[next_id] = {
                        'x': current_x,
                        'y': current_y,
                        'width': df.loc[idx, 'width'],
                        'height': df.loc[idx, 'height'],
                        'last_date': date
                    }
                    next_id += 1
    
    # Convert date back to original format for output
    df['date'] = df['date'].dt.strftime('%d/%m/%Y')
    
    # Save back to original file if requested
    if save_to_original:
        df.to_csv(csv_file_path, index=False)
        print(f"Tracking IDs added to original file: {csv_file_path}")
    
    return df

def organize_crop_images(csv_file_path, images_source_dir, output_base_dir="tracking_folders"):
    """
    Create folders for each tracking ID and organize crop images accordingly.
    
    Parameters:
    csv_file_path (str): Path to the CSV file with tracking IDs
    images_source_dir (str): Directory containing the crop images
    output_base_dir (str): Base directory where tracking folders will be created
    """
    # Read the CSV file
    df = pd.read_csv(csv_file_path)
    
    # Check if tracking_id column exists
    if 'tracking_id' not in df.columns:
        print("Error: tracking_id column not found. Please run assign_tracking_ids first.")
        return
    
    # Create base output directory
    output_path = Path(output_base_dir)
    output_path.mkdir(exist_ok=True)
    
    # Get unique tracking IDs
    unique_ids = sorted(df['tracking_id'].unique())
    
    print(f"Creating folders for {len(unique_ids)} unique tracking IDs...")
    
    # Create folders for each tracking ID
    for track_id in unique_ids:
        folder_path = output_path / f"tracking_id_{track_id:03d}"
        folder_path.mkdir(exist_ok=True)
    
    # Copy images to respective folders
    images_source_path = Path(images_source_dir)
    copied_count = 0
    missing_count = 0
    
    for idx, row in df.iterrows():
        track_id = row['tracking_id']
        image_name = row['crop_image_name']
        
        # Source and destination paths
        source_image_path = images_source_path / image_name
        dest_folder = output_path / f"tracking_id_{track_id:03d}"
        dest_image_path = dest_folder / image_name
        
        # Copy image if it exists
        if source_image_path.exists():
            try:
                shutil.copy2(source_image_path, dest_image_path)
                copied_count += 1
            except Exception as e:
                print(f"Error copying {image_name}: {e}")
        else:
            print(f"Warning: Image not found: {source_image_path}")
            missing_count += 1
    
    print(f"\nImage organization complete!")
    print(f"Successfully copied: {copied_count} images")
    print(f"Missing images: {missing_count}")
    print(f"Output directory: {output_path.absolute()}")
    
    # Create summary report
    create_tracking_summary(df, output_path)

def create_tracking_summary(df, output_path):
    """
    Create a summary report of tracking results.
    """
    summary_file = output_path / "tracking_summary.txt"
    
    with open(summary_file, 'w') as f:
        f.write("=== TRACKING SUMMARY REPORT ===\n\n")
        f.write(f"Total predictions: {len(df)}\n")
        f.write(f"Unique tracking IDs: {df['tracking_id'].nunique()}\n")
        f.write(f"Date range: {df['date'].min()} to {df['date'].max()}\n\n")
        
        # Tracking ID details
        f.write("TRACKING ID DETAILS:\n")
        f.write("-" * 50 + "\n")
        
        id_counts = df['tracking_id'].value_counts().sort_index()
        for track_id, count in id_counts.items():
            track_data = df[df['tracking_id'] == track_id]
            dates = sorted(track_data['date'].unique())
            classes = track_data['class_predicted'].value_counts().to_dict()
            avg_confidence = track_data['confidence'].mean()
            
            f.write(f"ID {track_id:03d}:\n")
            f.write(f"  - Total detections: {count}\n")
            f.write(f"  - Active dates: {len(dates)} ({dates[0]} to {dates[-1]})\n")
            f.write(f"  - Average confidence: {avg_confidence:.3f}\n")
            f.write(f"  - Class distribution: {classes}\n")
            f.write(f"  - Images: {', '.join(track_data['crop_image_name'].tolist())}\n\n")
        
        # Date-wise summary
        f.write("DATE-WISE SUMMARY:\n")
        f.write("-" * 50 + "\n")
        date_counts = df['date'].value_counts().sort_index()
        for date, count in date_counts.items():
            unique_ids = df[df['date'] == date]['tracking_id'].nunique()
            f.write(f"{date}: {count} predictions, {unique_ids} unique objects\n")
    
    print(f"Summary report saved to: {summary_file}")

def investigate_and_organize(csv_file_path, images_source_dir, output_base_dir="tracking_folders"):
    """
    Complete workflow: assign tracking IDs and organize images.
    
    Parameters:
    csv_file_path (str): Path to the CSV file
    images_source_dir (str): Directory containing the crop images  
    output_base_dir (str): Base directory where tracking folders will be created
    """
    print("Step 1: Assigning tracking IDs...")
    # First assign tracking IDs
    df = assign_tracking_ids(csv_file_path)
    
    print("\nStep 2: Organizing crop images...")
    # Then organize the images
    organize_crop_images(csv_file_path, images_source_dir, output_base_dir)
    
    return df
def analyze_tracking_results(df):
    """
    Analyze the tracking results and print summary statistics.
    """
    print("=== Tracking Analysis ===")
    print(f"Total predictions: {len(df)}")
    print(f"Unique tracking IDs: {df['tracking_id'].nunique()}")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    
    # Show tracking ID distribution
    id_counts = df['tracking_id'].value_counts().sort_index()
    print(f"\nTracking ID distribution:")
    for track_id, count in id_counts.items():
        dates = df[df['tracking_id'] == track_id]['date'].unique()
        print(f"  ID {track_id}: {count} detections across {len(dates)} dates")
    
    # Show predictions by date
    print(f"\nPredictions by date:")
    date_counts = df['date'].value_counts().sort_index()
    for date, count in date_counts.items():
        unique_ids = df[df['date'] == date]['tracking_id'].nunique()
        print(f"  {date}: {count} predictions, {unique_ids} unique objects")

def main():
    parser = argparse.ArgumentParser(description='Two-stage grape detection on image folder')
    parser.add_argument('--folder', required=True, help='Path to folder containing images')
    parser.add_argument('--main_model', required=True, help='Path to main YOLO11 model')
    parser.add_argument('--main_model_size', help='Input size for main model (default: 640)', type=int, default=640)
    parser.add_argument('--bbox_model', help='Path to bbox YOLO11 model (optional)')
    parser.add_argument('--output', help='Output directory for results')
    parser.add_argument('--main_model_conf', type=float, default=0.2, help='Confidence threshold for main model (default: 0.2)')
    parser.add_argument('--main_model_iou', type=float, default=0.1, help='IoU threshold for main model (default: 0.1)')
    parser.add_argument('--global_iou_threshold', type=float, default=0.3, help='Global confidence threshold for all detections (default: 0.2)')
    parser.add_argument('--memory_window', type=int, default=3, help='Number of previous images to remember (try 5-10 for better recall)')
    parser.add_argument('--bbox_model_conf', type=float, default=0.15, help='Min confidence for bbox model detections (try 0.2-0.4)')
    parser.add_argument('--expansion_factor', type=float, default=1.3, help='Bbox expansion factor (try 1.5-2.0 for larger search areas)')
    parser.add_argument('--no_vis', action='store_true', help='Skip saving visualizations')
    parser.add_argument('--debug', action='store_true', help='Enable detailed debug output')
    parser.add_argument('--use_sahi', action='store_true', help='Use SAHI for slicing and dicing images')
    parser.add_argument('--slice_height', type=int, default=640, help='Height of slices for SAHI (default: 640)')
    parser.add_argument('--slice_width', type=int, default=640, help='Width of slices for SAHI (default: 640)')
    parser.add_argument('--overlap_height_ratio', type=float, default=0.2, help='Overlap height ratio for SAHI (default: 0.2)')
    parser.add_argument('--overlap_width_ratio', type=float, default=0.2, help='Overlap width ratio for SAHI (default: 0.2)')
    parser.add_argument('--ground_truth_folder', help='Path to folder containing ground truth label files (optional)')
    parser.add_argument('--bbox_model_size', type=int, default=640, help='Input size for bbox model (default: 640)')
    parser.add_argument('--bbox_model_iou', type=float, default=0.1, help='IoU threshold for bbox model NMS (default: 0.1)')
    parser.add_argument('--main_eval_iou', type=float, default=0.5, help='IoU threshold for evaluating main model (default: 0.5)')
    parser.add_argument('--bbox_eval_iou', type=float, default=0.5, help='IoU threshold for evaluating bbox model (default: 0.5)')
    parser.add_argument('--ignore_class_mismatch', action='store_true', help='Ignore class mismatch when evaluating correctness')
    parser.add_argument('--target_size', type=int, default=640, help='Target size for bbox model input (default: 640)')
    parser.add_argument('--no_crops', action='store_true', help='Skip saving crop images')
    
    args = parser.parse_args()

    results = process_image_folder_with_crops_and_csv(
        folder_path=args.folder,
        main_model_path=args.main_model,
        bbox_model_path=args.bbox_model,
        ground_truth_folder=args.ground_truth_folder,
        main_model_size=args.main_model_size,
        main_model_conf=args.main_model_conf,
        main_model_iou=args.main_model_iou,
        target_size=args.target_size,
        global_iou_threshold=args.global_iou_threshold,
        output_dir=args.output,
        save_visualizations=not args.no_vis,
        save_crops=not args.no_crops,
        memory_window=args.memory_window,
        save_txt_files=True,
        bbox_model_conf=args.bbox_model_conf,
        expansion_factor=args.expansion_factor,
        main_eval_iou=args.main_eval_iou,
        bbox_eval_iou=args.bbox_eval_iou,
        bbox_model_size=args.bbox_model_size,  
        bbox_model_iou=args.bbox_model_iou,     
        ignore_class_mismatch=args.ignore_class_mismatch,  
        use_sahi=args.use_sahi,
        slice_height=args.slice_height,
        slice_width=args.slice_width,
        overlap_height_ratio=args.overlap_height_ratio,
        overlap_width_ratio=args.overlap_width_ratio,
        debug=args.debug
    )

    input_file = args.output + "/detections_summary.csv"  # Replace with your CSV file path
    images_directory = args.output + "/crops"  # Replace with your images directory path
    output_directory = args.output + "/id_tracking_results"  # Output directory for organized folders
    
    try:
        # Option 1: Complete workflow (assign IDs + organize images)
        print("=== COMPLETE WORKFLOW ===")
        result_df = investigate_and_organize(input_file, images_directory, output_directory)
        
        # Option 2: If you only want to assign tracking IDs
        # print("=== ASSIGNING TRACKING IDs ONLY ===")
        # result_df = assign_tracking_ids(input_file)
        
        # Option 3: If you already have tracking IDs and only want to organize images
        # print("=== ORGANIZING IMAGES ONLY ===")
        # organize_crop_images(input_file, images_directory, output_directory)
        
        # Display sample results
        print("\nSample of results:")
        print(result_df[['date', 'class_predicted', 'confidence', 'x_center', 'y_center', 'tracking_id']].head(10))
        
        # Analyze tracking results
        analyze_tracking_results(result_df)
        
    except Exception as e:
        print(f"Error processing: {e}")
        print("Make sure file paths are correct and the file format matches the expected structure.")


if __name__ == "__main__":
    main()