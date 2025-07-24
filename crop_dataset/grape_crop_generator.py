import cv2
import numpy as np
import os
from pathlib import Path
import random
from typing import List, Tuple

class GrapeCropGenerator:
    def __init__(self, 
                 images_dir: str,
                 labels_dir: str,
                 output_dir: str,
                 crop_size: int = 160,
                 expansion_factor: float = 1.4,
                 negative_ratio: float = 0.25):
        
        self.images_dir = Path(images_dir)
        self.labels_dir = Path(labels_dir)
        self.output_dir = Path(output_dir)
        self.crop_size = crop_size
        self.expansion_factor = expansion_factor
        self.negative_ratio = negative_ratio
        
        # Create output directories
        (self.output_dir / 'images').mkdir(parents=True, exist_ok=True)
        (self.output_dir / 'labels').mkdir(parents=True, exist_ok=True)
        
        self.crop_counter = 0
    
    def parse_yolo_labels(self, label_file: str) -> List[List[float]]:
        """Parse YOLO format labels: [class_id, x_center, y_center, width, height]"""
        annotations = []
        if os.path.exists(label_file):
            with open(label_file, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        class_id = int(parts[0])
                        if class_id == 1:
                            continue  # Skip class_id 1
                        bbox = [float(x) for x in parts[1:5]]
                        annotations.append([class_id] + bbox)
        return annotations
    
    def expand_bbox(self, bbox: List[float], img_shape: Tuple[int, int]) -> Tuple[int, int, int, int]:
        """
        Expand bounding box for context
        bbox: [x_center, y_center, width, height] (normalized 0-1)
        Returns: (x1, y1, width, height) in pixels
        """
        img_h, img_w = img_shape
        
        # Convert to pixel coordinates
        x_center = bbox[0] * img_w
        y_center = bbox[1] * img_h
        box_w = bbox[2] * img_w
        box_h = bbox[3] * img_h
        
        # Expand dimensions
        new_w = max(box_w * self.expansion_factor, self.crop_size * 0.3)  # Minimum 30% of crop size
        new_h = max(box_h * self.expansion_factor, self.crop_size * 0.3)
        
        # Calculate top-left corner
        x1 = max(0, int(x_center - new_w/2))
        y1 = max(0, int(y_center - new_h/2))
        
        # Ensure bbox stays within image bounds
        x1 = min(x1, img_w - int(new_w))
        y1 = min(y1, img_h - int(new_h))
        x2 = min(x1 + int(new_w), img_w)
        y2 = min(y1 + int(new_h), img_h)
        
        return x1, y1, x2 - x1, y2 - y1
    
    def letterbox_resize(self, image: np.ndarray) -> Tuple[np.ndarray, float, Tuple[int, int]]:
        """
        Resize image to crop_size x crop_size with letterboxing
        Returns: (padded_image, scale_factor, (pad_x, pad_y))
        """
        h, w = image.shape[:2]
        
        # Calculate scaling factor
        scale = min(self.crop_size / w, self.crop_size / h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        # Resize image
        resized = cv2.resize(image, (new_w, new_h))
        
        # Create padded image (gray background)
        padded = np.full((self.crop_size, self.crop_size, 3), 114, dtype=np.uint8)
        
        # Calculate padding offsets
        pad_x = (self.crop_size - new_w) // 2
        pad_y = (self.crop_size - new_h) // 2
        
        # Place resized image in center
        padded[pad_y:pad_y+new_h, pad_x:pad_x+new_w] = resized
        
        return padded, scale, (pad_x, pad_y)
    
    def transform_bbox_to_crop(self, bbox: List[float], 
                              crop_region: Tuple[int, int, int, int],
                              scale: float, 
                              pad_offset: Tuple[int, int],
                              original_img_shape: Tuple[int, int]) -> List[float]:
        """
        Transform original bbox coordinates to crop coordinates
        """
        img_h, img_w = original_img_shape
        crop_x, crop_y, crop_w, crop_h = crop_region
        pad_x, pad_y = pad_offset
        
        # Convert bbox to pixel coordinates in original image
        x_center = bbox[0] * img_w
        y_center = bbox[1] * img_h
        box_w = bbox[2] * img_w
        box_h = bbox[3] * img_h
        
        # Check if bbox center is within crop region
        if (crop_x <= x_center <= crop_x + crop_w and 
            crop_y <= y_center <= crop_y + crop_h):
            
            # Transform to crop coordinates
            crop_x_center = x_center - crop_x
            crop_y_center = y_center - crop_y
            
            # Apply scaling and padding
            new_x_center = (crop_x_center * scale + pad_x) / self.crop_size
            new_y_center = (crop_y_center * scale + pad_y) / self.crop_size
            new_w = (box_w * scale) / self.crop_size
            new_h = (box_h * scale) / self.crop_size
            
            # Ensure bbox is within bounds [0, 1]
            if (0 <= new_x_center <= 1 and 0 <= new_y_center <= 1 and
                new_w > 0 and new_h > 0):
                return [new_x_center, new_y_center, new_w, new_h]
        
        return None
    
    def generate_positive_crop(self, image: np.ndarray, 
                              target_bbox: List[float],
                              all_bboxes: List[List[float]],
                              img_name: str,
                              bbox_idx: int) -> None:
        """Generate positive crop centered on target bbox"""
        
        # Expand target bbox
        crop_x, crop_y, crop_w, crop_h = self.expand_bbox(target_bbox[1:], image.shape[:2])
        
        # Extract crop
        crop = image[crop_y:crop_y+crop_h, crop_x:crop_x+crop_w]
        
        # Resize with letterboxing
        resized_crop, scale, pad_offset = self.letterbox_resize(crop)
        
        # Transform all bboxes that fall within this crop
        crop_labels = []
        crop_region = (crop_x, crop_y, crop_w, crop_h)
        
        for bbox in all_bboxes:
            transformed_bbox = self.transform_bbox_to_crop(
                bbox[1:], crop_region, scale, pad_offset, image.shape[:2]
            )
            if transformed_bbox:
                crop_labels.append([bbox[0]] + transformed_bbox)
        
        # Save crop and labels
        self.crop_counter += 1
        crop_name = f"{img_name}_crop_{self.crop_counter:04d}"
        
        # Save image
        img_path = self.output_dir / 'images' / f"{crop_name}.jpg"
        cv2.imwrite(str(img_path), resized_crop)
        
        # Save labels
        label_path = self.output_dir / 'labels' / f"{crop_name}.txt"
        with open(label_path, 'w') as f:
            for label in crop_labels:
                class_id = int(label[0])
                bbox_str = ' '.join([f"{x:.6f}" for x in label[1:]])
                f.write(f"{class_id} {bbox_str}\n")
        
        return crop_name
    
    def generate_negative_crop(self, image: np.ndarray, 
                              all_bboxes: List[List[float]],
                              img_name: str) -> None:
        """Generate negative crop (no grapes)"""
        
        img_h, img_w = image.shape[:2]
        max_attempts = 50
        
        for attempt in range(max_attempts):
            # Random crop size around target size
            crop_w = random.randint(int(self.crop_size * 0.2), int(self.crop_size * 0.8))
            crop_h = random.randint(int(self.crop_size * 0.2), int(self.crop_size * 0.8))
            
            # Random position
            if crop_w >= img_w or crop_h >= img_h:
                continue
                
            crop_x = random.randint(0, img_w - crop_w)
            crop_y = random.randint(0, img_h - crop_h)
            
            # Check overlap with any grape bbox
            crop_center_x = (crop_x + crop_w/2) / img_w
            crop_center_y = (crop_y + crop_h/2) / img_h
            
            # Calculate minimum distance to any grape
            min_distance = float('inf')
            for bbox in all_bboxes:
                grape_x, grape_y = bbox[1], bbox[2]
                distance = np.sqrt((crop_center_x - grape_x)**2 + (crop_center_y - grape_y)**2)
                min_distance = min(min_distance, distance)
            
            # If far enough from grapes, use this crop
            if min_distance > 0.15:  # Threshold distance (normalized)
                crop = image[crop_y:crop_y+crop_h, crop_x:crop_x+crop_w]
                resized_crop, _, _ = self.letterbox_resize(crop)
                
                # Save negative crop
                self.crop_counter += 1
                crop_name = f"{img_name}_neg_{self.crop_counter:04d}"
                
                img_path = self.output_dir / 'images' / f"{crop_name}.jpg"
                cv2.imwrite(str(img_path), resized_crop)
                
                # Create empty label file
                label_path = self.output_dir / 'labels' / f"{crop_name}.txt"
                open(label_path, 'w').close()
                
                return crop_name
        
        return None
    
    def process_dataset(self):
        """Process entire dataset"""
        
        # Get all image files - FIXED VERSION
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
        image_files = []
        for ext in image_extensions:
            image_files.extend(list(self.images_dir.glob(ext)))
            image_files.extend(list(self.images_dir.glob(ext.upper())))
        
        # Remove duplicates by converting to set and back to list
        image_files = list(set(image_files))
        
        print(f"Found {len(image_files)} images to process")
        
        total_positive_crops = 0
        total_negative_crops = 0
        
        for img_file in image_files:
            print(f"Processing {img_file.name}...")
            
            # Load image
            image = cv2.imread(str(img_file))
            if image is None:
                print(f"  Warning: Could not load {img_file}")
                continue
            
            # Load corresponding labels
            label_file = self.labels_dir / f"{img_file.stem}.txt"
            annotations = self.parse_yolo_labels(str(label_file))
            
            if not annotations:
                print(f"  No annotations found for {img_file.name}")
                print(f"  Looking for: {label_file}")  # Debug line
                continue
            
            img_positive_crops = 0
            img_negative_crops = 0
            
            # Generate positive crops (one per bbox)
            for i, bbox in enumerate(annotations):
                crop_name = self.generate_positive_crop(
                    image, bbox, annotations, img_file.stem, i
                )
                if crop_name:
                    img_positive_crops += 1
            
            # Generate negative crops
            num_negatives = max(1, int(len(annotations) * self.negative_ratio))
            for _ in range(num_negatives):
                crop_name = self.generate_negative_crop(
                    image, annotations, img_file.stem
                )
                if crop_name:
                    img_negative_crops += 1
            
            total_positive_crops += img_positive_crops
            total_negative_crops += img_negative_crops
            
            print(f"  Generated {img_positive_crops} positive, {img_negative_crops} negative crops")
        
        print(f"\nDataset generation complete!")
        print(f"Total positive crops: {total_positive_crops}")
        print(f"Total negative crops: {total_negative_crops}")
        print(f"Total crops: {total_positive_crops + total_negative_crops}")
        print(f"Negative ratio: {total_negative_crops/(total_positive_crops + total_negative_crops):.2%}")
        print(f"Crop size: {self.crop_size}x{self.crop_size}")
        print(f"Output directory: {self.output_dir}")

    def debug_file_paths(self):
        """Debug helper to check file paths and existence"""
        
        print(f"Images directory: {self.images_dir}")
        print(f"Labels directory: {self.labels_dir}")
        print(f"Images directory exists: {self.images_dir.exists()}")
        print(f"Labels directory exists: {self.labels_dir.exists()}")
        print()
        
        # Get all image files
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
        image_files = []
        for ext in image_extensions:
            image_files.extend(list(self.images_dir.glob(ext)))
            image_files.extend(list(self.images_dir.glob(ext.upper())))
        
        # Remove duplicates
        image_files = list(set(image_files))
        
        print(f"Found {len(image_files)} image files:")
        for img_file in image_files:
            print(f"  Image: {img_file.name}")
            
            # Check corresponding label file
            label_file = self.labels_dir / f"{img_file.stem}.txt"
            print(f"  Looking for label: {label_file}")
            print(f"  Label exists: {label_file.exists()}")
            
            if label_file.exists():
                # Check if file is empty or has content
                try:
                    with open(label_file, 'r') as f:
                        content = f.read().strip()
                        if content:
                            lines = content.split('\n')
                            print(f"  Label has {len(lines)} lines")
                            # Show first line as example
                            print(f"  First line: {lines[0]}")
                        else:
                            print(f"  Label file is empty")
                except Exception as e:
                    print(f"  Error reading label file: {e}")
            else:
                # List what files ARE in the labels directory
                print(f"  Files in labels directory:")
                label_files = list(self.labels_dir.glob("*.txt"))
                for lf in label_files:
                    print(f"    {lf.name}")
            
            print()  # Empty line for readability

    def process_dataset(self):
        """Process entire dataset with better debugging"""
        
        # Add debug call
        self.debug_file_paths()
        
        # Get all image files
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
        image_files = []
        for ext in image_extensions:
            image_files.extend(list(self.images_dir.glob(ext)))
            image_files.extend(list(self.images_dir.glob(ext.upper())))
        
        # Remove duplicates
        image_files = list(set(image_files))
        
        print(f"Found {len(image_files)} images to process")
        
        total_positive_crops = 0
        total_negative_crops = 0
        
        for img_file in image_files:
            print(f"Processing {img_file.name}...")
            
            # Load image
            image = cv2.imread(str(img_file))
            if image is None:
                print(f"  Warning: Could not load {img_file}")
                continue
            
            # Load corresponding labels
            label_file = self.labels_dir / f"{img_file.stem}.txt"
            print(f"  Looking for label file: {label_file}")
            print(f"  Label file exists: {label_file.exists()}")
            
            annotations = self.parse_yolo_labels(str(label_file))
            
            if not annotations:
                print(f"  No annotations found for {img_file.name}")
                if label_file.exists():
                    print(f"  Label file exists but no valid annotations parsed")
                    # Show what's in the file
                    try:
                        with open(label_file, 'r') as f:
                            content = f.read()
                            print(f"  File content: '{content}'")
                    except Exception as e:
                        print(f"  Error reading file: {e}")
                continue
            
            print(f"  Found {len(annotations)} annotations")
            
            img_positive_crops = 0
            img_negative_crops = 0
            
            # Generate positive crops (one per bbox)
            for i, bbox in enumerate(annotations):
                crop_name = self.generate_positive_crop(
                    image, bbox, annotations, img_file.stem, i
                )
                if crop_name:
                    img_positive_crops += 1
            
            # Generate negative crops
            num_negatives = max(1, int(len(annotations) * self.negative_ratio))
            for _ in range(num_negatives):
                crop_name = self.generate_negative_crop(
                    image, annotations, img_file.stem
                )
                if crop_name:
                    img_negative_crops += 1
            
            total_positive_crops += img_positive_crops
            total_negative_crops += img_negative_crops
            
            print(f"  Generated {img_positive_crops} positive, {img_negative_crops} negative crops")
        
        print(f"\nDataset generation complete!")
        print(f"Total positive crops: {total_positive_crops}")
        print(f"Total negative crops: {total_negative_crops}")
        print(f"Total crops: {total_positive_crops + total_negative_crops}")
        if total_positive_crops + total_negative_crops > 0:
            print(f"Negative ratio: {total_negative_crops/(total_positive_crops + total_negative_crops):.2%}")
        print(f"Crop size: {self.crop_size}x{self.crop_size}")
        print(f"Output directory: {self.output_dir}")

    def parse_yolo_labels(self, label_file: str) -> List[List[float]]:
        """Parse YOLO format labels with better error handling"""
        annotations = []
        if not os.path.exists(label_file):
            return annotations
            
        try:
            with open(label_file, 'r') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:  # Skip empty lines
                        continue
                        
                    parts = line.split()
                    if len(parts) >= 5:
                        try:
                            class_id = int(parts[0])
                            if class_id == 1:
                                continue  # Skip class_id 1
                            bbox = [float(x) for x in parts[1:5]]
                            annotations.append([class_id] + bbox)
                        except ValueError as e:
                            print(f"  Warning: Invalid data on line {line_num}: {line} - {e}")
                    else:
                        print(f"  Warning: Line {line_num} has only {len(parts)} parts: {line}")
        except Exception as e:
            print(f"  Error reading label file {label_file}: {e}")
        
        return annotations

# Usage
if __name__ == "__main__":
    generator = GrapeCropGenerator(
        images_dir="../data_1506/test/images",           # Your original images folder
        labels_dir="../data_1506/test/labels",           # Your original labels folder  
        output_dir="./test_bbox/",   # Output folder for crops
        crop_size=320,                 # 160x160 pixels - good balance for grape detection
        expansion_factor=1.6,          # 40% expansion for context
        negative_ratio=0.6           # 25% negative samples
    )
    
    generator.process_dataset()