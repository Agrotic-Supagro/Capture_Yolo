import os
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns

def calculate_iou(box1, box2):
    """Calculate IoU between two bounding boxes in [x, y, w, h] format"""
    # Convert to [x1, y1, x2, y2] format
    x1_1, y1_1 = box1[0] - box1[2]/2, box1[1] - box1[3]/2
    x2_1, y2_1 = box1[0] + box1[2]/2, box1[1] + box1[3]/2
    
    x1_2, y1_2 = box2[0] - box2[2]/2, box2[1] - box2[3]/2
    x2_2, y2_2 = box2[0] + box2[2]/2, box2[1] + box2[3]/2
    
    # Calculate intersection
    x1 = max(x1_1, x1_2)
    y1 = max(y1_1, y1_2)
    x2 = min(x2_1, x2_2)
    y2 = min(y2_1, y2_2)

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = box1[2] * box1[3]
    area2 = box2[2] * box2[3]
    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0


def load_boxes(file_path):
    """Load bounding boxes from a file"""
    boxes = []
    if not os.path.exists(file_path):
        return boxes
        
    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            if line:  # Skip empty lines
                parts = list(map(float, line.split()))
                if len(parts) >= 5:  # Ensure we have at least class + 4 coordinates
                    boxes.append(parts)
    return boxes

def parse_loaded_boxes(boxes):
    """Parse loaded boxes into structured format"""
    data = []
    for box in boxes:
        if len(box) >= 5:
            class_id = int(box[0])
            x, y, w, h = box[1:5]
            confidence = box[5] if len(box) > 5 else 1.0
            data.append((class_id, x, y, w, h, confidence))
    return data

def get_class_name(class_id):
    """Convert class ID to class name"""
    class_map = {0: 'mature', 1: 'jeune', 2: 'medium'}
    return class_map.get(class_id, str(class_id))

def evaluate_metrics(predictions_dir, gt1_dir, gt2_dir, iou_thresholds):
    """Evaluate detection metrics"""
    # Initialize counters per class
    classes = set()
    TP = defaultdict(int)
    FP = defaultdict(int)
    FN = defaultdict(int)
    
    # Initialize class-agnostic counters
    agnostic_TP = 0
    agnostic_FP = 0
    agnostic_FN = 0
    
    processed_files = 0

    # Iterate over each prediction file
    for pred_file in os.listdir(predictions_dir):
        if not pred_file.endswith('.txt'):
            continue
            
        base_name = os.path.splitext(pred_file)[0]
        pred_path = os.path.join(predictions_dir, pred_file)
        gt1_path = os.path.join(gt1_dir, base_name + '.txt')
        gt2_path = os.path.join(gt2_dir, base_name + '.txt')

        # Debug: Check if files exist
        if not os.path.exists(gt1_path):
            print(f"Warning: GT1 file not found: {gt1_path}")
            continue
        if not os.path.exists(gt2_path):
            print(f"Warning: GT2 file not found: {gt2_path}")
            continue

        predictions = load_boxes(pred_path)
        gt1 = load_boxes(gt1_path)
        gt2 = load_boxes(gt2_path)
        
        processed_files += 1

        # Parse boxes into structured format
        pred_data = parse_loaded_boxes(predictions)
        gt1_data = parse_loaded_boxes(gt1)
        gt2_data = parse_loaded_boxes(gt2)

        # Collect all classes from GT1
        for class_id, _, _, _, _, _ in gt1_data:
            classes.add(class_id)
        
        # Collect all classes from predictions
        for class_id, _, _, _, _, _ in pred_data:
            classes.add(class_id)

        # Track which GT boxes have been matched for class-agnostic evaluation
        gt1_matched = [False] * len(gt1_data)
        gt2_matched = [False] * len(gt2_data)

        # Process each prediction (CLASS-SPECIFIC)
        for pred_class, pred_x, pred_y, pred_w, pred_h, _ in pred_data:
            matched_gt1 = False
            matched_gt2 = False

            iou_threshold = iou_thresholds.get(pred_class, 0.5)

            # Check against GT1 (class-specific)
            for gt_class, gt_x, gt_y, gt_w, gt_h, _ in gt1_data:
                if gt_class == pred_class:
                    iou = calculate_iou([pred_x, pred_y, pred_w, pred_h], 
                                      [gt_x, gt_y, gt_w, gt_h])
                    if iou > iou_threshold:
                        matched_gt1 = True
                        break

            # Check against GT2 if not matched with GT1 (class-specific)
            if not matched_gt1:
                for gt_class, gt_x, gt_y, gt_w, gt_h, _ in gt2_data:
                    if gt_class == pred_class:
                        iou = calculate_iou([pred_x, pred_y, pred_w, pred_h], 
                                          [gt_x, gt_y, gt_w, gt_h])
                        if iou > iou_threshold:
                            matched_gt2 = True
                            break

            # Update class-specific counters
            if matched_gt1 :
                TP[pred_class] += 1
            elif not matched_gt2:
                FP[pred_class] += 1

        # Process each prediction (CLASS-AGNOSTIC)
        for pred_class, pred_x, pred_y, pred_w, pred_h, _ in pred_data:
            matched_gt1_agnostic = False
            matched_gt2_agnostic = False

            iou_threshold = iou_thresholds.get(pred_class, 0.5)

            # Check against GT1 (class-agnostic)
            for idx, (gt_class, gt_x, gt_y, gt_w, gt_h, _) in enumerate(gt1_data):
                if not gt1_matched[idx]:
                    iou = calculate_iou([pred_x, pred_y, pred_w, pred_h], 
                                      [gt_x, gt_y, gt_w, gt_h])
                    if iou > iou_threshold:
                        matched_gt1_agnostic = True
                        gt1_matched[idx] = True
                        break

            # Check against GT2 if not matched with GT1 (class-agnostic)
            if not matched_gt1_agnostic:
                for idx, (gt_class, gt_x, gt_y, gt_w, gt_h, _) in enumerate(gt2_data):
                    if not gt2_matched[idx]:
                        iou = calculate_iou([pred_x, pred_y, pred_w, pred_h], 
                                          [gt_x, gt_y, gt_w, gt_h])
                        if iou > iou_threshold:
                            matched_gt2_agnostic = True
                            gt2_matched[idx] = True
                            break

            # Update class-agnostic counters
            if matched_gt1_agnostic :
                agnostic_TP += 1
            elif not matched_gt2_agnostic:
                agnostic_FP += 1

        # Count False Negatives (class-specific)
        for gt_class, gt_x, gt_y, gt_w, gt_h, _ in gt1_data:
            iou_threshold = iou_thresholds.get(gt_class, 0.5)
            matched = False
            
            for pred_class, pred_x, pred_y, pred_w, pred_h, _ in pred_data:
                if pred_class == gt_class:
                    iou = calculate_iou([pred_x, pred_y, pred_w, pred_h], 
                                      [gt_x, gt_y, gt_w, gt_h])
                    if iou > iou_threshold:
                        matched = True
                        break
            
            if not matched:
                FN[gt_class] += 1

        # Count False Negatives (class-agnostic)
        for idx, _ in enumerate(gt1_data):
            if not gt1_matched[idx]:
                agnostic_FN += 1

    # Calculate metrics per class
    metrics = {}
    for cls in classes:
        tp = TP[cls]
        fp = FP[cls]
        fn = FN[cls]
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        metrics[cls] = {
            'Precision': precision,
            'Recall': recall,
            'F1 Score': f1_score,
            'TP': tp,
            'FP': fp,
            'FN': fn
        }

    # Calculate class-agnostic metrics
    agnostic_precision = agnostic_TP / (agnostic_TP + agnostic_FP) if (agnostic_TP + agnostic_FP) > 0 else 0
    agnostic_recall = agnostic_TP / (agnostic_TP + agnostic_FN) if (agnostic_TP + agnostic_FN) > 0 else 0
    agnostic_f1 = 2 * (agnostic_precision * agnostic_recall) / (agnostic_precision + agnostic_recall) if (agnostic_precision + agnostic_recall) > 0 else 0

    metrics['class_agnostic'] = {
        'Precision': agnostic_precision,
        'Recall': agnostic_recall,
        'F1 Score': agnostic_f1,
        'TP': agnostic_TP,
        'FP': agnostic_FP,
        'FN': agnostic_FN
    }

    # Convert defaultdicts to regular dicts for printing
    TP_dict = dict(TP)
    FP_dict = dict(FP)
    FN_dict = dict(FN)
    
    print(f"Class-specific - TP: {TP_dict}, FP: {FP_dict}, FN: {FN_dict}")
    print(f"Class-agnostic - TP: {agnostic_TP}, FP: {agnostic_FP}, FN: {agnostic_FN}")
    
    return metrics

def create_confusion_matrix(gt_folder, pred_folder, iou_threshold=0.5):
    """Create confusion matrix for object detection"""
    confusion_matrix = defaultdict(lambda: defaultdict(int))

    # List all files in the GT and prediction folders
    gt_files = [f for f in os.listdir(gt_folder) if f.endswith('.txt')]
    pred_files = [f for f in os.listdir(pred_folder) if f.endswith('.txt')]

    # Iterate over ground truth files
    for gt_file in gt_files:
        gt_path = os.path.join(gt_folder, gt_file)
        gt_boxes = load_boxes(gt_path)
        gt_data = parse_loaded_boxes(gt_boxes)

        # Find the corresponding prediction file
        if gt_file in pred_files:
            pred_path = os.path.join(pred_folder, gt_file)
            pred_boxes = load_boxes(pred_path)
            pred_data = parse_loaded_boxes(pred_boxes)

            # Initialize a list to keep track of matched predictions
            matched_preds = [False] * len(pred_data)

            # Iterate over ground truth data
            for gt_class_id, gt_x, gt_y, gt_w, gt_h, _ in gt_data:
                max_iou = 0
                predicted_class_id = 'background'
                best_pred_idx = -1

                # Iterate over prediction data
                for idx, (pred_class_id, pred_x, pred_y, pred_w, pred_h, _) in enumerate(pred_data):
                    iou = calculate_iou([gt_x, gt_y, gt_w, gt_h], [pred_x, pred_y, pred_w, pred_h])
                    if iou > max_iou:
                        max_iou = iou
                        predicted_class_id = get_class_name(pred_class_id)
                        best_pred_idx = idx

                # Update confusion matrix based on max IoU
                if max_iou >= iou_threshold:
                    confusion_matrix[get_class_name(gt_class_id)][predicted_class_id] += 1
                    # Mark the prediction as matched
                    if best_pred_idx >= 0:
                        matched_preds[best_pred_idx] = True
                else:
                    confusion_matrix[get_class_name(gt_class_id)]['background'] += 1

            # Handle predictions that do not exist (GT is background)
            for idx, (pred_class_id, _, _, _, _, _) in enumerate(pred_data):
                if not matched_preds[idx]:
                    confusion_matrix['background'][get_class_name(pred_class_id)] += 1

    return confusion_matrix

def plot_confusion_matrix(confusion_matrix, save_path=None):
    """Plot confusion matrix with proper class names and order"""
    # Define the desired order: jeune, medium, mature, background
    class_order = ['jeune', 'medium', 'mature', 'background']
    
    # Get all classes that actually appear in the confusion matrix
    all_classes = set(confusion_matrix.keys())
    for pred_classes in confusion_matrix.values():
        all_classes.update(pred_classes.keys())
    
    # Filter class_order to only include classes that actually exist
    existing_classes = [cls for cls in class_order if cls in all_classes]
    
    # Add any additional classes not in our predefined order
    additional_classes = sorted(all_classes - set(class_order))
    final_class_order = existing_classes + additional_classes
    
    # Build confusion matrix data
    confusion_matrix_data = []
    for gt_class in final_class_order:
        row = []
        for pred_class in final_class_order:
            value = confusion_matrix[gt_class].get(pred_class, 0)
            row.append(value)
        confusion_matrix_data.append(row)
    
    confusion_matrix_array = np.array(confusion_matrix_data)

    # Plot the confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(confusion_matrix_array, annot=True, fmt='d', cmap='Blues',
                xticklabels=final_class_order, 
                yticklabels=final_class_order)

    plt.title('Confusion Matrix')
    plt.ylabel('True Class')
    plt.xlabel('Predicted Class')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def print_metrics_summary(metrics_result):
    """Print a formatted summary of all metrics"""
    print("\n" + "="*60)
    print("DETECTION METRICS SUMMARY")
    print("="*60)
    
    # Print class-specific metrics
    print("\nCLASS-SPECIFIC METRICS:")
    print("-" * 40)
    
    # Separate numeric class keys from string keys and sort them
    class_keys = [k for k in metrics_result.keys() if k != 'class_agnostic']
    class_keys = sorted(class_keys)
    
    for cls in class_keys:
        m = metrics_result[cls]
        class_name = get_class_name(cls)
        print(f"Class {cls} ({class_name}):")
        print(f"  Precision: {m['Precision']:.3f}")
        print(f"  Recall:    {m['Recall']:.3f}")
        print(f"  F1-Score:  {m['F1 Score']:.3f}")
        print(f"  TP/FP/FN:  {m['TP']}/{m['FP']}/{m['FN']}")
        print()
    
    # Print class-agnostic metrics
    if 'class_agnostic' in metrics_result:
        print("CLASS-AGNOSTIC METRICS:")
        print("-" * 40)
        m = metrics_result['class_agnostic']
        print(f"Overall Detection Performance (ignoring class labels):")
        print(f"  Precision: {m['Precision']:.3f}")
        print(f"  Recall:    {m['Recall']:.3f}")
        print(f"  F1-Score:  {m['F1 Score']:.3f}")
        print(f"  TP/FP/FN:  {m['TP']}/{m['FP']}/{m['FN']}")
        print()
        print("Note: In class-agnostic evaluation, a 'mature grape' detected as")
        print("      'medium grape' is still considered a correct detection.")
    
    print("="*60)

def metrics(predictions_dir, gt1_dir, gt2_dir, iou_thresholds, save_path=None):
    """Main function to calculate metrics and plot confusion matrix"""
    metrics_result = evaluate_metrics(predictions_dir, gt1_dir, gt2_dir, iou_thresholds)
    
    # Print formatted summary
    print_metrics_summary(metrics_result)
    
    # Create confusion matrix (class-specific)
    confusion_matrix = create_confusion_matrix(gt1_dir, predictions_dir, iou_threshold=0.5)
    plot_confusion_matrix(confusion_matrix, save_path)
    
    return metrics_result