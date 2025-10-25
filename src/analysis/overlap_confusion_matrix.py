import numpy as np
import matplotlib
matplotlib.use("Agg")  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse
from typing import Tuple, List, Dict
import warnings
warnings.filterwarnings("ignore")

DESED_CLASSES = [
    "Alarm_bell_ringing",
    "Blender", 
    "Cat",
    "Dishes",
    "Dog",
    "Electric_shaver_toothbrush",
    "Frying",
    "Running_water",
    "Speech",
    "Vacuum_cleaner"
]

def load_data(outputs_path: str, targets_path: str) -> Tuple[np.ndarray, np.ndarray]:
    outputs = np.load(outputs_path)
    targets = np.load(targets_path)
    
    print(f"Loaded outputs shape: {outputs.shape}")
    print(f"Loaded targets shape: {targets.shape}")
    print(f"Outputs range: [{outputs.min():.3f}, {outputs.max():.3f}]")
    print(f"Targets range: [{targets.min():.3f}, {targets.max():.3f}]")
    
    return outputs, targets

def find_overlapping_events(targets: np.ndarray, threshold: float = 0.5) -> List[Tuple[int, int, int]]:
    overlapping_frames = []
    
    for batch_idx in range(targets.shape[0]):
        for time_frame in range(targets.shape[1]):
            # Get events present at this time frame
            events_present = targets[batch_idx, time_frame, :] > threshold
            
            # Check if multiple events overlap
            if np.sum(events_present) > 1:
                overlapping_frames.append((batch_idx, time_frame, events_present))
    
    return overlapping_frames

def compute_overlap_confusion_matrix(outputs: np.ndarray, targets: np.ndarray, 
                                   threshold: float = 0.5, confidence_threshold: float = 0.5) -> np.ndarray:
    
    probs = 1 / (1 + np.exp(-np.clip(outputs, -500, 500)))
    overlapping_frames = find_overlapping_events(targets, threshold)
    print(f"Found {len(overlapping_frames)} overlapping event frames")
    
    # Initialize confusion matrix
    num_classes = len(DESED_CLASSES)
    confusion_matrix = np.zeros((num_classes, num_classes), dtype=np.float32)
    overlap_counts = np.zeros(num_classes, dtype=np.float32)
    
    for batch_idx, time_frame, events_present in overlapping_frames:
        gt_events = events_present
        pred_events = probs[batch_idx, time_frame, :] > confidence_threshold
        
        # For each ground truth event, check what was predicted
        for gt_class in range(num_classes):
            if gt_events[gt_class]:
                overlap_counts[gt_class] += 1
                
                # Count predictions for this ground truth event
                for pred_class in range(num_classes):
                    if pred_events[pred_class]:
                        confusion_matrix[gt_class, pred_class] += 1
    
    # Normalize by overlap counts to get error rates
    for i in range(num_classes):
        if overlap_counts[i] > 0:
            confusion_matrix[i, :] /= overlap_counts[i]
    
    return confusion_matrix, overlap_counts

def plot_confusion_matrix(confusion_matrix: np.ndarray, overlap_counts: np.ndarray, 
                         save_path: str = None, figsize: Tuple[int, int] = (12, 10)):
    plt.figure(figsize=figsize)
    error_mask = np.ones_like(confusion_matrix, dtype=bool)
    np.fill_diagonal(error_mask, False)
    # Plot confusion matrix
    ax = sns.heatmap(confusion_matrix, 
                     annot=True, 
                     fmt=".3f",
                     cmap="RdYlBu_r",
                     xticklabels=DESED_CLASSES,
                     yticklabels=DESED_CLASSES,
                     cbar_kws={"label": "Prediction Rate"},
                     mask=~error_mask,
                     vmin=0, vmax=1)
    # Highlight error cases (off-diagonal) with different colormap
    sns.heatmap(confusion_matrix,
                annot=True,
                fmt=".3f", 
                cmap="Reds",
                xticklabels=DESED_CLASSES,
                yticklabels=DESED_CLASSES,
                mask=error_mask,
                vmin=0, vmax=1,
                cbar=False)
    
    plt.title("Confusion Matrix for Overlapping Events\n(Rows: Ground Truth, Cols: Predictions)\nRed: Errors, Blue: Correct Predictions", 
              fontsize=14, pad=20)
    plt.xlabel("Predicted Events", fontsize=12)
    plt.ylabel("Ground Truth Events", fontsize=12)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    overlap_text = "Overlap Counts:\n"
    for i, (class_name, count) in enumerate(zip(DESED_CLASSES, overlap_counts)):
        overlap_text += f"{class_name}: {int(count)}\n"
    plt.figtext(0.02, 0.02, overlap_text, fontsize=8, verticalalignment="bottom")
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Confusion matrix saved to: {save_path}")
    
    # Don't show plot in headless mode
    plt.close()

def analyze_error_patterns(confusion_matrix: np.ndarray, overlap_counts: np.ndarray):
    print("\n" + "="*60)
    print("OVERLAPPING EVENTS ERROR ANALYSIS")
    print("="*60)
    
    # Find most confused event pairs
    error_matrix = confusion_matrix.copy()
    np.fill_diagonal(error_matrix, 0)  # Remove correct predictions
    
    print("\nTop 10 Most Confused Event Pairs:")
    print("-" * 40)
    
    # Get top error pairs
    flat_indices = np.argsort(error_matrix.flatten())[::-1]
    top_errors = []
    
    for idx in flat_indices[:10]:
        gt_idx, pred_idx = np.unravel_index(idx, error_matrix.shape)
        error_rate = error_matrix[gt_idx, pred_idx]
        if error_rate > 0:
            top_errors.append((gt_idx, pred_idx, error_rate))
    
    for i, (gt_idx, pred_idx, error_rate) in enumerate(top_errors):
        print(f"{i+1:2d}. {DESED_CLASSES[gt_idx]:25s} → {DESED_CLASSES[pred_idx]:25s}: {error_rate:.3f}")
    
    # Analyze per-class performance
    print("\nPer-Class Overlap Performance:")
    print("-" * 40)
    
    for i, class_name in enumerate(DESED_CLASSES):
        if overlap_counts[i] > 0:
            correct_rate = confusion_matrix[i, i]
            total_error_rate = 1 - correct_rate
            print(f"{class_name:25s}: Correct={correct_rate:.3f}, Error={total_error_rate:.3f}, Count={int(overlap_counts[i])}")
        else:
            print(f"{class_name:25s}: No overlaps found")

def main():
    parser = argparse.ArgumentParser(description="Analyze overlapping events confusion matrix")
    parser.add_argument("--outputs", default="autrainer/epoch_65/train_outputs.npy", 
                       help="Path to train_outputs.npy")
    parser.add_argument("--targets", default="autrainer/epoch_65/train_targets.npy",
                       help="Path to train_targets.npy") 
    parser.add_argument("--threshold", type=float, default=0.5,
                       help="Threshold for ground truth events")
    parser.add_argument("--confidence_threshold", type=float, default=0.5,
                       help="Threshold for model predictions")
    parser.add_argument("--save_path", default="overlap_confusion_matrix.png",
                       help="Path to save the confusion matrix plot")
    
    args = parser.parse_args()
    
    # Load data
    print("Loading data...")
    outputs, targets = load_data(args.outputs, args.targets)
    
    # Compute confusion matrix
    print("\nComputing confusion matrix...")
    confusion_matrix, overlap_counts = compute_overlap_confusion_matrix(
        outputs, targets, args.threshold, args.confidence_threshold
    )
    
    # Plot results
    print("\nPlotting confusion matrix...")
    plot_confusion_matrix(confusion_matrix, overlap_counts, args.save_path)
    
    # Analyze error patterns
    analyze_error_patterns(confusion_matrix, overlap_counts)
    
    print(f"\nAnalysis complete! Confusion matrix saved to: {args.save_path}")

if __name__ == "__main__":
    main()
