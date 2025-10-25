"""
Summary Analysis Script for Overlapping Events

This script provides additional insights and summary statistics
for the overlapping events analysis.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pandas as pd
from collections import Counter
import argparse

# DESED event classes
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

def analyze_overlap_patterns(outputs_path: str, targets_path: str, 
                           threshold: float = 0.5, save_path: str = None):
    """
    Analyze patterns in overlapping events
    
    Args:
        outputs_path: Path to train_outputs.npy
        targets_path: Path to train_targets.npy
        threshold: Threshold for ground truth events
        save_path: Path to save analysis plots
    """
    # Load data
    outputs = np.load(outputs_path)
    targets = np.load(targets_path)
    
    print(f"Analyzing overlap patterns...")
    print(f"Data shape: {outputs.shape}")
    
    # Find all overlapping combinations
    overlap_combinations = []
    overlap_counts = Counter()
    
    for batch_idx in range(outputs.shape[0]):
        for time_frame in range(outputs.shape[1]):
            events_present = targets[batch_idx, time_frame, :] > threshold
            
            if np.sum(events_present) > 1:
                present_classes = [DESED_CLASSES[i] for i in range(len(DESED_CLASSES)) if events_present[i]]
                combination = "+".join(sorted(present_classes))
                overlap_combinations.append(combination)
                overlap_counts[combination] += 1
    
    # Create visualizations
    fig, axes = plt.subplots(2, 2, figsize=(20, 15))
    
    # 1. Most common overlap combinations
    top_combinations = overlap_counts.most_common(15)
    combinations, counts = zip(*top_combinations)
    
    axes[0, 0].barh(range(len(combinations)), counts)
    axes[0, 0].set_yticks(range(len(combinations)))
    axes[0, 0].set_yticklabels(combinations, fontsize=8)
    axes[0, 0].set_xlabel("Frequency")
    axes[0, 0].set_title("Most Common Overlap Combinations")
    axes[0, 0].invert_yaxis()
    
    # 2. Event co-occurrence matrix
    cooccurrence_matrix = np.zeros((len(DESED_CLASSES), len(DESED_CLASSES)))
    
    for combination in overlap_combinations:
        events = combination.split("+")
        for i, event1 in enumerate(events):
            for j, event2 in enumerate(events):
                if i != j:
                    idx1 = DESED_CLASSES.index(event1)
                    idx2 = DESED_CLASSES.index(event2)
                    cooccurrence_matrix[idx1, idx2] += 1
    
    # Normalize by diagonal (self-cooccurrence)
    for i in range(len(DESED_CLASSES)):
        if cooccurrence_matrix[i, i] > 0:
            cooccurrence_matrix[i, :] /= cooccurrence_matrix[i, i]
    
    sns.heatmap(cooccurrence_matrix, 
                annot=True, 
                fmt=".2f",
                xticklabels=DESED_CLASSES,
                yticklabels=DESED_CLASSES,
                ax=axes[0, 1],
                cmap="Blues")
    axes[0, 1].set_title("Event Co-occurrence Matrix\n(Normalized by Self-occurrence)")
    axes[0, 1].set_xlabel("Co-occurs with")
    axes[0, 1].set_ylabel("Event")
    
    # 3. Number of overlapping events distribution
    num_overlaps = [len(combo.split("+")) for combo in overlap_combinations]
    overlap_dist = Counter(num_overlaps)
    
    axes[1, 0].bar(overlap_dist.keys(), overlap_dist.values())
    axes[1, 0].set_xlabel("Number of Overlapping Events")
    axes[1, 0].set_ylabel("Frequency")
    axes[1, 0].set_title("Distribution of Overlap Complexity")
    
    # 4. Event participation in overlaps
    event_participation = Counter()
    for combination in overlap_combinations:
        events = combination.split("+")
        for event in events:
            event_participation[event] += 1
    
    events = list(event_participation.keys())
    participations = list(event_participation.values())
    
    axes[1, 1].bar(range(len(events)), participations)
    axes[1, 1].set_xticks(range(len(events)))
    axes[1, 1].set_xticklabels(events, rotation=45, ha="right")
    axes[1, 1].set_ylabel("Times Participating in Overlaps")
    axes[1, 1].set_title("Event Participation in Overlaps")
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Overlap pattern analysis saved to: {save_path}")
    
    plt.close()
    
    # Print summary statistics
    print("\n" + "="*60)
    print("OVERLAP PATTERN SUMMARY")
    print("="*60)
    
    print(f"Total overlapping instances: {len(overlap_combinations)}")
    print(f"Unique overlap combinations: {len(overlap_counts)}")
    print(f"Most common combination: {top_combinations[0][0]} ({top_combinations[0][1]} times)")
    
    print(f"\nOverlap complexity distribution:")
    for num_events, count in sorted(overlap_dist.items()):
        print(f"  {num_events} events: {count} instances")
    
    print(f"\nMost active events in overlaps:")
    for event, count in event_participation.most_common(5):
        print(f"  {event}: {count} participations")
    
    return overlap_counts, cooccurrence_matrix

def create_summary_report(outputs_path: str, targets_path: str, 
                         output_dir: str = "analysis_summary"):
    """
    Create a comprehensive summary report
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    print("Creating comprehensive summary report...")
    
    # Run overlap pattern analysis
    overlap_counts, cooccurrence_matrix = analyze_overlap_patterns(
        outputs_path, targets_path, 
        save_path=output_dir / "overlap_patterns.png"
    )
    
    # Create a summary table
    summary_data = []
    
    for i, event in enumerate(DESED_CLASSES):
        # Count how many times this event appears in overlaps
        participation_count = sum(count for combo, count in overlap_counts.items() if event in combo)
        
        # Count unique combinations this event participates in
        unique_combinations = len([combo for combo in overlap_counts.keys() if event in combo])
        
        # Calculate average co-occurrence with other events
        avg_cooccurrence = np.mean(cooccurrence_matrix[i, :])
        
        summary_data.append({
            "Event": event,
            "Total_Participations": participation_count,
            "Unique_Combinations": unique_combinations,
            "Avg_Cooccurrence": avg_cooccurrence,
            "Most_Common_Partner": DESED_CLASSES[np.argmax(cooccurrence_matrix[i, :])] if np.max(cooccurrence_matrix[i, :]) > 0 else "None"
        })
    
    summary_df = pd.DataFrame(summary_data)
    summary_df = summary_df.sort_values("Total_Participations", ascending=False)
    
    # Save summary table
    summary_df.to_csv(output_dir / "event_summary.csv", index=False)
    
    # Create summary visualization
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Participation counts
    axes[0, 0].barh(summary_df["Event"], summary_df["Total_Participations"])
    axes[0, 0].set_xlabel("Total Participations in Overlaps")
    axes[0, 0].set_title("Event Participation in Overlaps")
    
    # Unique combinations
    axes[0, 1].barh(summary_df["Event"], summary_df["Unique_Combinations"])
    axes[0, 1].set_xlabel("Unique Overlap Combinations")
    axes[0, 1].set_title("Diversity of Overlap Combinations")
    
    # Average co-occurrence
    axes[1, 0].barh(summary_df["Event"], summary_df["Avg_Cooccurrence"])
    axes[1, 0].set_xlabel("Average Co-occurrence Rate")
    axes[1, 0].set_title("Average Co-occurrence with Other Events")
    
    # Participation vs Diversity scatter
    axes[1, 1].scatter(summary_df["Total_Participations"], summary_df["Unique_Combinations"], 
                      s=100, alpha=0.7)
    for i, event in enumerate(summary_df["Event"]):
        axes[1, 1].annotate(event, 
                           (summary_df["Total_Participations"].iloc[i], 
                            summary_df["Unique_Combinations"].iloc[i]),
                           fontsize=8, ha="center")
    axes[1, 1].set_xlabel("Total Participations")
    axes[1, 1].set_ylabel("Unique Combinations")
    axes[1, 1].set_title("Participation vs Diversity")
    
    plt.tight_layout()
    plt.savefig(output_dir / "summary_visualization.png", dpi=300, bbox_inches="tight")
    plt.close()
    
    print(f"\nSummary report saved to: {output_dir}")
    print("Files created:")
    print(f"  - overlap_patterns.png: Detailed overlap pattern analysis")
    print(f"  - event_summary.csv: Summary statistics for each event")
    print(f"  - summary_visualization.png: Visual summary of event characteristics")
    
    return summary_df

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Summary analysis for overlapping events")
    parser.add_argument("--outputs", default="autrainer/epoch_65/train_outputs.npy", 
                       help="Path to train_outputs.npy")
    parser.add_argument("--targets", default="autrainer/epoch_65/train_targets.npy",
                       help="Path to train_targets.npy") 
    parser.add_argument("--threshold", type=float, default=0.5,
                       help="Threshold for ground truth events")
    parser.add_argument("--output_dir", default="analysis_summary",
                       help="Directory to save summary results")
    
    args = parser.parse_args()
    
    # Create summary report
    summary_df = create_summary_report(args.outputs, args.targets, args.output_dir)
    
    print("\n" + "="*60)
    print("FINAL SUMMARY")
    print("="*60)
    print(summary_df.to_string(index=False))

if __name__ == "__main__":
    main()
