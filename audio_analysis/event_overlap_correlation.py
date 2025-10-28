#!/usr/bin/env python3
"""
DESED Event Overlap and Correlation Analysis

This script analyzes temporal overlaps between events, calculates correlation
matrices, and identifies which events tend to occur together temporally.
"""

import os
import pandas as pd
import numpy as np
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy.stats import pearsonr, spearmanr
from scipy.spatial.distance import pdist, squareform
import warnings
warnings.filterwarnings("ignore")

class EventOverlapAnalyzer:
    def __init__(self, desed_root_path):
        """
        Initialize the analyzer with DESED dataset root path.
        
        Args:
            desed_root_path (str): Path to the DESED dataset root directory
        """
        self.desed_root = Path(desed_root_path)
        self.annotations_path = self.desed_root / "annotations"
        
        # Define dataset splits and their annotation files
        self.dataset_splits = {
            "public": "public.tsv",
            "weak": "weak(1578).tsv", 
            "synth_train": "synth_train(10000).tsv",
            "synth_val": "synth_val(2500).tsv",
            "real_strong": "real_audioset_strong(3373).tsv",
            "real_validation": "real_validation(1168).tsv"
        }
        
        self.timed_data = None
        self.overlap_stats = {}
        
    def load_timed_annotations(self):
        """Load annotations with timing information."""
        print("Loading timed annotation files...")
        
        timed_dfs = []
        
        for split_name, filename in self.dataset_splits.items():
            file_path = self.annotations_path / filename
            
            if not file_path.exists():
                print(f"Warning: {file_path} does not exist")
                continue
                
            df = pd.read_csv(file_path, sep="\t")
            
            # Only process files with timing information
            if "onset" in df.columns and "offset" in df.columns:
                df["split"] = split_name
                timed_dfs.append(df)
                print(f"  Loaded {len(df)} timed events from {split_name}")
            else:
                print(f"  Skipping {split_name} (no timing information)")
        
        if timed_dfs:
            self.timed_data = pd.concat(timed_dfs, ignore_index=True)
            print(f"Total timed events: {len(self.timed_data)}")
        else:
            print("No timed data found!")
            return False
            
        return True
    
    def calculate_temporal_overlaps(self):
        """Calculate temporal overlaps between events within the same file."""
        print("\nCalculating temporal overlaps...")
        
        if self.timed_data is None:
            print("No timed data available")
            return {}
        
        overlap_results = []
        
        # Group by filename to analyze overlaps within each file
        for filename, file_events in self.timed_data.groupby("filename"):
            events = file_events.sort_values("onset").reset_index(drop=True)
            
            # Calculate overlaps between all pairs of events in the file
            for i in range(len(events)):
                for j in range(i + 1, len(events)):
                    event1 = events.iloc[i]
                    event2 = events.iloc[j]
                    
                    # Calculate overlap
                    overlap_start = max(event1["onset"], event2["onset"])
                    overlap_end = min(event1["offset"], event2["offset"])
                    
                    if overlap_start < overlap_end:  # There is an overlap
                        overlap_duration = overlap_end - overlap_start
                        
                        # Calculate overlap percentage for each event
                        event1_duration = event1["offset"] - event1["onset"]
                        event2_duration = event2["offset"] - event2["onset"]
                        
                        overlap_pct_event1 = overlap_duration / event1_duration * 100
                        overlap_pct_event2 = overlap_duration / event2_duration * 100
                        
                        overlap_results.append({
                            "filename": filename,
                            "event1": event1["event_label"],
                            "event2": event2["event_label"],
                            "event1_onset": event1["onset"],
                            "event1_offset": event1["offset"],
                            "event2_onset": event2["onset"],
                            "event2_offset": event2["offset"],
                            "overlap_start": overlap_start,
                            "overlap_end": overlap_end,
                            "overlap_duration": overlap_duration,
                            "overlap_pct_event1": overlap_pct_event1,
                            "overlap_pct_event2": overlap_pct_event2,
                            "split": event1["split"]
                        })
        
        self.overlap_df = pd.DataFrame(overlap_results)
        
        if len(self.overlap_df) > 0:
            print(f"Found {len(self.overlap_df)} overlapping event pairs")
            
            # Calculate overlap statistics
            overlap_stats = {
                "total_overlaps": len(self.overlap_df),
                "unique_files_with_overlaps": self.overlap_df["filename"].nunique(),
                "avg_overlap_duration": self.overlap_df["overlap_duration"].mean(),
                "median_overlap_duration": self.overlap_df["overlap_duration"].median(),
                "max_overlap_duration": self.overlap_df["overlap_duration"].max(),
                "min_overlap_duration": self.overlap_df["overlap_duration"].min(),
                "avg_overlap_percentage": (self.overlap_df["overlap_pct_event1"].mean() + 
                                         self.overlap_df["overlap_pct_event2"].mean()) / 2
            }
        else:
            print("No overlaps found")
            overlap_stats = {}
        
        return overlap_stats
    
    def analyze_overlap_patterns(self):
        """Analyze patterns in event overlaps."""
        print("\nAnalyzing overlap patterns...")
        
        if self.overlap_df is None or len(self.overlap_df) == 0:
            return {}
        
        # Most frequently overlapping event pairs
        overlap_pairs = self.overlap_df.groupby(["event1", "event2"]).agg({
            "overlap_duration": ["count", "mean", "sum"],
            "overlap_pct_event1": "mean",
            "overlap_pct_event2": "mean"
        }).round(3)
        
        overlap_pairs.columns = ["overlap_count", "avg_overlap_duration", "total_overlap_duration",
                               "avg_overlap_pct_event1", "avg_overlap_pct_event2"]
        
        # Sort by overlap count
        overlap_pairs = overlap_pairs.sort_values("overlap_count", ascending=False)
        
        # Overlap statistics by event type
        event_overlap_stats = {}
        
        # For each event, calculate how often it overlaps with others
        all_events = set(self.timed_data["event_label"].unique())
        
        for event in all_events:
            # Events that overlap with this event
            overlapping_with = self.overlap_df[
                (self.overlap_df["event1"] == event) | (self.overlap_df["event2"] == event)
            ]
            
            if len(overlapping_with) > 0:
                # Get the other event in each overlap
                other_events = []
                for _, row in overlapping_with.iterrows():
                    if row["event1"] == event:
                        other_events.append(row["event2"])
                    else:
                        other_events.append(row["event1"])
                
                event_overlap_stats[event] = {
                    "total_overlaps": len(overlapping_with),
                    "unique_overlapping_events": len(set(other_events)),
                    "avg_overlap_duration": overlapping_with["overlap_duration"].mean(),
                    "most_common_overlap_partner": Counter(other_events).most_common(1)[0] if other_events else None
                }
        
        # Overlap patterns by split
        overlap_by_split = {}
        for split_name in self.timed_data["split"].unique():
            split_overlaps = self.overlap_df[self.overlap_df["split"] == split_name]
            if len(split_overlaps) > 0:
                overlap_by_split[split_name] = {
                    "total_overlaps": len(split_overlaps),
                    "unique_files": split_overlaps["filename"].nunique(),
                    "avg_overlap_duration": split_overlaps["overlap_duration"].mean(),
                    "overlap_ratio": len(split_overlaps) / len(self.timed_data[self.timed_data["split"] == split_name])
                }
        
        overlap_patterns = {
            "overlap_pairs": overlap_pairs,
            "event_overlap_stats": event_overlap_stats,
            "overlap_by_split": overlap_by_split
        }
        
        return overlap_patterns
    
    def calculate_event_correlations(self):
        """Calculate correlation matrices between events."""
        print("\nCalculating event correlations...")
        
        if self.timed_data is None:
            return {}
        
        # Create binary presence matrix (events present in each file)
        file_event_matrix = self.timed_data.groupby(["filename", "event_label"]).size().unstack(fill_value=0)
        
        # Convert to binary (1 if event present, 0 if not)
        file_event_binary = (file_event_matrix > 0).astype(int)
        
        # Calculate Pearson correlation
        pearson_corr = file_event_binary.corr()
        
        # Calculate Spearman correlation
        spearman_corr = file_event_binary.corr(method="spearman")
        
        # Calculate Jaccard similarity (intersection over union)
        jaccard_matrix = pd.DataFrame(index=file_event_binary.columns, 
                                     columns=file_event_binary.columns)
        
        for event1 in file_event_binary.columns:
            for event2 in file_event_binary.columns:
                if event1 == event2:
                    jaccard_matrix.loc[event1, event2] = 1.0
                else:
                    # Jaccard = intersection / union
                    intersection = ((file_event_binary[event1] == 1) & 
                                  (file_event_binary[event2] == 1)).sum()
                    union = ((file_event_binary[event1] == 1) | 
                           (file_event_binary[event2] == 1)).sum()
                    
                    if union > 0:
                        jaccard_matrix.loc[event1, event2] = intersection / union
                    else:
                        jaccard_matrix.loc[event1, event2] = 0.0
        
        jaccard_matrix = jaccard_matrix.astype(float)
        
        # Find most correlated pairs
        correlation_pairs = []
        
        # Get upper triangle indices (avoid duplicates)
        upper_tri_indices = np.triu_indices_from(pearson_corr.values, k=1)
        
        for i, j in zip(upper_tri_indices[0], upper_tri_indices[1]):
            event1 = pearson_corr.index[i]
            event2 = pearson_corr.index[j]
            
            correlation_pairs.append({
                "event1": event1,
                "event2": event2,
                "pearson_corr": pearson_corr.iloc[i, j],
                "spearman_corr": spearman_corr.iloc[i, j],
                "jaccard_similarity": jaccard_matrix.iloc[i, j]
            })
        
        # Sort by Pearson correlation
        correlation_pairs = sorted(correlation_pairs, 
                                key=lambda x: abs(x["pearson_corr"]), 
                                reverse=True)
        
        correlation_analysis = {
            "pearson_correlation": pearson_corr,
            "spearman_correlation": spearman_corr,
            "jaccard_similarity": jaccard_matrix,
            "top_correlated_pairs": correlation_pairs[:50],  # Top 50 pairs
            "file_event_matrix": file_event_binary
        }
        
        return correlation_analysis
    
    def analyze_temporal_correlations(self):
        """Analyze temporal correlations between events."""
        print("\nAnalyzing temporal correlations...")
        
        if self.timed_data is None or len(self.timed_data) == 0:
            return {}
        
        # Create temporal features for each event
        temporal_features = []
        
        for filename, file_events in self.timed_data.groupby("filename"):
            events = file_events.sort_values("onset")
            
            # Calculate temporal features for each event
            for idx, event in events.iterrows():
                features = {
                    "filename": filename,
                    "event_label": event["event_label"],
                    "onset": event["onset"],
                    "offset": event["offset"],
                    "duration": event["offset"] - event["onset"],
                    "position_in_file": idx / len(events),  # Relative position
                    "is_first_event": idx == 0,
                    "is_last_event": idx == len(events) - 1,
                    "events_before": idx,
                    "events_after": len(events) - idx - 1,
                    "split": event["split"]
                }
                
                temporal_features.append(features)
        
        temporal_df = pd.DataFrame(temporal_features)
        
        # Analyze temporal patterns by event type
        temporal_patterns = {}
        
        for event in temporal_df["event_label"].unique():
            event_data = temporal_df[temporal_df["event_label"] == event]
            
            temporal_patterns[event] = {
                "avg_onset": event_data["onset"].mean(),
                "median_onset": event_data["onset"].median(),
                "avg_duration": event_data["duration"].mean(),
                "median_duration": event_data["duration"].median(),
                "avg_position": event_data["position_in_file"].mean(),
                "first_event_ratio": event_data["is_first_event"].mean(),
                "last_event_ratio": event_data["is_last_event"].mean(),
                "avg_events_before": event_data["events_before"].mean(),
                "avg_events_after": event_data["events_after"].mean(),
                "total_occurrences": len(event_data)
            }
        
        # Calculate temporal correlations between events
        # Create a matrix of average temporal features
        feature_matrix = []
        event_labels = []
        
        for event, patterns in temporal_patterns.items():
            feature_vector = [
                patterns["avg_onset"],
                patterns["avg_duration"],
                patterns["avg_position"],
                patterns["first_event_ratio"],
                patterns["last_event_ratio"]
            ]
            feature_matrix.append(feature_vector)
            event_labels.append(event)
        
        feature_matrix = np.array(feature_matrix)
        
        # Calculate temporal similarity matrix
        temporal_similarity = 1 - pdist(feature_matrix, metric="cosine")
        temporal_similarity_matrix = squareform(temporal_similarity)
        temporal_similarity_df = pd.DataFrame(temporal_similarity_matrix,
                                             index=event_labels,
                                             columns=event_labels)
        
        temporal_analysis = {
            "temporal_patterns": temporal_patterns,
            "temporal_similarity_matrix": temporal_similarity_df,
            "temporal_features_df": temporal_df
        }
        
        return temporal_analysis
    
    def generate_overlap_report(self):
        """Generate comprehensive overlap and correlation report."""
        print("=" * 80)
        print("DESED EVENT OVERLAP AND CORRELATION ANALYSIS")
        print("=" * 80)
        
        # Load timed data
        if not self.load_timed_annotations():
            print("No timed data available for overlap analysis")
            return {}
        
        # Run all analyses
        overlap_stats = self.calculate_temporal_overlaps()
        overlap_patterns = self.analyze_overlap_patterns()
        correlation_analysis = self.calculate_event_correlations()
        temporal_analysis = self.analyze_temporal_correlations()
        
        # Print overlap analysis
        print("\n1. TEMPORAL OVERLAP ANALYSIS:")
        print("-" * 50)
        
        if overlap_stats:
            print(f"Total overlapping event pairs: {overlap_stats['total_overlaps']:,}")
            print(f"Files with overlaps: {overlap_stats['unique_files_with_overlaps']:,}")
            print(f"Average overlap duration: {overlap_stats['avg_overlap_duration']:.2f}s")
            print(f"Median overlap duration: {overlap_stats['median_overlap_duration']:.2f}s")
            print(f"Maximum overlap duration: {overlap_stats['max_overlap_duration']:.2f}s")
            print(f"Average overlap percentage: {overlap_stats['avg_overlap_percentage']:.1f}%")
        
        # Print overlap patterns
        if overlap_patterns and "overlap_pairs" in overlap_patterns:
            print("\n2. OVERLAP PATTERNS:")
            print("-" * 50)
            
            print("Top 10 most frequently overlapping event pairs:")
            top_pairs = overlap_patterns["overlap_pairs"].head(10)
            for i, (pair, data) in enumerate(top_pairs.iterrows()):
                event1, event2 = pair
                print(f"  {i+1:2}. {event1:20} + {event2:20}: {data['overlap_count']:3} overlaps "
                      f"(avg {data['avg_overlap_duration']:.2f}s)")
        
        # Print correlation analysis
        print("\n3. EVENT CORRELATION ANALYSIS:")
        print("-" * 50)
        
        if correlation_analysis and "top_correlated_pairs" in correlation_analysis:
            print("Top 10 most correlated event pairs (Pearson):")
            for i, pair in enumerate(correlation_analysis["top_correlated_pairs"][:10]):
                print(f"  {i+1:2}. {pair['event1']:20} + {pair['event2']:20}: "
                      f"r={pair['pearson_corr']:.3f}, Jaccard={pair['jaccard_similarity']:.3f}")
        
        # Print temporal analysis
        print("\n4. TEMPORAL PATTERN ANALYSIS:")
        print("-" * 50)
        
        if temporal_analysis and "temporal_patterns" in temporal_analysis:
            print("Events that tend to occur early vs late:")
            
            # Sort events by average position
            sorted_events = sorted(temporal_analysis["temporal_patterns"].items(),
                                  key=lambda x: x[1]["avg_position"])
            
            print("\nEvents that occur early in files:")
            for event, patterns in sorted_events[:5]:
                print(f"  {event:25}: avg position {patterns['avg_position']:.3f}, "
                      f"first event {patterns['first_event_ratio']*100:.1f}%")
            
            print("\nEvents that occur late in files:")
            for event, patterns in sorted_events[-5:]:
                print(f"  {event:25}: avg position {patterns['avg_position']:.3f}, "
                      f"last event {patterns['last_event_ratio']*100:.1f}%")
        
        # Save detailed results
        self.save_overlap_results(overlap_stats, overlap_patterns, correlation_analysis, temporal_analysis)
        
        return {
            "overlap_stats": overlap_stats,
            "overlap_patterns": overlap_patterns,
            "correlation_analysis": correlation_analysis,
            "temporal_analysis": temporal_analysis
        }
    
    def save_overlap_results(self, overlap_stats, overlap_patterns, correlation_analysis, temporal_analysis):
        """Save overlap and correlation analysis results."""
        output_dir = Path("audio_analysis")
        output_dir.mkdir(exist_ok=True)
        
        # Save overlap data
        if hasattr(self, "overlap_df") and len(self.overlap_df) > 0:
            self.overlap_df.to_csv(output_dir / "event_overlaps_detailed.csv", index=False)
        
        # Save overlap patterns
        if overlap_patterns and "overlap_pairs" in overlap_patterns:
            overlap_patterns["overlap_pairs"].to_csv(output_dir / "overlap_patterns.csv")
        
        # Save correlation matrices
        if correlation_analysis:
            if "pearson_correlation" in correlation_analysis:
                correlation_analysis["pearson_correlation"].to_csv(output_dir / "pearson_correlation_matrix.csv")
            
            if "spearman_correlation" in correlation_analysis:
                correlation_analysis["spearman_correlation"].to_csv(output_dir / "spearman_correlation_matrix.csv")
            
            if "jaccard_similarity" in correlation_analysis:
                correlation_analysis["jaccard_similarity"].to_csv(output_dir / "jaccard_similarity_matrix.csv")
            
            if "top_correlated_pairs" in correlation_analysis:
                pairs_df = pd.DataFrame(correlation_analysis["top_correlated_pairs"])
                pairs_df.to_csv(output_dir / "top_correlated_pairs.csv", index=False)
        
        # Save temporal patterns
        if temporal_analysis and "temporal_patterns" in temporal_analysis:
            temporal_df = pd.DataFrame(temporal_analysis["temporal_patterns"]).T
            temporal_df.to_csv(output_dir / "temporal_patterns_by_event.csv")
        
        print(f"\nOverlap and correlation results saved to {output_dir}/")
        print("  - event_overlaps_detailed.csv")
        print("  - overlap_patterns.csv")
        print("  - pearson_correlation_matrix.csv")
        print("  - spearman_correlation_matrix.csv")
        print("  - jaccard_similarity_matrix.csv")
        print("  - top_correlated_pairs.csv")
        print("  - temporal_patterns_by_event.csv")
    
    def create_overlap_plots(self, analysis_results):
        """Create visualization plots for overlap and correlation analysis."""
        output_dir = Path("audio_analysis")
        
        # Set up plotting style
        plt.style.use("default")
        sns.set_palette("husl")
        
        # Create comprehensive plots
        fig, axes = plt.subplots(3, 2, figsize=(16, 18))
        
        # Plot 1: Overlap duration distribution
        if hasattr(self, "overlap_df") and len(self.overlap_df) > 0:
            axes[0, 0].hist(self.overlap_df["overlap_duration"], bins=30, alpha=0.7)
            axes[0, 0].set_title("Distribution of Overlap Durations")
            axes[0, 0].set_xlabel("Overlap Duration (seconds)")
            axes[0, 0].set_ylabel("Frequency")
        
        # Plot 2: Top overlapping pairs
        if (analysis_results["overlap_patterns"] and 
            "overlap_pairs" in analysis_results["overlap_patterns"]):
            
            top_pairs = analysis_results["overlap_patterns"]["overlap_pairs"].head(10)
            pair_labels = [f"{pair[0]}\n+ {pair[1]}" for pair in top_pairs.index]
            
            axes[0, 1].bar(range(len(pair_labels)), top_pairs["overlap_count"])
            axes[0, 1].set_title("Top 10 Most Overlapping Event Pairs")
            axes[0, 1].set_ylabel("Number of Overlaps")
            axes[0, 1].set_xticks(range(len(pair_labels)))
            axes[0, 1].set_xticklabels(pair_labels, rotation=45, ha="right", fontsize=8)
        
        # Plot 3: Correlation heatmap (top 15x15)
        if (analysis_results["correlation_analysis"] and 
            "pearson_correlation" in analysis_results["correlation_analysis"]):
            
            corr_matrix = analysis_results["correlation_analysis"]["pearson_correlation"]
            top_events = corr_matrix.index[:15]
            corr_subset = corr_matrix.loc[top_events, top_events]
            
            sns.heatmap(corr_subset, annot=True, fmt=".2f", cmap="RdBu_r", center=0,
                       ax=axes[1, 0], cbar_kws={"shrink": 0.8})
            axes[1, 0].set_title("Event Correlation Matrix (Top 15)")
        
        # Plot 4: Jaccard similarity heatmap
        if (analysis_results["correlation_analysis"] and 
            "jaccard_similarity" in analysis_results["correlation_analysis"]):
            
            jaccard_matrix = analysis_results["correlation_analysis"]["jaccard_similarity"]
            top_events = jaccard_matrix.index[:15]
            jaccard_subset = jaccard_matrix.loc[top_events, top_events]
            
            sns.heatmap(jaccard_subset, annot=True, fmt=".2f", cmap="YlOrRd",
                       ax=axes[1, 1], cbar_kws={"shrink": 0.8})
            axes[1, 1].set_title("Event Jaccard Similarity (Top 15)")
        
        # Plot 5: Temporal patterns
        if (analysis_results["temporal_analysis"] and 
            "temporal_patterns" in analysis_results["temporal_analysis"]):
            
            temporal_patterns = analysis_results["temporal_analysis"]["temporal_patterns"]
            
            # Plot average position vs duration
            events = list(temporal_patterns.keys())
            positions = [temporal_patterns[event]["avg_position"] for event in events]
            durations = [temporal_patterns[event]["avg_duration"] for event in events]
            
            scatter = axes[2, 0].scatter(positions, durations, alpha=0.7)
            axes[2, 0].set_title("Event Temporal Patterns")
            axes[2, 0].set_xlabel("Average Position in File")
            axes[2, 0].set_ylabel("Average Duration (seconds)")
            
            # Add event labels for extreme points
            for i, event in enumerate(events):
                if positions[i] < 0.1 or positions[i] > 0.9 or durations[i] > 5:
                    axes[2, 0].annotate(event, (positions[i], durations[i]), 
                                      fontsize=8, alpha=0.7)
        
        # Plot 6: Overlap percentage distribution
        if hasattr(self, "overlap_df") and len(self.overlap_df) > 0:
            overlap_pct = (self.overlap_df["overlap_pct_event1"] + 
                          self.overlap_df["overlap_pct_event2"]) / 2
            
            axes[2, 1].hist(overlap_pct, bins=30, alpha=0.7)
            axes[2, 1].set_title("Distribution of Overlap Percentages")
            axes[2, 1].set_xlabel("Average Overlap Percentage")
            axes[2, 1].set_ylabel("Frequency")
        
        plt.tight_layout()
        plt.savefig(output_dir / "event_overlap_correlation_plots.png", dpi=300, bbox_inches="tight")
        plt.close()
        
        print(f"Overlap and correlation plots saved to: {output_dir / 'event_overlap_correlation_plots.png'}")

def main():
    """Main function to run the overlap and correlation analysis."""
    # Initialize analyzer
    desed_path = "/Users/sophon/Software/Python/SSL_SED/DESED"
    analyzer = EventOverlapAnalyzer(desed_path)
    
    # Generate overlap analysis
    analysis_results = analyzer.generate_overlap_report()
    
    # Create overlap plots
    analyzer.create_overlap_plots(analysis_results)
    
    print("\n" + "=" * 80)
    print("OVERLAP AND CORRELATION ANALYSIS COMPLETE")
    print("=" * 80)

if __name__ == "__main__":
    main()

