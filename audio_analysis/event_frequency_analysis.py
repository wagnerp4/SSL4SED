#!/usr/bin/env python3
"""
DESED Event Frequency and Distribution Analysis

This script provides detailed analysis of event frequencies, distributions,
and temporal characteristics across different dataset splits.
"""

import os
import pandas as pd
import numpy as np
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats
import warnings
warnings.filterwarnings("ignore")

class EventFrequencyAnalyzer:
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
        
        self.all_data = {}
        self.event_stats = {}
        
    def load_all_annotations(self):
        """Load all annotation files and combine them."""
        print("Loading all annotation files...")
        
        for split_name, filename in self.dataset_splits.items():
            file_path = self.annotations_path / filename
            
            if not file_path.exists():
                print(f"Warning: {file_path} does not exist")
                continue
                
            df = pd.read_csv(file_path, sep="\t")
            
            # Handle different annotation formats
            if "event_labels" in df.columns:
                # Weak annotations format (comma-separated labels)
                df["event_label"] = df["event_labels"].str.split(",")
                df = df.explode("event_label")
                df["event_label"] = df["event_label"].str.strip()
                df["onset"] = 0.0
                df["offset"] = 10.0
                df["split"] = split_name
            elif "onset" in df.columns and "offset" in df.columns:
                # Strong annotations format (with timing)
                df["split"] = split_name
            else:
                print(f"Warning: Unknown format for {split_name}")
                continue
                
            self.all_data[split_name] = df
            print(f"  Loaded {len(df)} events from {split_name}")
        
        # Combine all data
        self.combined_data = pd.concat(self.all_data.values(), ignore_index=True)
        print(f"Total combined events: {len(self.combined_data)}")
        
    def analyze_event_frequency_distribution(self):
        """Analyze frequency distribution of events."""
        print("\nAnalyzing event frequency distributions...")
        
        # Global frequency analysis
        event_counts = self.combined_data["event_label"].value_counts()
        
        # Calculate frequency statistics
        freq_stats = {
            "total_events": len(self.combined_data),
            "unique_events": len(event_counts),
            "most_frequent": event_counts.iloc[0],
            "least_frequent": event_counts.iloc[-1],
            "mean_frequency": event_counts.mean(),
            "median_frequency": event_counts.median(),
            "std_frequency": event_counts.std(),
            "frequency_range": event_counts.max() - event_counts.min()
        }
        
        # Frequency distribution analysis
        freq_distribution = {
            "very_rare": len(event_counts[event_counts <= 10]),
            "rare": len(event_counts[(event_counts > 10) & (event_counts <= 50)]),
            "moderate": len(event_counts[(event_counts > 50) & (event_counts <= 200)]),
            "common": len(event_counts[(event_counts > 200) & (event_counts <= 1000)]),
            "very_common": len(event_counts[event_counts > 1000])
        }
        
        # Per-split frequency analysis
        split_frequencies = {}
        for split_name, df in self.all_data.items():
            split_counts = df["event_label"].value_counts()
            split_frequencies[split_name] = {
                "total_events": len(df),
                "unique_events": len(split_counts),
                "top_5_events": split_counts.head(5).to_dict(),
                "frequency_stats": {
                    "mean": split_counts.mean(),
                    "median": split_counts.median(),
                    "std": split_counts.std()
                }
            }
        
        self.event_stats = {
            "global_frequency": event_counts.to_dict(),
            "frequency_statistics": freq_stats,
            "frequency_distribution": freq_distribution,
            "split_frequencies": split_frequencies
        }
        
        return self.event_stats
    
    def analyze_temporal_distribution(self):
        """Analyze temporal distribution of events."""
        print("\nAnalyzing temporal distributions...")
        
        # Filter data with timing information
        timed_data = self.combined_data[
            (self.combined_data["onset"] >= 0) & 
            (self.combined_data["offset"] > self.combined_data["onset"])
        ].copy()
        
        if len(timed_data) == 0:
            print("No timing information available")
            return {}
        
        # Calculate durations
        timed_data["duration"] = timed_data["offset"] - timed_data["onset"]
        
        # Global temporal statistics
        temporal_stats = {
            "total_timed_events": len(timed_data),
            "avg_duration": timed_data["duration"].mean(),
            "median_duration": timed_data["duration"].median(),
            "min_duration": timed_data["duration"].min(),
            "max_duration": timed_data["duration"].max(),
            "std_duration": timed_data["duration"].std(),
            "duration_percentiles": {
                "25th": timed_data["duration"].quantile(0.25),
                "75th": timed_data["duration"].quantile(0.75),
                "90th": timed_data["duration"].quantile(0.90),
                "95th": timed_data["duration"].quantile(0.95)
            }
        }
        
        # Duration by event type
        duration_by_event = timed_data.groupby("event_label")["duration"].agg([
            "count", "mean", "median", "std", "min", "max"
        ]).round(3)
        
        # Onset distribution (when events start)
        onset_stats = {
            "avg_onset": timed_data["onset"].mean(),
            "median_onset": timed_data["onset"].median(),
            "onset_std": timed_data["onset"].std()
        }
        
        # Temporal patterns by split
        split_temporal = {}
        for split_name in self.all_data.keys():
            split_timed = timed_data[timed_data["split"] == split_name]
            if len(split_timed) > 0:
                split_temporal[split_name] = {
                    "avg_duration": split_timed["duration"].mean(),
                    "median_duration": split_timed["duration"].median(),
                    "avg_onset": split_timed["onset"].mean(),
                    "event_count": len(split_timed)
                }
        
        temporal_analysis = {
            "global_temporal": temporal_stats,
            "duration_by_event": duration_by_event.to_dict(),
            "onset_statistics": onset_stats,
            "split_temporal": split_temporal
        }
        
        return temporal_analysis
    
    def analyze_event_co_occurrence(self):
        """Analyze which events tend to co-occur in the same files."""
        print("\nAnalyzing event co-occurrence patterns...")
        
        # Group events by filename
        file_events = self.combined_data.groupby("filename")["event_label"].apply(list)
        
        # Calculate co-occurrence matrix
        all_events = set(self.combined_data["event_label"].unique())
        co_occurrence_matrix = pd.DataFrame(0, index=all_events, columns=all_events)
        
        # Count co-occurrences
        for filename, events in file_events.items():
            unique_events = list(set(events))  # Remove duplicates within file
            for i, event1 in enumerate(unique_events):
                for j, event2 in enumerate(unique_events):
                    if i != j:
                        co_occurrence_matrix.loc[event1, event2] += 1
        
        # Calculate co-occurrence statistics
        co_occurrence_stats = {}
        
        # Most co-occurring pairs
        co_occurrence_pairs = []
        for event1 in all_events:
            for event2 in all_events:
                if event1 != event2 and co_occurrence_matrix.loc[event1, event2] > 0:
                    co_occurrence_pairs.append({
                        "event1": event1,
                        "event2": event2,
                        "co_occurrence_count": co_occurrence_matrix.loc[event1, event2]
                    })
        
        co_occurrence_pairs = sorted(co_occurrence_pairs, 
                                   key=lambda x: x["co_occurrence_count"], 
                                   reverse=True)
        
        # Calculate conditional probabilities
        event_totals = self.combined_data["event_label"].value_counts()
        conditional_probs = {}
        
        for pair in co_occurrence_pairs[:20]:  # Top 20 pairs
            event1, event2 = pair["event1"], pair["event2"]
            co_count = pair["co_occurrence_count"]
            
            # P(event2 | event1) = co_occurrence_count / event1_total
            prob_given_event1 = co_count / event_totals[event1]
            prob_given_event2 = co_count / event_totals[event2]
            
            conditional_probs[f"{event1} -> {event2}"] = {
                "co_occurrence_count": co_count,
                "prob_event2_given_event1": prob_given_event1,
                "prob_event1_given_event2": prob_given_event2
            }
        
        co_occurrence_analysis = {
            "co_occurrence_matrix": co_occurrence_matrix,
            "top_co_occurring_pairs": co_occurrence_pairs[:20],
            "conditional_probabilities": conditional_probs
        }
        
        return co_occurrence_analysis
    
    def analyze_polyphonicity_patterns(self):
        """Analyze polyphonicity patterns and complexity."""
        print("\nAnalyzing polyphonicity patterns...")
        
        # Group by filename to analyze polyphonicity
        file_analysis = self.combined_data.groupby("filename").agg({
            "event_label": ["count", "nunique", list],
            "split": "first"
        }).round(3)
        
        file_analysis.columns = ["total_events", "unique_events", "event_list", "split"]
        
        # Calculate polyphonicity metrics
        polyphonicity_stats = {
            "total_files": len(file_analysis),
            "monophonic_files": len(file_analysis[file_analysis["unique_events"] == 1]),
            "polyphonic_files": len(file_analysis[file_analysis["unique_events"] > 1]),
            "avg_events_per_file": file_analysis["total_events"].mean(),
            "avg_unique_events_per_file": file_analysis["unique_events"].mean(),
            "max_events_per_file": file_analysis["total_events"].max(),
            "max_unique_events_per_file": file_analysis["unique_events"].max()
        }
        
        # Polyphonicity distribution
        polyphonicity_dist = file_analysis["unique_events"].value_counts().sort_index()
        
        # Most complex files (highest polyphonicity)
        most_complex_files = file_analysis.nlargest(10, "unique_events")[
            ["unique_events", "total_events", "event_list", "split"]
        ]
        
        # Polyphonicity by split
        polyphonicity_by_split = {}
        for split_name in self.all_data.keys():
            split_files = file_analysis[file_analysis["split"] == split_name]
            if len(split_files) > 0:
                polyphonicity_by_split[split_name] = {
                    "total_files": len(split_files),
                    "polyphonic_files": len(split_files[split_files["unique_events"] > 1]),
                    "polyphonicity_ratio": len(split_files[split_files["unique_events"] > 1]) / len(split_files),
                    "avg_unique_events": split_files["unique_events"].mean(),
                    "max_unique_events": split_files["unique_events"].max()
                }
        
        polyphonicity_analysis = {
            "global_stats": polyphonicity_stats,
            "polyphonicity_distribution": polyphonicity_dist.to_dict(),
            "most_complex_files": most_complex_files.to_dict(),
            "polyphonicity_by_split": polyphonicity_by_split
        }
        
        return polyphonicity_analysis
    
    def generate_detailed_report(self):
        """Generate detailed frequency and distribution report."""
        print("=" * 80)
        print("DESED EVENT FREQUENCY AND DISTRIBUTION ANALYSIS")
        print("=" * 80)
        
        # Load all data
        self.load_all_annotations()
        
        # Run all analyses
        freq_stats = self.analyze_event_frequency_distribution()
        temporal_stats = self.analyze_temporal_distribution()
        co_occurrence_stats = self.analyze_event_co_occurrence()
        polyphonicity_stats = self.analyze_polyphonicity_patterns()
        
        # Print frequency analysis
        print("\n1. EVENT FREQUENCY ANALYSIS:")
        print("-" * 50)
        
        freq_info = freq_stats["frequency_statistics"]
        print(f"Total events: {freq_info['total_events']:,}")
        print(f"Unique event types: {freq_info['unique_events']}")
        print(f"Most frequent event: {freq_stats['global_frequency'].get(list(freq_stats['global_frequency'].keys())[0], 'N/A')} occurrences")
        print(f"Mean frequency: {freq_info['mean_frequency']:.1f}")
        print(f"Median frequency: {freq_info['median_frequency']:.1f}")
        print(f"Frequency std: {freq_info['std_frequency']:.1f}")
        
        print("\nFrequency distribution categories:")
        dist_info = freq_stats["frequency_distribution"]
        for category, count in dist_info.items():
            print(f"  {category:12}: {count:3} events")
        
        # Print temporal analysis
        if temporal_stats:
            print("\n2. TEMPORAL DISTRIBUTION ANALYSIS:")
            print("-" * 50)
            
            temp_info = temporal_stats["global_temporal"]
            print(f"Timed events: {temp_info['total_timed_events']:,}")
            print(f"Average duration: {temp_info['avg_duration']:.2f}s")
            print(f"Median duration: {temp_info['median_duration']:.2f}s")
            print(f"Duration range: {temp_info['min_duration']:.2f}s - {temp_info['max_duration']:.2f}s")
            
            print("\nDuration percentiles:")
            for percentile, value in temp_info["duration_percentiles"].items():
                print(f"  {percentile:6}: {value:.2f}s")
        
        # Print co-occurrence analysis
        print("\n3. EVENT CO-OCCURRENCE ANALYSIS:")
        print("-" * 50)
        
        print("Top 10 most co-occurring event pairs:")
        for i, pair in enumerate(co_occurrence_stats["top_co_occurring_pairs"][:10]):
            print(f"  {i+1:2}. {pair['event1']:20} + {pair['event2']:20}: {pair['co_occurrence_count']:4} times")
        
        print("\nTop 10 conditional probabilities:")
        sorted_probs = sorted(co_occurrence_stats["conditional_probabilities"].items(),
                            key=lambda x: x[1]["prob_event2_given_event1"], reverse=True)
        for i, (pair, probs) in enumerate(sorted_probs[:10]):
            print(f"  {i+1:2}. {pair:40}: {probs['prob_event2_given_event1']:.3f}")
        
        # Print polyphonicity analysis
        print("\n4. POLYPHONICITY ANALYSIS:")
        print("-" * 50)
        
        poly_info = polyphonicity_stats["global_stats"]
        print(f"Total files: {poly_info['total_files']:,}")
        print(f"Monophonic files: {poly_info['monophonic_files']:,} ({poly_info['monophonic_files']/poly_info['total_files']*100:.1f}%)")
        print(f"Polyphonic files: {poly_info['polyphonic_files']:,} ({poly_info['polyphonic_files']/poly_info['total_files']*100:.1f}%)")
        print(f"Average events per file: {poly_info['avg_events_per_file']:.2f}")
        print(f"Average unique events per file: {poly_info['avg_unique_events_per_file']:.2f}")
        print(f"Maximum events in a file: {poly_info['max_events_per_file']}")
        print(f"Maximum unique events in a file: {poly_info['max_unique_events_per_file']}")
        
        print("\nPolyphonicity distribution:")
        for unique_count, file_count in list(polyphonicity_stats["polyphonicity_distribution"].items())[:10]:
            print(f"  {unique_count:2} unique events: {file_count:4} files")
        
        # Save detailed results
        self.save_detailed_results(freq_stats, temporal_stats, co_occurrence_stats, polyphonicity_stats)
        
        return {
            "frequency": freq_stats,
            "temporal": temporal_stats,
            "co_occurrence": co_occurrence_stats,
            "polyphonicity": polyphonicity_stats
        }
    
    def save_detailed_results(self, freq_stats, temporal_stats, co_occurrence_stats, polyphonicity_stats):
        """Save detailed analysis results to files."""
        output_dir = Path("audio_analysis")
        output_dir.mkdir(exist_ok=True)
        
        # Save frequency statistics
        freq_df = pd.DataFrame(list(freq_stats["global_frequency"].items()),
                             columns=["event_label", "frequency"])
        freq_df["percentage"] = freq_df["frequency"] / freq_df["frequency"].sum() * 100
        freq_df = freq_df.sort_values("frequency", ascending=False)
        freq_df.to_csv(output_dir / "event_frequency_detailed.csv", index=False)
        
        # Save temporal statistics
        if temporal_stats and "duration_by_event" in temporal_stats:
            temp_df = pd.DataFrame(temporal_stats["duration_by_event"])
            temp_df.to_csv(output_dir / "event_duration_statistics.csv")
        
        # Save co-occurrence matrix
        if "co_occurrence_matrix" in co_occurrence_stats:
            co_occurrence_stats["co_occurrence_matrix"].to_csv(output_dir / "event_co_occurrence_matrix.csv")
        
        # Save top co-occurring pairs
        if "top_co_occurring_pairs" in co_occurrence_stats:
            pairs_df = pd.DataFrame(co_occurrence_stats["top_co_occurring_pairs"])
            pairs_df.to_csv(output_dir / "top_co_occurring_pairs.csv", index=False)
        
        # Save polyphonicity statistics
        poly_df = pd.DataFrame(list(polyphonicity_stats["polyphonicity_distribution"].items()),
                              columns=["unique_events_per_file", "file_count"])
        poly_df.to_csv(output_dir / "polyphonicity_distribution.csv", index=False)
        
        print(f"\nDetailed results saved to {output_dir}/")
        print("  - event_frequency_detailed.csv")
        print("  - event_duration_statistics.csv")
        print("  - event_co_occurrence_matrix.csv")
        print("  - top_co_occurring_pairs.csv")
        print("  - polyphonicity_distribution.csv")
    
    def create_frequency_plots(self, analysis_results):
        """Create detailed frequency and distribution plots."""
        output_dir = Path("audio_analysis")
        
        # Set up plotting style
        plt.style.use("default")
        sns.set_palette("husl")
        
        # Create comprehensive plots
        fig, axes = plt.subplots(3, 2, figsize=(16, 18))
        
        # Plot 1: Event frequency distribution (log scale)
        freq_data = analysis_results["frequency"]["global_frequency"]
        events = list(freq_data.keys())[:20]  # Top 20 events
        frequencies = list(freq_data.values())[:20]
        
        axes[0, 0].bar(range(len(events)), frequencies)
        axes[0, 0].set_title("Top 20 Most Frequent Events")
        axes[0, 0].set_ylabel("Frequency")
        axes[0, 0].set_xticks(range(len(events)))
        axes[0, 0].set_xticklabels(events, rotation=45, ha="right")
        
        # Plot 2: Frequency distribution histogram
        all_frequencies = list(freq_data.values())
        axes[0, 1].hist(all_frequencies, bins=30, alpha=0.7)
        axes[0, 1].set_title("Distribution of Event Frequencies")
        axes[0, 1].set_xlabel("Frequency")
        axes[0, 1].set_ylabel("Number of Events")
        axes[0, 1].set_yscale("log")
        
        # Plot 3: Duration distribution (if available)
        if analysis_results["temporal"] and "global_temporal" in analysis_results["temporal"]:
            # This would need actual duration data - placeholder for now
            axes[1, 0].text(0.5, 0.5, "Duration Analysis\n(requires timing data)", 
                           ha="center", va="center", transform=axes[1, 0].transAxes)
            axes[1, 0].set_title("Event Duration Distribution")
        
        # Plot 4: Polyphonicity distribution
        poly_dist = analysis_results["polyphonicity"]["polyphonicity_distribution"]
        unique_counts = list(poly_dist.keys())[:15]
        file_counts = list(poly_dist.values())[:15]
        
        axes[1, 1].bar(unique_counts, file_counts)
        axes[1, 1].set_title("Polyphonicity Distribution")
        axes[1, 1].set_xlabel("Unique Events per File")
        axes[1, 1].set_ylabel("Number of Files")
        
        # Plot 5: Co-occurrence heatmap (top 15x15)
        if "co_occurrence_matrix" in analysis_results["co_occurrence"]:
            co_matrix = analysis_results["co_occurrence"]["co_occurrence_matrix"]
            top_events = co_matrix.index[:15]
            co_subset = co_matrix.loc[top_events, top_events]
            
            sns.heatmap(co_subset, annot=True, fmt="d", cmap="YlOrRd", 
                       ax=axes[2, 0], cbar_kws={"shrink": 0.8})
            axes[2, 0].set_title("Event Co-occurrence Matrix (Top 15)")
        
        # Plot 6: Split comparison
        split_freq = analysis_results["frequency"]["split_frequencies"]
        splits = list(split_freq.keys())
        unique_events = [split_freq[split]["unique_events"] for split in splits]
        
        axes[2, 1].bar(splits, unique_events)
        axes[2, 1].set_title("Unique Events by Dataset Split")
        axes[2, 1].set_ylabel("Number of Unique Events")
        axes[2, 1].tick_params(axis="x", rotation=45)
        
        plt.tight_layout()
        plt.savefig(output_dir / "event_frequency_analysis_plots.png", dpi=300, bbox_inches="tight")
        plt.close()
        
        print(f"Frequency analysis plots saved to: {output_dir / 'event_frequency_analysis_plots.png'}")

def main():
    """Main function to run the event frequency analysis."""
    # Initialize analyzer
    desed_path = "/Users/sophon/Software/Python/SSL_SED/DESED"
    analyzer = EventFrequencyAnalyzer(desed_path)
    
    # Generate detailed analysis
    analysis_results = analyzer.generate_detailed_report()
    
    # Create frequency plots
    analyzer.create_frequency_plots(analysis_results)
    
    print("\n" + "=" * 80)
    print("EVENT FREQUENCY ANALYSIS COMPLETE")
    print("=" * 80)

if __name__ == "__main__":
    main()
