"""
Advanced Analysis for Overlapping Events

This script provides multiple analysis methods for understanding overlapping events:
1. t-SNE visualization of event embeddings
2. PCA dimensionality reduction
3. Clustering analysis (K-means, DBSCAN)
4. Event similarity analysis
5. Temporal pattern analysis
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import dendrogram, linkage
import pandas as pd
from pathlib import Path
import argparse
from typing import Tuple, List, Dict, Optional
import warnings
warnings.filterwarnings("ignore")

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

def load_data(outputs_path: str, targets_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load training outputs and targets from .npy files"""
    outputs = np.load(outputs_path)
    targets = np.load(targets_path)
    
    print(f"Loaded outputs shape: {outputs.shape}")
    print(f"Loaded targets shape: {targets.shape}")
    
    return outputs, targets

def sigmoid(x: np.ndarray) -> np.ndarray:
    """Apply sigmoid activation to convert logits to probabilities"""
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

def extract_event_embeddings(outputs: np.ndarray, targets: np.ndarray, 
                           threshold: float = 0.5, max_samples: int = 5000) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Extract embeddings for events present in overlapping scenarios
    
    Args:
        outputs: Model outputs (batch_size, time_frames, num_classes)
        targets: Ground truth targets (batch_size, time_frames, num_classes)
        threshold: Threshold for considering an event present
        max_samples: Maximum number of samples to extract (for performance)
        
    Returns:
        Tuple of (embeddings, labels, event_names)
    """
    embeddings = []
    labels = []
    event_names = []
    
    probs = sigmoid(outputs)
    
    for batch_idx in range(outputs.shape[0]):
        for time_frame in range(outputs.shape[1]):
            # Get events present at this time frame
            events_present = targets[batch_idx, time_frame, :] > threshold
            
            # Check if multiple events overlap
            if np.sum(events_present) > 1:
                # Use the raw output logits as embeddings
                embedding = outputs[batch_idx, time_frame, :]
                embeddings.append(embedding)
                
                # Create label for this overlap combination
                present_classes = [DESED_CLASSES[i] for i in range(len(DESED_CLASSES)) if events_present[i]]
                labels.append(len(present_classes))  # Number of overlapping events
                event_names.append("+".join(present_classes))
                
                # Stop if we have enough samples
                if len(embeddings) >= max_samples:
                    break
        if len(embeddings) >= max_samples:
            break
    
    return np.array(embeddings), np.array(labels), event_names

def plot_tsne_analysis(embeddings: np.ndarray, labels: np.ndarray, event_names: List[str],
                      save_path: str = None, perplexity: int = 30, n_iter: int = 1000):
    """
    Create t-SNE visualization of event embeddings
    
    Args:
        embeddings: Event embeddings array
        labels: Labels for coloring (number of overlapping events)
        event_names: Names of event combinations
        save_path: Path to save the plot
        perplexity: t-SNE perplexity parameter
        n_iter: Number of iterations for t-SNE
    """
    print(f"Running t-SNE on {embeddings.shape[0]} samples...")
    
    # Standardize embeddings
    scaler = StandardScaler()
    embeddings_scaled = scaler.fit_transform(embeddings)
    
    # Run t-SNE
    tsne = TSNE(n_components=2, perplexity=perplexity, n_iter=n_iter, random_state=42)
    embeddings_2d = tsne.fit_transform(embeddings_scaled)
    
    # Create the plot
    plt.figure(figsize=(15, 10))
    
    # Color by number of overlapping events
    scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                         c=labels, cmap="viridis", alpha=0.6, s=50)
    
    plt.colorbar(scatter, label="Number of Overlapping Events")
    plt.title("t-SNE Visualization of Overlapping Event Embeddings", fontsize=16)
    plt.xlabel("t-SNE Component 1", fontsize=12)
    plt.ylabel("t-SNE Component 2", fontsize=12)
    
    # Add some annotations for interesting points
    unique_labels = np.unique(labels)
    for label in unique_labels:
        mask = labels == label
        if np.sum(mask) > 0:
            center_x = np.mean(embeddings_2d[mask, 0])
            center_y = np.mean(embeddings_2d[mask, 1])
            plt.annotate(f"{int(label)} events", (center_x, center_y), 
                        fontsize=10, ha="center", va="center",
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"t-SNE plot saved to: {save_path}")
    
    plt.close()
    
    return embeddings_2d

def plot_pca_analysis(embeddings: np.ndarray, labels: np.ndarray, event_names: List[str],
                     save_path: str = None, n_components: int = 2):
    """
    Create PCA visualization of event embeddings
    
    Args:
        embeddings: Event embeddings array
        labels: Labels for coloring
        event_names: Names of event combinations
        save_path: Path to save the plot
        n_components: Number of PCA components
    """
    print(f"Running PCA on {embeddings.shape[0]} samples...")
    
    # Standardize embeddings
    scaler = StandardScaler()
    embeddings_scaled = scaler.fit_transform(embeddings)
    
    # Run PCA
    pca = PCA(n_components=n_components)
    embeddings_pca = pca.fit_transform(embeddings_scaled)
    
    # Create the plot
    plt.figure(figsize=(12, 8))
    
    scatter = plt.scatter(embeddings_pca[:, 0], embeddings_pca[:, 1], 
                         c=labels, cmap="viridis", alpha=0.6, s=50)
    
    plt.colorbar(scatter, label="Number of Overlapping Events")
    plt.title(f"PCA Visualization of Overlapping Event Embeddings\nExplained Variance: {pca.explained_variance_ratio_.sum():.3f}", 
              fontsize=14)
    plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.3f})", fontsize=12)
    plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.3f})", fontsize=12)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"PCA plot saved to: {save_path}")
    
    plt.close()
    
    return embeddings_pca, pca

def clustering_analysis(embeddings: np.ndarray, event_names: List[str],
                       save_path: str = None) -> Dict:
    """
    Perform clustering analysis on event embeddings
    
    Args:
        embeddings: Event embeddings array
        event_names: Names of event combinations
        save_path: Path to save the plot
        
    Returns:
        Dictionary with clustering results
    """
    print("Performing clustering analysis...")
    
    # Standardize embeddings
    scaler = StandardScaler()
    embeddings_scaled = scaler.fit_transform(embeddings)
    
    # Try different numbers of clusters for K-means
    n_clusters_range = range(2, min(11, len(embeddings)//10))
    silhouette_scores = []
    
    for n_clusters in n_clusters_range:
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(embeddings_scaled)
        silhouette_avg = silhouette_score(embeddings_scaled, cluster_labels)
        silhouette_scores.append(silhouette_avg)
    
    # Find optimal number of clusters
    optimal_n_clusters = n_clusters_range[np.argmax(silhouette_scores)]
    
    # Run K-means with optimal number of clusters
    kmeans = KMeans(n_clusters=optimal_n_clusters, random_state=42, n_init=10)
    kmeans_labels = kmeans.fit_predict(embeddings_scaled)
    
    # Run DBSCAN
    dbscan = DBSCAN(eps=0.5, min_samples=5)
    dbscan_labels = dbscan.fit_predict(embeddings_scaled)
    
    # Create visualization
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    
    # K-means results
    scatter1 = axes[0].scatter(embeddings_scaled[:, 0], embeddings_scaled[:, 1], 
                              c=kmeans_labels, cmap="tab10", alpha=0.6, s=50)
    axes[0].set_title(f"K-means Clustering (k={optimal_n_clusters})\nSilhouette Score: {max(silhouette_scores):.3f}")
    axes[0].set_xlabel("Feature 1")
    axes[0].set_ylabel("Feature 2")
    
    # DBSCAN results
    scatter2 = axes[1].scatter(embeddings_scaled[:, 0], embeddings_scaled[:, 1], 
                              c=dbscan_labels, cmap="tab10", alpha=0.6, s=50)
    axes[1].set_title(f"DBSCAN Clustering\nClusters: {len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)}")
    axes[1].set_xlabel("Feature 1")
    axes[1].set_ylabel("Feature 2")
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Clustering plot saved to: {save_path}")
    
    plt.close()
    
    # Analyze cluster contents
    cluster_analysis = {
        "kmeans": {
            "n_clusters": optimal_n_clusters,
            "labels": kmeans_labels,
            "silhouette_score": max(silhouette_scores)
        },
        "dbscan": {
            "labels": dbscan_labels,
            "n_clusters": len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
        }
    }
    
    return cluster_analysis

def event_similarity_analysis(outputs: np.ndarray, targets: np.ndarray,
                            save_path: str = None) -> np.ndarray:
    """
    Analyze similarity between different event types
    
    Args:
        outputs: Model outputs
        targets: Ground truth targets
        save_path: Path to save the plot
        
    Returns:
        Similarity matrix
    """
    print("Computing event similarity matrix...")
    
    # Get average embeddings for each event type
    event_embeddings = []
    
    for event_idx in range(len(DESED_CLASSES)):
        # Find all frames where this event is present
        event_mask = targets[:, :, event_idx] > 0.5
        if np.sum(event_mask) > 0:
            # Get average embedding for this event
            event_outputs = outputs[event_mask]
            avg_embedding = np.mean(event_outputs, axis=0)
            event_embeddings.append(avg_embedding)
        else:
            # If event never appears, use zero embedding
            event_embeddings.append(np.zeros(outputs.shape[2]))
    
    event_embeddings = np.array(event_embeddings)
    
    # Compute cosine similarity matrix
    similarity_matrix = np.zeros((len(DESED_CLASSES), len(DESED_CLASSES)))
    
    for i in range(len(DESED_CLASSES)):
        for j in range(len(DESED_CLASSES)):
            # Cosine similarity
            dot_product = np.dot(event_embeddings[i], event_embeddings[j])
            norm_i = np.linalg.norm(event_embeddings[i])
            norm_j = np.linalg.norm(event_embeddings[j])
            
            if norm_i > 0 and norm_j > 0:
                similarity_matrix[i, j] = dot_product / (norm_i * norm_j)
            else:
                similarity_matrix[i, j] = 0
    
    # Create heatmap
    plt.figure(figsize=(12, 10))
    
    mask = np.zeros_like(similarity_matrix, dtype=bool)
    np.fill_diagonal(mask, True)
    
    sns.heatmap(similarity_matrix, 
                annot=True, 
                fmt=".3f",
                cmap="RdYlBu_r",
                xticklabels=DESED_CLASSES,
                yticklabels=DESED_CLASSES,
                mask=mask,
                vmin=-1, vmax=1,
                cbar_kws={"label": "Cosine Similarity"})
    
    plt.title("Event Similarity Matrix\n(Based on Average Embeddings)", fontsize=14)
    plt.xlabel("Events", fontsize=12)
    plt.ylabel("Events", fontsize=12)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Similarity matrix saved to: {save_path}")
    
    plt.close()
    
    return similarity_matrix

def temporal_pattern_analysis(outputs: np.ndarray, targets: np.ndarray,
                            save_path: str = None):
    """
    Analyze temporal patterns in overlapping events
    
    Args:
        outputs: Model outputs
        targets: Ground truth targets
        save_path: Path to save the plot
    """
    print("Analyzing temporal patterns...")
    
    # Count overlaps by time position
    time_overlap_counts = np.zeros(outputs.shape[1])  # 500 time frames
    
    for batch_idx in range(outputs.shape[0]):
        for time_frame in range(outputs.shape[1]):
            events_present = targets[batch_idx, time_frame, :] > 0.5
            if np.sum(events_present) > 1:
                time_overlap_counts[time_frame] += 1
    
    # Create temporal plot
    plt.figure(figsize=(15, 6))
    
    plt.plot(time_overlap_counts, linewidth=2)
    plt.title("Temporal Distribution of Overlapping Events", fontsize=14)
    plt.xlabel("Time Frame", fontsize=12)
    plt.ylabel("Number of Overlapping Instances", fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # Add some statistics
    mean_overlaps = np.mean(time_overlap_counts)
    std_overlaps = np.std(time_overlap_counts)
    
    plt.axhline(y=mean_overlaps, color="red", linestyle="--", alpha=0.7, 
                label=f"Mean: {mean_overlaps:.1f}")
    plt.axhline(y=mean_overlaps + std_overlaps, color="orange", linestyle="--", alpha=0.7,
                label=f"Mean + Std: {mean_overlaps + std_overlaps:.1f}")
    plt.axhline(y=mean_overlaps - std_overlaps, color="orange", linestyle="--", alpha=0.7,
                label=f"Mean - Std: {mean_overlaps - std_overlaps:.1f}")
    
    plt.legend()
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Temporal analysis saved to: {save_path}")
    
    plt.close()

def main():
    """Main function to run all advanced analyses"""
    parser = argparse.ArgumentParser(description="Advanced analysis for overlapping events")
    parser.add_argument("--outputs", default="autrainer/epoch_65/train_outputs.npy", 
                       help="Path to train_outputs.npy")
    parser.add_argument("--targets", default="autrainer/epoch_65/train_targets.npy",
                       help="Path to train_targets.npy") 
    parser.add_argument("--threshold", type=float, default=0.5,
                       help="Threshold for ground truth events")
    parser.add_argument("--max_samples", type=int, default=5000,
                       help="Maximum number of samples for analysis (for performance)")
    parser.add_argument("--output_dir", default="advanced_analysis",
                       help="Directory to save analysis results")
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Load data
    print("Loading data...")
    outputs, targets = load_data(args.outputs, args.targets)
    
    # Extract event embeddings
    print("Extracting event embeddings...")
    embeddings, labels, event_names = extract_event_embeddings(outputs, targets, args.threshold, args.max_samples)
    
    print(f"Extracted {len(embeddings)} overlapping event instances")
    print(f"Event combinations range from {min(labels)} to {max(labels)} overlapping events")
    
    # Run all analyses
    print("\n" + "="*60)
    print("RUNNING ADVANCED ANALYSES")
    print("="*60)
    
    # 1. t-SNE Analysis
    print("\n1. t-SNE Analysis...")
    tsne_embeddings = plot_tsne_analysis(
        embeddings, labels, event_names,
        save_path=output_dir / "tsne_analysis.png"
    )
    
    # 2. PCA Analysis
    print("\n2. PCA Analysis...")
    pca_embeddings, pca_model = plot_pca_analysis(
        embeddings, labels, event_names,
        save_path=output_dir / "pca_analysis.png"
    )
    
    # 3. Clustering Analysis
    print("\n3. Clustering Analysis...")
    cluster_results = clustering_analysis(
        embeddings, event_names,
        save_path=output_dir / "clustering_analysis.png"
    )
    
    # 4. Event Similarity Analysis
    print("\n4. Event Similarity Analysis...")
    similarity_matrix = event_similarity_analysis(
        outputs, targets,
        save_path=output_dir / "event_similarity.png"
    )
    
    # 5. Temporal Pattern Analysis
    print("\n5. Temporal Pattern Analysis...")
    temporal_pattern_analysis(
        outputs, targets,
        save_path=output_dir / "temporal_patterns.png"
    )
    
    # Print summary
    print("\n" + "="*60)
    print("ANALYSIS SUMMARY")
    print("="*60)
    
    print(f"Total overlapping instances analyzed: {len(embeddings)}")
    print(f"PCA explained variance ratio: {pca_model.explained_variance_ratio_[:2]}")
    print(f"Optimal K-means clusters: {cluster_results['kmeans']['n_clusters']}")
    print(f"K-means silhouette score: {cluster_results['kmeans']['silhouette_score']:.3f}")
    print(f"DBSCAN clusters found: {cluster_results['dbscan']['n_clusters']}")
    
    # Find most similar event pairs
    similarity_no_diag = similarity_matrix.copy()
    np.fill_diagonal(similarity_no_diag, -1)  # Remove diagonal
    max_sim_idx = np.unravel_index(np.argmax(similarity_no_diag), similarity_matrix.shape)
    
    print(f"Most similar event pair: {DESED_CLASSES[max_sim_idx[0]]} ↔ {DESED_CLASSES[max_sim_idx[1]]} "
          f"(similarity: {similarity_matrix[max_sim_idx]:.3f})")
    
    print(f"\nAll analysis results saved to: {output_dir}")
    print("Analysis complete!")

if __name__ == "__main__":
    main()
