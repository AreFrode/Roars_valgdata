"""
Clustering analysis module for election prediction analysis.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple, List
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import pdist, squareform
import networkx as nx

from .config import PARTIES, DEFAULT_N_CLUSTERS, PCA_COMPONENTS, TSNE_PERPLEXITY


def prepare_clustering_data(predictions_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
    """
    Prepare prediction data for clustering analysis.

    Args:
        predictions_df: DataFrame with predictions

    Returns:
        Tuple of (X, X_scaled, respondent_names, party_columns)
    """
    # Create feature matrix (respondents as rows, parties as columns)
    X = predictions_df[PARTIES].values
    respondent_names = predictions_df['Eget navn'].tolist()

    # Standardize the data (important for clustering)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X, X_scaled, respondent_names, PARTIES


def perform_clustering(X_scaled: np.ndarray,
                       respondent_names: List[str],
                       n_clusters: int = DEFAULT_N_CLUSTERS) -> Dict[str, Any]:
    """
    Perform multiple types of clustering analysis.

    Args:
        X_scaled: Scaled feature matrix
        respondent_names: List of respondent names
        n_clusters: Number of clusters

    Returns:
        Dictionary with clustering results
    """
    # K-Means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    kmeans_labels = kmeans.fit_predict(X_scaled)

    # Hierarchical clustering
    hierarchical = AgglomerativeClustering(n_clusters=n_clusters)
    hierarchical_labels = hierarchical.fit_predict(X_scaled)

    # PCA for dimensionality reduction
    pca = PCA(n_components=PCA_COMPONENTS)
    X_pca = pca.fit_transform(X_scaled)

    # t-SNE for non-linear dimensionality reduction
    perplexity = min(TSNE_PERPLEXITY, len(respondent_names) - 1)
    tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
    X_tsne = tsne.fit_transform(X_scaled)

    return {
        'kmeans_labels': kmeans_labels,
        'hierarchical_labels': hierarchical_labels,
        'X_pca': X_pca,
        'X_tsne': X_tsne,
        'pca': pca,
        'tsne': tsne,
        'kmeans': kmeans
    }


def calculate_similarity_network(X: np.ndarray,
                                 respondent_names: List[str],
                                 threshold_percentile: float = 75) -> Tuple[nx.Graph, np.ndarray, np.ndarray]:
    """
    Create a similarity network between respondents.

    Args:
        X: Feature matrix
        respondent_names: List of respondent names
        threshold_percentile: Percentile for similarity threshold

    Returns:
        Tuple of (G, similarity_matrix, correlation_matrix)
    """
    # Calculate pairwise distances (using correlation for similarity)
    correlation_matrix = np.corrcoef(X)

    # Convert correlation to distance (1 - correlation)
    distance_matrix = 1 - np.abs(correlation_matrix)

    # Create similarity matrix (higher = more similar)
    similarity_matrix = 1 - distance_matrix

    # Create network graph
    G = nx.Graph()

    # Add nodes
    for name in respondent_names:
        G.add_node(name)

    # Add edges for similar respondents (above threshold)
    threshold = np.percentile(similarity_matrix[np.triu_indices_from(similarity_matrix, k=1)],
                              threshold_percentile)

    for i, name1 in enumerate(respondent_names):
        for j, name2 in enumerate(respondent_names[i+1:], i+1):
            similarity = similarity_matrix[i, j]
            if similarity > threshold:
                G.add_edge(name1, name2, weight=similarity)

    return G, similarity_matrix, correlation_matrix


def analyze_cluster_characteristics(predictions_df: pd.DataFrame,
                                    clustering_results: Dict[str, Any],
                                    party_columns: List[str]) -> Dict[int, Dict[str, Any]]:
    """
    Analyze what makes each cluster unique.

    Args:
        predictions_df: DataFrame with predictions
        clustering_results: Results from clustering
        party_columns: List of party names

    Returns:
        Dictionary with cluster analysis
    """
    X, X_scaled, respondent_names, _ = prepare_clustering_data(predictions_df)
    kmeans_labels = clustering_results['kmeans_labels']

    cluster_analysis = {}

    for cluster_id in np.unique(kmeans_labels):
        cluster_mask = kmeans_labels == cluster_id
        cluster_members = [respondent_names[i]
                           for i in range(len(respondent_names)) if cluster_mask[i]]

        # Calculate cluster centroid (average predictions)
        cluster_data = X[cluster_mask]
        cluster_mean = cluster_data.mean(axis=0)

        # Find parties this cluster over/under-estimates compared to overall average
        overall_mean = X.mean(axis=0)
        relative_bias = cluster_mean - overall_mean

        # Rank parties by how much this cluster differs from average
        party_deviations = [(party_columns[i], relative_bias[i])
                            for i in range(len(party_columns))]
        party_deviations.sort(key=lambda x: abs(x[1]), reverse=True)

        cluster_analysis[cluster_id] = {
            'members': cluster_members,
            'size': len(cluster_members),
            'average_predictions': dict(zip(party_columns, cluster_mean)),
            'relative_bias': dict(zip(party_columns, relative_bias)),
            # Top 3 distinctive characteristics
            'top_deviations': party_deviations[:3],
            # Internal consistency
            'cluster_variance': cluster_data.var(axis=0).mean()
        }

    return cluster_analysis


def get_clustering_insights(predictions_df: pd.DataFrame,
                            clustering_results: Dict[str, Any],
                            cluster_analysis: Dict[int, Dict[str, Any]],
                            correlation_matrix: np.ndarray,
                            respondent_names: List[str]) -> str:
    """
    Generate text insights about the clustering patterns.

    Args:
        predictions_df: DataFrame with predictions
        clustering_results: Clustering results
        cluster_analysis: Cluster analysis
        correlation_matrix: Correlation matrix
        respondent_names: List of respondent names

    Returns:
        String with clustering insights
    """
    insights = "🎭 POLITICAL CLUSTERING INSIGHTS:\n\n"

    # Overall clustering quality
    pca = clustering_results['pca']
    explained_var = sum(pca.explained_variance_ratio_)

    insights += f"📊 PCA Analysis: First 2 components explain {explained_var:.1%} of prediction variance\n"
    if explained_var > 0.7:
        insights += "   → Strong clustering patterns detected!\n"
    elif explained_var > 0.5:
        insights += "   → Moderate clustering patterns found\n"
    else:
        insights += "   → Weak clustering - predictions are quite diverse\n"

    # Cluster characteristics
    n_clusters = len(cluster_analysis)
    insights += f"\n🎯 Found {n_clusters} distinct prediction groups:\n\n"

    cluster_names = ["The Realists", "The Optimists",
                     "The Contrarians", "The Cautious", "The Wildcards"]

    for cluster_id, analysis in cluster_analysis.items():
        name = cluster_names[cluster_id] if cluster_id < len(
            cluster_names) else f"Group {cluster_id+1}"
        insights += f"🎭 {name} ({analysis['size']} members):\n"
        insights += f"   Members: {', '.join(analysis['members'])}\n"

        # Top deviations
        insights += f"   Distinctive traits:\n"
        for party, deviation in analysis['top_deviations']:
            direction = "higher" if deviation > 0 else "lower"
            insights += f"   • Predicts {party} {abs(deviation):.1f}pp {direction} than average\n"
        consistency = "very consistent" if analysis['cluster_variance'] < 1 else "somewhat varied" if analysis[
            'cluster_variance'] < 3 else "quite diverse"
        insights += f"   Internal consistency: {consistency}\n\n"

    # Find most similar pair
    np.fill_diagonal(correlation_matrix, -1)  # Ignore self-correlation
    most_similar_idx = np.unravel_index(
        np.argmax(correlation_matrix), correlation_matrix.shape)
    most_similar_corr = correlation_matrix[most_similar_idx]
    most_similar_pair = (
        respondent_names[most_similar_idx[0]], respondent_names[most_similar_idx[1]])

    insights += f"👯 MOST SIMILAR PREDICTORS: {most_similar_pair[0]} & {most_similar_pair[1]} "
    insights += f"(correlation: {most_similar_corr:.3f})\n"
    # Find least similar pair
    correlation_matrix_temp = np.where(
        correlation_matrix > 0, correlation_matrix, np.inf)
    most_different_idx = np.unravel_index(
        np.argmin(correlation_matrix_temp), correlation_matrix_temp.shape)
    most_different_corr = correlation_matrix_temp[most_different_idx]
    most_different_pair = (
        respondent_names[most_different_idx[0]], respondent_names[most_different_idx[1]])

    insights += f"🔄 MOST DIFFERENT PREDICTORS: {most_different_pair[0]} & {most_different_pair[1]} "
    insights += f"(correlation: {most_different_corr:.3f})\n"
    return insights


def find_political_archetypes(cluster_analysis: Dict[int, Dict[str, Any]],
                              results_df: pd.DataFrame) -> Dict[int, str]:
    """
    Determine what political archetype each cluster represents.

    Args:
        cluster_analysis: Cluster analysis results
        results_df: DataFrame with actual results

    Returns:
        Dictionary mapping cluster IDs to archetype descriptions
    """
    archetypes = {}

    for cluster_id, analysis in cluster_analysis.items():
        relative_bias = analysis['relative_bias']

        # Simple archetype detection based on bias patterns
        left_parties = ['Arbeiderpartiet', 'SV',
                        'Rødt', 'SP']  # Left-leaning parties
        right_parties = ['Høyre', 'FrP', 'KrF',
                         'Venstre']  # Right-leaning parties

        left_bias = np.mean([relative_bias.get(party, 0)
                            for party in left_parties if party in relative_bias])
        right_bias = np.mean([relative_bias.get(party, 0)
                             for party in right_parties if party in relative_bias])

        # Determine archetype
        if left_bias > 1 and right_bias < -1:
            archetype = "Left-Optimistic (overestimates left parties)"
        elif left_bias < -1 and right_bias > 1:
            archetype = "Right-Optimistic (overestimates right parties)"
        elif abs(left_bias) < 0.5 and abs(right_bias) < 0.5:
            archetype = "Balanced/Realistic"
        elif left_bias < 0 and right_bias < 0:
            archetype = "Generally Pessimistic"
        elif left_bias > 0 and right_bias > 0:
            archetype = "Generally Optimistic"
        else:
            archetype = "Mixed Pattern"

        archetypes[cluster_id] = archetype

    return archetypes
