"""
Visualization module for election prediction analysis.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import networkx as nx
from typing import Dict, Any, Optional
from matplotlib.patches import Ellipse
from matplotlib.colors import TwoSlopeNorm

from .config import FIGURE_SIZE, PLOT_STYLE, PARTIES


def set_plotting_style():
    """Set the default plotting style."""
    plt.style.use(PLOT_STYLE)
    sns.set_palette("husl")


def create_error_distribution_plots(error_df: pd.DataFrame) -> plt.Figure:
    """
    Create comprehensive error distribution plots.

    Args:
        error_df: DataFrame with total errors

    Returns:
        Matplotlib figure
    """
    set_plotting_style()

    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=FIGURE_SIZE)
    fig.suptitle('Total Error Distribution Analysis',
                 fontsize=16, fontweight='bold')

    # 1. Histogram with KDE
    axes[0, 0].hist(error_df['total_error'], bins=20,
                    alpha=0.7, color='skyblue', edgecolor='black')
    axes[0, 0].axvline(error_df['total_error'].mean(), color='red', linestyle='--',
                       label=f'Mean: {error_df["total_error"].mean():.2f}')
    axes[0, 0].axvline(error_df['total_error'].median(), color='green', linestyle='--',
                       label=f'Median: {error_df["total_error"].median():.2f}')
    axes[0, 0].set_xlabel('Total Error (percentage points)')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Distribution of Total Prediction Errors')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # 2. Box plot with individual points
    box_plot = axes[0, 1].boxplot(error_df['total_error'], patch_artist=True)
    box_plot['boxes'][0].set_facecolor('lightblue')

    # Add individual points
    y_points = np.random.normal(1, 0.04, len(error_df))  # Add some jitter
    axes[0, 1].scatter(y_points, error_df['total_error'],
                       alpha=0.6, color='red', s=30)

    axes[0, 1].set_ylabel('Total Error (percentage points)')
    axes[0, 1].set_title('Box Plot of Total Errors')
    axes[0, 1].set_xticklabels(['All Respondents'])
    axes[0, 1].grid(True, alpha=0.3)

    # 3. Ranking plot (sorted errors)
    sorted_errors = error_df.sort_values('total_error')
    axes[1, 0].plot(range(len(sorted_errors)), sorted_errors['total_error'],
                    marker='o', linewidth=2, markersize=6)
    axes[1, 0].set_xlabel('Rank (Best to Worst)')
    axes[1, 0].set_ylabel('Total Error (percentage points)')
    axes[1, 0].set_title('Performance Ranking')
    axes[1, 0].grid(True, alpha=0.3)

    # Highlight best and worst performers
    best_idx = 0
    worst_idx = len(sorted_errors) - 1
    axes[1, 0].scatter(best_idx, sorted_errors.iloc[best_idx]['total_error'],
                       color='gold', s=100, label='Best')
    axes[1, 0].scatter(worst_idx, sorted_errors.iloc[worst_idx]['total_error'],
                       color='red', s=100, label='Worst')
    axes[1, 0].legend()

    # 4. Summary statistics
    axes[1, 1].axis('off')

    # Calculate statistics
    stats = {
        'Count': len(error_df),
        'Mean Error': f"{error_df['total_error'].mean():.2f}",
        'Median Error': f"{error_df['total_error'].median():.2f}",
        'Std Dev': f"{error_df['total_error'].std():.2f}",
        'Min Error': f"{error_df['total_error'].min():.2f}",
        'Max Error': f"{error_df['total_error'].max():.2f}",
        'Range': f"{error_df['total_error'].max() - error_df['total_error'].min():.2f}"
    }

    # Best and worst performers
    best_performer = error_df.loc[error_df['total_error'].idxmin()]
    worst_performer = error_df.loc[error_df['total_error'].idxmax()]

    stats_text = "Summary Statistics:\n\n"
    for key, value in stats.items():
        stats_text += f"{key}: {value}\n"

    stats_text += f"\nBest Performer:\n{best_performer.name} ({best_performer['total_error']:.2f})\n"
    stats_text += f"\nWorst Performer:\n{worst_performer.name} ({worst_performer['total_error']:.2f})"

    axes[1, 1].text(0.1, 0.95, stats_text, transform=axes[1, 1].transAxes,
                    fontsize=12, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))

    plt.tight_layout()
    return fig


def create_party_analysis_plots(party_errors_df: pd.DataFrame,
                                party_stats: Dict[str, Dict[str, float]]) -> Tuple[plt.Figure, Dict[str, Dict[str, float]]]:
    """
    Create comprehensive party-by-party analysis plots.

    Args:
        party_errors_df: DataFrame with party errors
        party_stats: Party statistics

    Returns:
        Tuple of (figure, party_stats)
    """
    set_plotting_style()

    # Get party columns (exclude respondent)
    party_columns = [
        col for col in party_errors_df.columns if col != 'respondent']

    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=FIGURE_SIZE)
    fig.suptitle('Party-by-Party Prediction Error Analysis',
                 fontsize=16, fontweight='bold')

    # 1. HEATMAP: Respondents vs Parties
    # Prepare data for heatmap (respondents as rows, parties as columns)
    heatmap_data = party_errors_df.set_index('respondent')[party_columns]

    # Create heatmap
    im = axes[0, 0].imshow(heatmap_data.values, cmap='Reds', aspect='auto')

    # Set labels
    axes[0, 0].set_xticks(range(len(party_columns)))
    axes[0, 0].set_xticklabels(party_columns, rotation=45, ha='right')
    axes[0, 0].set_yticks(range(len(heatmap_data)))
    axes[0, 0].set_yticklabels(heatmap_data.index, fontsize=8)
    axes[0, 0].set_title('Error Heatmap: Darker = Worse Predictions')
    axes[0, 0].set_xlabel('Parties')
    axes[0, 0].set_ylabel('Respondents')

    # Add colorbar
    cbar = plt.colorbar(im, ax=axes[0, 0])
    cbar.set_label('Absolute Error (percentage points)')

    # 2. VIOLIN PLOTS: Distribution of errors by party
    # Prepare data for violin plot
    violin_data = []
    violin_labels = []

    for party in party_columns:
        if party in party_errors_df.columns:
            errors = party_errors_df[party].dropna()
            violin_data.append(errors)
            violin_labels.append(f"{party}\n(avg: {errors.mean():.1f})")

    # Create violin plot
    if violin_data:
        parts = axes[0, 1].violinplot(violin_data, positions=range(len(violin_data)),
                                      showmeans=True, showmedians=True)

        # Color the violins
        colors = plt.cm.Set3(np.linspace(0, 1, len(violin_data)))
        for pc, color in zip(parts['bodies'], colors):
            pc.set_facecolor(color)
            pc.set_alpha(0.7)

    axes[0, 1].set_xticks(range(len(violin_labels)))
    axes[0, 1].set_xticklabels(
        violin_labels, rotation=45, ha='right', fontsize=9)
    axes[0, 1].set_ylabel('Absolute Error (percentage points)')
    axes[0, 1].set_title('Error Distribution by Party')
    axes[0, 1].grid(True, alpha=0.3)

    # 3. BAR CHART: Average error by party (sorted)
    # Prepare data for bar chart
    party_avg_errors = [(party, party_stats[party]['mean_error'])
                        for party in party_columns if party in party_stats]
    # Sort by error (hardest first)
    party_avg_errors.sort(key=lambda x: x[1], reverse=True)

    parties_sorted = [x[0] for x in party_avg_errors]
    errors_sorted = [x[1] for x in party_avg_errors]

    bars = axes[1, 0].bar(range(len(parties_sorted)), errors_sorted,
                          color=plt.cm.Reds(np.linspace(0.3, 0.8, len(parties_sorted))))

    axes[1, 0].set_xticks(range(len(parties_sorted)))
    axes[1, 0].set_xticklabels(parties_sorted, rotation=45, ha='right')
    axes[1, 0].set_ylabel('Average Absolute Error')
    axes[1, 0].set_title('Hardest Parties to Predict (Highest Avg Error)')
    axes[1, 0].grid(True, alpha=0.3, axis='y')

    # 4. SUMMARY TABLE: Party statistics
    axes[1, 1].axis('off')

    # Create summary table
    table_data = []
    headers = ['Party', 'Actual %', 'Avg Error', 'Std Dev', 'Worst Error']

    for party in parties_sorted:  # Use same order as bar chart
        stats = party_stats[party]
        table_data.append([
            party,
            f"{stats['actual_result']:.1f}%",
            f"{stats['mean_error']:.1f}",
            f"{stats['std_error']:.1f}",
            f"{stats['max_error']:.1f}"
        ])

    # Create table
    if table_data:
        table = axes[1, 1].table(cellText=table_data, colLabels=headers,
                                 cellLoc='center', loc='center',
                                 colWidths=[0.15, 0.12, 0.12, 0.12, 0.12])

        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.2, 2)

        # Color code the table rows by difficulty (matching bar chart colors)
        colors_normalized = plt.cm.Reds(
            np.linspace(0.3, 0.8, len(parties_sorted)))
        for i in range(len(table_data)):
            for j in range(len(headers)):
                table[(i+1, j)].set_facecolor(colors_normalized[i])
                table[(i+1, j)].set_alpha(0.3)

        # Header styling
        for j in range(len(headers)):
            table[(0, j)].set_facecolor('#40466e')
            table[(0, j)].set_text_props(weight='bold', color='white')

    axes[1, 1].set_title(
        'Party Prediction Difficulty Rankings', pad=20, fontweight='bold')

    plt.tight_layout()
    return fig, party_stats


def create_bias_detection_plots(bias_df: pd.DataFrame,
                                party_bias_stats: Dict[str, Dict[str, Any]],
                                respondent_bias_stats: Dict[str, Dict[str, Any]],
                                predictions_df: pd.DataFrame,
                                results_df: pd.DataFrame) -> plt.Figure:
    """
    Create comprehensive bias detection visualizations.

    Args:
        bias_df: DataFrame with bias data
        party_bias_stats: Party bias statistics
        respondent_bias_stats: Respondent bias statistics
        predictions_df: Original predictions
        results_df: Actual results

    Returns:
        Matplotlib figure
    """
    set_plotting_style()

    # Get data
    actual_results = results_df.iloc[0] if len(results_df) == 1 else results_df

    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=FIGURE_SIZE)
    fig.suptitle('Prediction Bias Detection Analysis',
                 fontsize=16, fontweight='bold')

    # 1. SCATTER PLOT: Predicted vs Actual for each party
    colors = plt.cm.Set1(np.linspace(0, 1, len(PARTIES)))

    for i, party in enumerate(PARTIES):
        if party in actual_results:
            actual_val = actual_results[party]
            predicted_vals = predictions_df[party]

            axes[0, 0].scatter(predicted_vals, [actual_val] * len(predicted_vals),
                               alpha=0.6, s=60, color=colors[i], label=party,
                               edgecolors='black', linewidth=0.5)

    # Add perfect prediction line (diagonal)
    min_val = min(predictions_df[PARTIES].min().min(), actual_results.min())
    max_val = max(predictions_df[PARTIES].max().max(), actual_results.max())
    axes[0, 0].plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.8,
                    linewidth=2, label='Perfect Prediction')

    axes[0, 0].set_xlabel('Predicted %')
    axes[0, 0].set_ylabel('Actual %')
    axes[0, 0].set_title('Predicted vs Actual Results by Party')
    axes[0, 0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    axes[0, 0].grid(True, alpha=0.3)

    # 2. BAR CHART: Average bias by party
    parties = list(party_bias_stats.keys())
    mean_biases = [party_bias_stats[party]['mean_bias'] for party in parties]

    # Color bars by bias direction
    colors = ['red' if bias > 0 else 'blue' if bias <
              0 else 'gray' for bias in mean_biases]

    bars = axes[0, 1].bar(range(len(parties)), mean_biases, color=colors, alpha=0.7,
                          edgecolor='black', linewidth=0.5)

    axes[0, 1].axhline(y=0, color='black', linestyle='-', linewidth=1)
    axes[0, 1].set_xticks(range(len(parties)))
    axes[0, 1].set_xticklabels(parties, rotation=45, ha='right')
    axes[0, 1].set_ylabel('Average Bias (Predicted - Actual)')
    axes[0, 1].set_title(
        'Systematic Bias by Party\n(Red = Overestimated, Blue = Underestimated)')
    axes[0, 1].grid(True, alpha=0.3, axis='y')

    # 3. INDIVIDUAL BIAS PATTERNS: Heatmap of respondent biases
    # Create bias heatmap data
    bias_heatmap_data = bias_df.set_index('respondent')[PARTIES]

    # Create custom colormap centered at 0
    vmin, vmax = bias_heatmap_data.min().min(), bias_heatmap_data.max().max()
    norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)

    im = axes[1, 0].imshow(bias_heatmap_data.values,
                           cmap='RdBu_r', norm=norm, aspect='auto')

    # Set labels
    axes[1, 0].set_xticks(range(len(PARTIES)))
    axes[1, 0].set_xticklabels(PARTIES, rotation=45, ha='right')
    axes[1, 0].set_yticks(range(len(bias_heatmap_data)))
    axes[1, 0].set_yticklabels(bias_heatmap_data.index, fontsize=8)
    axes[1, 0].set_title(
        'Individual Bias Patterns\n(Red = Overestimated, Blue = Underestimated)')
    axes[1, 0].set_xlabel('Parties')
    axes[1, 0].set_ylabel('Respondents')

    # Add colorbar
    cbar = plt.colorbar(im, ax=axes[1, 0])
    cbar.set_label('Bias (Predicted - Actual)')

    # 4. RESPONDENT BIAS SUMMARY
    axes[1, 1].axis('off')

    # Calculate optimism ranking
    respondent_optimism = [(name, stats['mean_bias'], stats['optimism_ratio'])
                           for name, stats in respondent_bias_stats.items()]
    # Most optimistic first
    respondent_optimism.sort(key=lambda x: x[1], reverse=True)

    # Create summary text
    summary_text = "INDIVIDUAL BIAS PATTERNS:\n\n"
    summary_text += "MOST OPTIMISTIC (Overestimators):\n"

    for i, (name, mean_bias, opt_ratio) in enumerate(respondent_optimism[:3]):
        summary_text += f"{i+1}. {name}: {mean_bias:+.1f} avg bias ({opt_ratio:.0%} overestimates)\n"

    summary_text += "MOST PESSIMISTIC (Underestimators):\n"
    for i, (name, mean_bias, opt_ratio) in enumerate(respondent_optimism[-3:]):
        summary_text += f"{i+1}. {name}: {mean_bias:+.1f} avg bias ({(1-opt_ratio):.0%} underestimates)\n"

    # Party bias summary
    party_overest = [(party, stats['mean_bias']) for party, stats in party_bias_stats.items()
                     if stats['mean_bias'] > 0.5]
    party_underest = [(party, stats['mean_bias']) for party, stats in party_bias_stats.items()
                      if stats['mean_bias'] < -0.5]

    summary_text += "MOST OVERESTIMATED PARTIES:\n"
    party_overest.sort(key=lambda x: x[1], reverse=True)
    for party, bias in party_overest[:3]:
        pct_over = party_bias_stats[party]['overestimated_count'] / \
            len(predictions_df) * 100
        summary_text += f"• {party}: {bias:+.1f} avg ({pct_over:.0f}% overestimated it)\n"

    summary_text += "MOST UNDERESTIMATED PARTIES:\n"
    party_underest.sort(key=lambda x: x[1])
    for party, bias in party_underest[:3]:
        pct_under = party_bias_stats[party]['underestimated_count'] / \
            len(predictions_df) * 100
        summary_text += f"• {party}: {bias:+.1f} avg ({pct_under:.0f}% underestimated it)\n"

    # Add the text
    axes[1, 1].text(-0.05, 0.95, summary_text, transform=axes[1, 1].transAxes,
                    fontsize=10, verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))

    plt.tight_layout()
    return fig


def create_political_clustering_plots(predictions_df: pd.DataFrame,
                                      n_clusters: int = 3) -> Tuple[plt.Figure, Dict[str, Any], nx.Graph, np.ndarray, np.ndarray]:
    """
    Create comprehensive political clustering visualizations.

    Args:
        predictions_df: DataFrame with predictions
        n_clusters: Number of clusters

    Returns:
        Tuple of (figure, clustering_results, network, similarity_matrix, correlation_matrix)
    """
    from .clustering import (prepare_clustering_data, perform_clustering,
                             calculate_similarity_network)

    set_plotting_style()

    # Prepare data
    X, X_scaled, respondent_names, party_columns = prepare_clustering_data(
        predictions_df)

    # Perform clustering
    clustering_results = perform_clustering(
        X_scaled, respondent_names, n_clusters)

    # Calculate similarity network
    G, similarity_matrix, correlation_matrix = calculate_similarity_network(
        X, respondent_names)

    # Set up plotting
    colors = plt.cm.Set1(np.linspace(0, 1, n_clusters))

    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=FIGURE_SIZE)
    fig.suptitle('Political Clustering Analysis: Who Thinks Alike?',
                 fontsize=16, fontweight='bold')

    # 1. PCA CLUSTERING (Top Left)
    X_pca = clustering_results['X_pca']
    kmeans_labels = clustering_results['kmeans_labels']
    pca = clustering_results['pca']

    # Plot PCA with clusters
    for i in range(n_clusters):
        cluster_mask = kmeans_labels == i
        axes[0, 0].scatter(X_pca[cluster_mask, 0], X_pca[cluster_mask, 1],
                           c=[colors[i]], s=100, alpha=0.7,
                           label=f'Cluster {i+1}', edgecolors='black', linewidth=0.5)

        # Add confidence ellipse for each cluster
        if np.sum(cluster_mask) > 2:  # Need at least 3 points for ellipse
            cluster_points = X_pca[cluster_mask]
            mean = cluster_points.mean(axis=0)
            cov = np.cov(cluster_points.T)

            # Calculate ellipse parameters
            eigenvals, eigenvecs = np.linalg.eigh(cov)
            angle = np.degrees(np.arctan2(eigenvecs[1, 0], eigenvecs[0, 0]))
            width, height = 2 * np.sqrt(eigenvals) * 2  # 2 sigma

            ellipse = Ellipse(mean, width, height, angle=angle,
                              facecolor=colors[i], alpha=0.2, edgecolor=colors[i])
            axes[0, 0].add_patch(ellipse)

    # Add names to points
    for i, name in enumerate(respondent_names):
        axes[0, 0].annotate(name, (X_pca[i, 0], X_pca[i, 1]),
                            xytext=(5, 5), textcoords='offset points',
                            fontsize=8, ha='left')

    axes[0, 0].set_xlabel(
        f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
    axes[0, 0].set_ylabel(
        f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
    axes[0, 0].set_title('PCA Clustering: Similar Prediction Patterns')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # 2. t-SNE CLUSTERING (Top Right)
    X_tsne = clustering_results['X_tsne']

    for i in range(n_clusters):
        cluster_mask = kmeans_labels == i
        axes[0, 1].scatter(X_tsne[cluster_mask, 0], X_tsne[cluster_mask, 1],
                           c=[colors[i]], s=100, alpha=0.7,
                           label=f'Cluster {i+1}', edgecolors='black', linewidth=0.5)

    # Add names to points
    for i, name in enumerate(respondent_names):
        axes[0, 1].annotate(name, (X_tsne[i, 0], X_tsne[i, 1]),
                            xytext=(5, 5), textcoords='offset points',
                            fontsize=8, ha='left')

    axes[0, 1].set_xlabel('t-SNE Dimension 1')
    axes[0, 1].set_ylabel('t-SNE Dimension 2')
    axes[0, 1].set_title('t-SNE Clustering: Non-linear Similarity Detection')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # 3. HIERARCHICAL CLUSTERING DENDROGRAM (Bottom Left)
    # Calculate linkage matrix
    from scipy.cluster.hierarchy import linkage
    linkage_matrix = linkage(X_scaled, method='ward')

    # Create dendrogram
    from scipy.cluster.hierarchy import dendrogram
    dendro = dendrogram(linkage_matrix, labels=respondent_names,
                        ax=axes[1, 0], orientation='top',
                        color_threshold=0.7*max(linkage_matrix[:, 2]))

    axes[1, 0].set_title(
        'Hierarchical Clustering Dendrogram\n(Lower = More Similar)')
    axes[1, 0].set_ylabel('Distance')
    axes[1, 0].tick_params(axis='x', rotation=45)

    # 4. SIMILARITY NETWORK (Bottom Right)
    if len(G.edges()) > 0:  # Only if we have connections
        # Calculate layout
        pos = nx.spring_layout(G, k=1, iterations=50)

        # Draw network
        nx.draw_networkx_nodes(G, pos, ax=axes[1, 1],
                               node_color=[colors[kmeans_labels[respondent_names.index(node)]]
                                           for node in G.nodes()],
                               node_size=300, alpha=0.8)

        # Draw edges with thickness based on similarity
        edges = G.edges(data=True)
        weights = [edge[2]['weight'] for edge in edges]
        nx.draw_networkx_edges(G, pos, ax=axes[1, 1],
                               width=[w*3 for w in weights],
                               alpha=0.6, edge_color='gray')

        # Draw labels
        nx.draw_networkx_labels(G, pos, ax=axes[1, 1], font_size=8)

        axes[1, 1].set_title(
            'Similarity Network\n(Thick lines = Very similar predictions)')
    else:
        axes[1, 1].text(0.5, 0.5, 'No strong similarities detected\n(Everyone predicted quite differently)',
                        ha='center', va='center', transform=axes[1, 1].transAxes,
                        fontsize=12, bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        axes[1, 1].set_title('Similarity Network')

    axes[1, 1].axis('off')

    plt.tight_layout()

    return fig, clustering_results, G, similarity_matrix, correlation_matrix
