import json
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

def load_comparison_results(file_path):
    """Load pixel vs perceptual comparison results"""
    with open(file_path, 'r') as f:
        return json.load(f)

def plot_overall_comparison(data, output_path):
    """Compare pixel matching vs SSIM overall"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle('Pixel Matching vs Perceptual Similarity (SSIM)', fontsize=18, fontweight='bold')
    
    # Bar chart comparison
    metrics = ['Pixel Match', 'SSIM', 'Histogram\nSimilarity', 'MSE\nSimilarity']
    scores = [
        data['overall_stats']['pixel_avg'],
        data['overall_stats']['ssim_avg'],
        np.mean([d['histogram_similarity'] for d in data['combined_data']]),
        np.mean([d['mse_similarity'] for d in data['combined_data']])
    ]
    
    colors = ['#3498db', '#2ecc71', '#f39c12', '#e74c3c']
    bars = ax1.bar(metrics, scores, color=colors, alpha=0.8, edgecolor='black', linewidth=2)
    
    ax1.set_ylabel('Average Score (%)', fontsize=13, fontweight='bold')
    ax1.set_title('Average Scores by Metric', fontsize=14, fontweight='bold')
    ax1.set_ylim(0, 100)
    ax1.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bar, score in zip(bars, scores):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{score:.1f}%',
                ha='center', va='bottom', fontweight='bold', fontsize=11)
    
    # Scatter plot: Pixel vs SSIM
    pixel_scores = [d['pixel_match'] for d in data['combined_data']]
    ssim_scores = [d['ssim'] for d in data['combined_data']]
    
    ax2.scatter(pixel_scores, ssim_scores, alpha=0.5, s=80, c='#3498db', edgecolors='black', linewidth=0.5)
    
    # Add diagonal line (perfect correlation)
    ax2.plot([0, 100], [0, 100], 'r--', linewidth=2, alpha=0.5, label='Perfect Correlation')
    
    # Add reference lines
    ax2.axhline(y=90, color='green', linestyle='--', alpha=0.3, linewidth=1.5)
    ax2.axvline(x=90, color='green', linestyle='--', alpha=0.3, linewidth=1.5)
    
    ax2.set_xlabel('Pixel Match (%)', fontsize=13, fontweight='bold')
    ax2.set_ylabel('SSIM (%)', fontsize=13, fontweight='bold')
    ax2.set_title('Pixel Match vs SSIM Correlation', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(-5, 105)
    ax2.set_ylim(-5, 105)
    ax2.legend()
    
    # Add correlation coefficient
    correlation = np.corrcoef(pixel_scores, ssim_scores)[0, 1]
    ax2.text(0.05, 0.95, f'Correlation: {correlation:.3f}',
            transform=ax2.transAxes, fontsize=12, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()

def plot_performance_classification(data, output_path):
    """Pie chart and bar chart of performance classifications"""
    classifications = data['classifications']
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle('Performance Classification', fontsize=18, fontweight='bold')
    
    # Pie chart
    labels = []
    sizes = []
    colors_pie = ['#2ecc71', '#3498db', '#f39c12', '#e67e22', '#e74c3c']
    
    classification_order = [
        'Excellent (pixel-perfect)',
        'Good (near pixel-perfect)',
        'Perceptually correct (not pixel-perfect)',
        'Acceptable (some differences)',
        'Poor (failed edit)'
    ]
    
    for classification in classification_order:
        count = classifications.get(classification, 0)
        if count > 0:
            labels.append(classification.replace(' (', '\n('))
            sizes.append(count)
    
    colors_used = colors_pie[:len(sizes)]
    
    wedges, texts, autotexts = ax1.pie(sizes, labels=labels, colors=colors_used,
                                        autopct='%1.1f%%', startangle=90,
                                        textprops={'fontsize': 10})
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
        autotext.set_fontsize(11)
    
    ax1.set_title('Distribution by Quality', fontsize=14, fontweight='bold')
    
    # Bar chart with details
    y_pos = np.arange(len(labels))
    bars = ax2.barh(y_pos, sizes, color=colors_used, alpha=0.85, edgecolor='black', linewidth=2)
    
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(labels, fontsize=10)
    ax2.set_xlabel('Number of Tasks', fontsize=13, fontweight='bold')
    ax2.set_title('Task Count by Quality Level', fontsize=14, fontweight='bold')
    ax2.grid(axis='x', alpha=0.3)
    
    # Add count labels
    for i, (bar, count) in enumerate(zip(bars, sizes)):
        width = bar.get_width()
        ax2.text(width + 0.5, bar.get_y() + bar.get_height()/2.,
                f'{count}',
                ha='left', va='center', fontweight='bold', fontsize=11)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()

def plot_category_comparison(data, output_path):
    """Compare pixel vs SSIM by category"""
    categories = data['category_comparison']
    
    # Sort by improvement
    categories = sorted(categories, key=lambda x: x['improvement'], reverse=True)
    
    fig, ax = plt.subplots(figsize=(16, 10))
    
    cat_names = [c['category'].replace('_', ' ').title() for c in categories]
    pixel_scores = [c['avg_pixel'] for c in categories]
    ssim_scores = [c['avg_ssim'] for c in categories]
    improvements = [c['improvement'] for c in categories]
    
    x = np.arange(len(cat_names))
    width = 0.35
    
    # Create bars
    bars1 = ax.bar(x - width/2, pixel_scores, width, label='Pixel Match',
                   color='#3498db', alpha=0.8, edgecolor='black', linewidth=1.5)
    bars2 = ax.bar(x + width/2, ssim_scores, width, label='SSIM',
                   color='#2ecc71', alpha=0.8, edgecolor='black', linewidth=1.5)
    
    ax.set_xlabel('Task Category', fontsize=14, fontweight='bold')
    ax.set_ylabel('Average Score (%)', fontsize=14, fontweight='bold')
    ax.set_title('Pixel Match vs SSIM by Task Category', fontsize=18, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(cat_names, rotation=45, ha='right', fontsize=11)
    ax.legend(fontsize=12)
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, 105)
    
    # Add improvement annotations
    for i, (p, s, imp) in enumerate(zip(pixel_scores, ssim_scores, improvements)):
        if imp > 5:  # Only show significant improvements
            ax.annotate(f'+{imp:.0f}%',
                       xy=(i + width/2, s),
                       xytext=(0, 5),
                       textcoords='offset points',
                       ha='center',
                       fontsize=9,
                       fontweight='bold',
                       color='green')
    
    # Add text box with explanation
    explanation = (
        "Green bars show SSIM (perceptual similarity)\n"
        "Blue bars show Pixel Match (exact pixel comparison)\n"
        "Higher SSIM with lower Pixel Match indicates\n"
        "perceptually correct but not pixel-perfect results"
    )
    ax.text(0.02, 0.98, explanation, transform=ax.transAxes,
           fontsize=10, verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()

def plot_improvement_analysis(data, output_path):
    """Analyze where SSIM improves over pixel matching"""
    categories = data['category_comparison']
    
    # Sort by improvement
    categories = sorted(categories, key=lambda x: x['improvement'], reverse=True)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle('SSIM Improvement over Pixel Matching', fontsize=18, fontweight='bold')
    
    # Improvement by category
    cat_names = [c['category'].replace('_', ' ').title() for c in categories[:10]]
    improvements = [c['improvement'] for c in categories[:10]]
    counts = [c['count'] for c in categories[:10]]
    
    colors = ['#2ecc71' if imp > 0 else '#e74c3c' for imp in improvements]
    bars = ax1.barh(range(len(cat_names)), improvements, color=colors, alpha=0.8,
                   edgecolor='black', linewidth=2)
    
    ax1.set_yticks(range(len(cat_names)))
    ax1.set_yticklabels(cat_names, fontsize=11)
    ax1.set_xlabel('SSIM Improvement over Pixel Match (%)', fontsize=12, fontweight='bold')
    ax1.set_title('Top 10 Categories by Improvement', fontsize=14, fontweight='bold')
    ax1.axvline(x=0, color='black', linestyle='-', linewidth=1)
    ax1.grid(axis='x', alpha=0.3)
    
    # Add count and value labels
    for i, (bar, imp, count) in enumerate(zip(bars, improvements, counts)):
        width = bar.get_width()
        ax1.text(width + (1 if imp > 0 else -1), bar.get_y() + bar.get_height()/2.,
                f'{imp:+.1f}% (n={count})',
                ha='left' if imp > 0 else 'right', va='center',
                fontweight='bold', fontsize=10)
    
    # Distribution of improvements
    all_improvements = [c['improvement'] for c in data['category_comparison']]
    
    ax2.hist(all_improvements, bins=20, color='#3498db', alpha=0.7, edgecolor='black', linewidth=1.5)
    ax2.axvline(x=0, color='red', linestyle='--', linewidth=2, label='No improvement')
    ax2.axvline(x=np.mean(all_improvements), color='green', linestyle='--', linewidth=2,
               label=f'Mean: {np.mean(all_improvements):.1f}%')
    
    ax2.set_xlabel('SSIM Improvement (%)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Number of Categories', fontsize=12, fontweight='bold')
    ax2.set_title('Distribution of Improvements', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()

def plot_detailed_scatter(data, output_path):
    """Detailed scatter with classification colors"""
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Color by classification
    classification_colors = {
        'Excellent (pixel-perfect)': '#2ecc71',
        'Good (near pixel-perfect)': '#3498db',
        'Perceptually correct (not pixel-perfect)': '#f39c12',
        'Acceptable (some differences)': '#e67e22',
        'Poor (failed edit)': '#e74c3c'
    }
    
    # Classify each task
    for item in data['combined_data']:
        pixel = item['pixel_match']
        ssim = item['ssim']
        
        # Determine classification
        if ssim >= 90:
            if pixel >= 80:
                classification = 'Excellent (pixel-perfect)'
            elif pixel >= 50:
                classification = 'Good (near pixel-perfect)'
            else:
                classification = 'Perceptually correct (not pixel-perfect)'
        elif ssim >= 70:
            classification = 'Acceptable (some differences)'
        else:
            classification = 'Poor (failed edit)'
        
        color = classification_colors[classification]
        ax.scatter(pixel, ssim, c=color, s=100, alpha=0.6,
                  edgecolors='black', linewidth=0.5)
    
    # Add reference zones
    ax.axhspan(90, 100, alpha=0.1, color='green', label='Excellent SSIM (≥90%)')
    ax.axhspan(70, 90, alpha=0.1, color='yellow', label='Acceptable SSIM (70-90%)')
    ax.axhspan(0, 70, alpha=0.1, color='red', label='Poor SSIM (<70%)')
    
    ax.plot([0, 100], [0, 100], 'k--', linewidth=2, alpha=0.3, label='Perfect Correlation')
    
    ax.set_xlabel('Pixel Match (%)', fontsize=14, fontweight='bold')
    ax.set_ylabel('SSIM (%)', fontsize=14, fontweight='bold')
    ax.set_title('Detailed Task Distribution: Pixel Match vs SSIM', fontsize=18, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-5, 105)
    ax.set_ylim(-5, 105)
    
    # Create custom legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=color, label=label, edgecolor='black')
                      for label, color in classification_colors.items()]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=10)
    
    # Add key insight box
    insight = (
        "Key Insight:\n"
        "Tasks in orange zone (high SSIM, low pixel match)\n"
        "are perceptually correct but not pixel-perfect.\n"
        "This is often acceptable for many edit tasks."
    )
    ax.text(0.05, 0.95, insight, transform=ax.transAxes,
           fontsize=11, verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()

# Parse arguments
parser = argparse.ArgumentParser(description='Visualize pixel vs perceptual comparison')
parser.add_argument('--comparison-file', type=str, required=True,
                    help='Path to pixel_vs_perceptual_comparison.json')
parser.add_argument('--output-dir', type=str, default='graphs',
                    help='Directory to save graphs')

args = parser.parse_args()

# Create output directory
output_dir = Path(args.output_dir)
output_dir.mkdir(exist_ok=True)

# Load data
data = load_comparison_results(args.comparison_file)

print(f"Loaded comparison data for {len(data['combined_data'])} tasks")

# Generate plots
print("\nGenerating pixel vs perceptual visualizations...")
plot_overall_comparison(data, output_dir / "pixel_vs_ssim_comparison.png")
plot_performance_classification(data, output_dir / "performance_classification.png")
plot_category_comparison(data, output_dir / "category_comparison.png")
plot_improvement_analysis(data, output_dir / "ssim_improvement.png")
plot_detailed_scatter(data, output_dir / "detailed_scatter.png")

print(f"\n✓ All graphs saved to {output_dir}/")
print("\nKey Findings:")
print(f"  - Overall SSIM improvement: {data['overall_stats']['improvement']:.2f}%")
print(f"  - Pixel match average: {data['overall_stats']['pixel_avg']:.2f}%")
print(f"  - SSIM average: {data['overall_stats']['ssim_avg']:.2f}%")
