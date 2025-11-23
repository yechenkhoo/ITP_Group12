"""
Run this using:
python -m MainScripts.comparison
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os 

def plot_overall_metrics_bar(overall_improvements_dict, model1_name, model2_name, ax):
    metrics = ['accuracy', 'precision', 'recall', 'f1_score']
    model1_scores = [overall_improvements_dict[m]['model1'] for m in metrics if m in overall_improvements_dict]
    model2_scores = [overall_improvements_dict[m]['model2'] for m in metrics if m in overall_improvements_dict]
    labels = [m.capitalize() for m in metrics if m in overall_improvements_dict]

    x = np.arange(len(labels))
    width = 0.35

    ax.bar(x - width/2, model1_scores, width, label=model1_name, color='skyblue')
    ax.bar(x + width/2, model2_scores, width, label=model2_name, color='salmon')

    ax.set_ylabel('Score')
    ax.set_title('Overall Macro-Averaged Metrics')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(True, alpha=0.3)

def compare_per_class_metrics(csv1_path, csv2_path, model1_name="Model 1", model2_name="Model 2", save_path=None):
    """
    Compare per-class metrics between two models
    Args:
        csv1_path: Path to first model's per-class metrics CSV
        csv2_path: Path to second model's per-class metrics CSV  
        model1_name: Name for first model (for display)
        model2_name: Name for second model (for display)
        save_path: Optional path to save comparison results
    """
    os.makedirs(save_path, exist_ok=True)

    # Load the data
    df1 = pd.read_csv(csv1_path)
    df2 = pd.read_csv(csv2_path)
    
    # Merge on class_id and class_name
    comparison_df = df1.merge(df2, on=['class_id', 'class_name'], suffixes=('_model1', '_model2'))
    
    # Calculate improvements (Model 2 - Model 1)
    metrics = ['accuracy', 'precision', 'recall', 'f1_score']
    for metric in metrics:
        comparison_df[f'{metric}_improvement'] = comparison_df[f'{metric}_model2'] - comparison_df[f'{metric}_model1']
        comparison_df[f'{metric}_improvement_pct'] = (comparison_df[f'{metric}_improvement'] / comparison_df[f'{metric}_model1']) * 100
    
    # Calculate overall improvements
    overall_improvements = {}
    for metric in metrics:
        model1_macro = comparison_df[f'{metric}_model1'].mean()
        model2_macro = comparison_df[f'{metric}_model2'].mean()
        improvement = model2_macro - model1_macro
        improvement_pct = (improvement / model1_macro) * 100 if model1_macro > 0 else 0
        overall_improvements[metric] = {
            'model1': model1_macro,
            'model2': model2_macro,
            'improvement': improvement,
            'improvement_pct': improvement_pct
        }

    # Print summary
    print("="*60)
    print(f"COMPARISON: {model1_name} vs {model2_name}")
    print("="*60)
    
    print("\nOVERALL PERFORMANCE COMPARISON:")
    print("-" * 40)
    for metric in metrics:
        data = overall_improvements[metric]
        print(f"{metric.upper()}:")
        print(f"  {model1_name}: {data['model1']:.4f}")
        print(f"  {model2_name}: {data['model2']:.4f}")
        print(f"  Improvement: {data['improvement']:+.4f} ({data['improvement_pct']:+.2f}%)")
        print()
    
    # Find best and worst improvements
    print("PER-CLASS IMPROVEMENTS:")
    print("-" * 40)
    
    for metric in metrics:
        print(f"\n{metric.upper()} Changes:")
        
        # Best improvements
        best_idx = comparison_df[f'{metric}_improvement'].idxmax()
        best_class = comparison_df.loc[best_idx, 'class_name']
        best_improvement = comparison_df.loc[best_idx, f'{metric}_improvement']
        best_improvement_pct = comparison_df.loc[best_idx, f'{metric}_improvement_pct']
        
        # Worst improvements (biggest decreases)
        worst_idx = comparison_df[f'{metric}_improvement'].idxmin()
        worst_class = comparison_df.loc[worst_idx, 'class_name']
        worst_improvement = comparison_df.loc[worst_idx, f'{metric}_improvement']
        worst_improvement_pct = comparison_df.loc[worst_idx, f'{metric}_improvement_pct']
        
        print(f"  Best improvement: {best_class} ({best_improvement:+.4f}, {best_improvement_pct:+.2f}%)")
        print(f"  Worst change: {worst_class} ({worst_improvement:+.4f}, {worst_improvement_pct:+.2f}%)")
    
    # Detailed per-class table
    print(f"\nDETAILED PER-CLASS COMPARISON:")
    print("-" * 40)
    
    display_cols = ['class_name']
    for metric in metrics:
        display_cols.extend([f'{metric}_model1', f'{metric}_model2', f'{metric}_improvement'])
    
    display_df = comparison_df[display_cols].round(4)
    print(display_df.to_string(index=False))
    
    # Create visualization
    create_comparison_plots(comparison_df, model1_name, model2_name, save_path)
    
    # Save detailed comparison if path provided
    if save_path:
        detailed_path = save_path + "/detailed.csv"
        comparison_df.to_csv(detailed_path, index=False)
        print(f"\n[INFO] Detailed comparison saved to: {detailed_path}")
        
        # Save summary
        summary_data = []
        for metric in metrics:
            summary_data.append({
                'metric': metric,
                f'{model1_name}_macro': overall_improvements[metric]['model1'],
                f'{model2_name}_macro': overall_improvements[metric]['model2'],
                'improvement': overall_improvements[metric]['improvement'],
                'improvement_pct': overall_improvements[metric]['improvement_pct']
            })
        
        summary_df = pd.DataFrame(summary_data)
        summary_path = save_path + "/summary.csv"
        summary_df.to_csv(summary_path, index=False)
        print(f"[INFO] Summary comparison saved to: {summary_path}")
    
    return comparison_df

def create_comparison_plots(comparison_df, model1_name, model2_name, save_path=None):
    """Create visualization plots for the comparison"""

    # Plot 1 (Separate Figure) - Precision and Recall Comparison by Class
    fig1, ax1 = plt.subplots(figsize=(18, 8))
    x = np.arange(len(comparison_df))
    width = 0.2
    ax1.bar(x - 1.5 * width, comparison_df['precision_model1'], width, label=f'{model1_name} Precision', color='#aec6cf')
    ax1.bar(x - 0.5 * width, comparison_df['recall_model1'], width, label=f'{model1_name} Recall', color='#6ca0b6')
    ax1.bar(x + 0.5 * width, comparison_df['precision_model2'], width, label=f'{model2_name} Precision', color='#f4cccc')
    ax1.bar(x + 1.5 * width, comparison_df['recall_model2'], width, label=f'{model2_name} Recall', color='#d46a6a')

    ax1.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=2)
    ax1.set_xlabel('Golf Poses')
    ax1.set_ylabel('Score')
    ax1.set_title('Precision and Recall Comparison by Class')
    ax1.set_xticks(x)
    ax1.set_xticklabels(comparison_df['class_name'], rotation=45)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    fig1.tight_layout()
    fig1.savefig(save_path + "/plot1_precision_recall_per_class_comparison.png", dpi=300)

    # Plot 2 (Separate Figure) - Improvement heatmap
    fig2, ax2 = plt.subplots(figsize=(12, 6))
    metrics = ['precision', 'recall', 'f1_score']
    improvement_data = comparison_df[['precision_improvement', 'recall_improvement', 'f1_score_improvement']].T
    im = ax2.imshow(improvement_data, cmap='RdYlGn', aspect='auto', vmin=-0.2, vmax=0.2)

    ax2.set_xticks(range(len(comparison_df)))
    ax2.set_xticklabels(comparison_df['class_name'], rotation=45)
    ax2.set_yticks(range(len(metrics)))
    ax2.set_yticklabels(['Precision', 'Recall', 'F1-Score'])
    ax2.set_title('Improvement Heatmap\n')

    for i in range(len(metrics)):
        for j in range(len(comparison_df)):
            ax2.text(j, i, f'{improvement_data.iloc[i, j]:.3f}', ha="center", va="center", color="black", fontsize=8)

    plt.colorbar(im, ax=ax2)
    fig2.tight_layout()
    fig2.savefig(save_path + "/plot2_improvement_heatmap.png", dpi=300)

    # Plot 3 (Separate Figure) - Overall improvement bar chart (macro-averaged)
    fig3, ax3 = plt.subplots(figsize=(8, 5))
    overall_improvements = []
    metric_labels = []
    for metric in metrics:
        model1_macro = comparison_df[f'{metric}_model1'].mean()
        model2_macro = comparison_df[f'{metric}_model2'].mean()
        improvement = model2_macro - model1_macro
        overall_improvements.append(improvement)
        metric_labels.append(metric.capitalize())

    ax3.bar(metric_labels, overall_improvements, color=['skyblue', 'salmon', 'lightgreen'])
    ax3.set_ylabel('Improvement')
    ax3.set_title('Overall Macro-Averaged Improvements')
    ax3.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax3.grid(True, alpha=0.3)
    fig3.tight_layout()
    fig3.savefig(save_path + "/plot3_macro_averaged_improvements.png", dpi=300)

    # Plot 4 (Separate Figure) - Overall metrics comparison
    fig4, ax4 = plt.subplots(figsize=(8, 5))
    plot_overall_metrics_bar(
        overall_improvements_dict={m: {
            'model1': comparison_df[f'{m}_model1'].mean(),
            'model2': comparison_df[f'{m}_model2'].mean()
        } for m in metrics},
        model1_name=model1_name,
        model2_name=model2_name,
        ax=ax4
    )
    fig4.tight_layout()
    fig4.savefig(save_path + "/plot4_macro_metric_comparison.png", dpi=300)


if __name__ == "__main__":
    # csv1 = "output/SENR_BASELINE/BASELINEeval_per_class_metrics.csv"
    # csv2 = "output/D4_tomtest/MLP_Basic/eval_per_class_metrics.csv"
    
    csv1 = "output/D4_tomtest/MLP_Basic/eval_per_class_metrics.csv"
    csv2 = "output/TomSplit/hyperparameter_tuning_kerastuner_bayesian_tomsplit/mlp/eval_per_class_metrics.csv"

    compare_per_class_metrics(
        csv1_path=csv1,
        csv2_path=csv2,
        # model1_name="Baseline Model",
        # model2_name="Model After Dataset Refinement",
        model1_name="Chosen Model",
        model2_name="Tuned Model",
        save_path="comparison_results2"
    )