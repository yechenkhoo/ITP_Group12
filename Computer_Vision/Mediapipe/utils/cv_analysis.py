# To use:
# python utils/cv_analysis.py "output/{folder_name}"

import pandas as pd
import numpy as np
import glob
import os
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import inch

def main():
    import sys
    
    if len(sys.argv) > 1:
        folder_path = sys.argv[1]
    else:
        folder_path = "output/D3a_10"
    
    if not os.path.exists(folder_path):
        print(f"Error: Folder {folder_path} does not exist")
        print(f"Usage: python {__file__} <folder_path>")
        return
    
    print(f"Analyzing CV results in: {folder_path}")
    analyze_cv_results(folder_path)


def analyze_cv_results(folder_path):
    """
    Analyze all cross-validation results in a folder
    
    Args:
        folder_path (str): Path to folder containing *_cv_results.csv files
    """
    
    # Find all CV results files
    cv_files = glob.glob(os.path.join(folder_path, "*_cv_results.csv"))
    
    if not cv_files:
        print(f"No CV results files found in {folder_path}")
        return None
    
    print(f"Found {len(cv_files)} CV results files")
    print("=" * 80)
    
    # Store results for summary table
    summary_data = []
    all_model_details = {}  # Store detailed info for reports
    
    for cv_file in sorted(cv_files):
        # Extract model name from filename
        filename = os.path.basename(cv_file)
        model_name = filename.replace("_cv_results.csv", "")
        
        try:
            # Read CV results
            df = pd.read_csv(cv_file)
            
            if 'accuracy' not in df.columns:
                print(f"Warning: No 'accuracy' column found in {filename}")
                continue

            # Calculate statistics
            accuracies = df['accuracy'].values
            mean_acc = np.mean(accuracies)
            std_acc = np.std(accuracies)
            min_acc = np.min(accuracies)
            max_acc = np.max(accuracies)
            range_acc = max_acc - min_acc
            
            # Calculate F1 statistics if available
            f1_stats = {}
            if 'f1_score' in df.columns:
                f1_scores = df['f1_score'].values
                f1_stats = {
                    'mean': np.mean(f1_scores),
                    'std': np.std(f1_scores),
                    'min': np.min(f1_scores),
                    'max': np.max(f1_scores),
                    'range': np.max(f1_scores) - np.min(f1_scores)
                }
            
            recall_stats = {}
            if 'recall' in df.columns:
                recall_scores = df['recall'].values
                recall_stats = {
                    'mean': np.mean(recall_scores),
                    'std': np.std(recall_scores),
                    'min': np.min(recall_scores),
                    'max': np.max(recall_scores),
                    'range': np.max(recall_scores) - np.min(recall_scores)
                }
            
            # Store metrics for objective notes generation later
            temp_metrics = {
                'mean': mean_acc,
                'std': std_acc,
                'range': range_acc,
                'min': min_acc,
                'max': max_acc,
                'f1_stats': f1_stats,
                'recall_stats': recall_stats
            }
            
            # Store for summary
            summary_data.append({
                'Model': model_name,
                'Mean_Acc': mean_acc,
                'Std_Acc': std_acc,
                'Min_Acc': min_acc,
                'Max_Acc': max_acc,
                'Range': range_acc,
                'Mean_F1': f1_stats.get('mean', 0) if f1_stats else 0,
                'Std_F1': f1_stats.get('std', 0) if f1_stats else 0,
                'Mean_Recall': recall_stats.get('mean', 0) if recall_stats else 0,
                'Std_Recall': recall_stats.get('std', 0) if recall_stats else 0,
                'temp_metrics': temp_metrics
            })
            
            # Store detailed results for reports
            all_model_details[model_name] = {
                'mean': mean_acc,
                'std': std_acc,
                'min': min_acc,
                'max': max_acc,
                'range': range_acc,
                'folds': [f'{acc:.3f}' for acc in accuracies],
                'f1_stats': f1_stats,
                'recall_stats': recall_stats,
                'temp_metrics': temp_metrics
            }
            
            # Print concise results for this model
            print(f"\n{model_name}: {mean_acc:.3f}±{std_acc:.3f}")
            
        except Exception as e:
            print(f"Error processing {filename}: {e}")
    
    if summary_data:
        print("\n" + "=" * 80)
        print("CONCISE SUMMARY")
        print("=" * 80)
        
        # Generate objective notes based on relative rankings
        if summary_data:
            # Extract metrics for ranking
            means = [item['Mean_Acc'] for item in summary_data]
            stds = [item['Std_Acc'] for item in summary_data]
            ranges = [item['Range'] for item in summary_data]
            
            # Create sorted indices for finding top 2 and bottom 2
            mean_sorted_indices = sorted(range(len(means)), key=lambda i: means[i], reverse=True)
            std_sorted_indices = sorted(range(len(stds)), key=lambda i: stds[i], reverse=True)
            range_sorted_indices = sorted(range(len(ranges)), key=lambda i: ranges[i], reverse=True)
            
            # Generate objective notes for each model
            for i, item in enumerate(summary_data):
                notes = []
                
                # Check mean accuracy rankings
                if i == mean_sorted_indices[0]:
                    notes.append("highest mean")
                elif len(mean_sorted_indices) > 1 and i == mean_sorted_indices[1]:
                    notes.append("2nd highest mean")
                elif i == mean_sorted_indices[-1]:
                    notes.append("lowest mean")
                elif len(mean_sorted_indices) > 1 and i == mean_sorted_indices[-2]:
                    notes.append("2nd lowest mean")
                
                # Check std dev rankings (highest std = most variable, lowest std = most stable)
                if i == std_sorted_indices[0]:
                    notes.append("highest std")
                elif len(std_sorted_indices) > 1 and i == std_sorted_indices[1]:
                    notes.append("2nd highest std")
                elif i == std_sorted_indices[-1]:
                    notes.append("lowest std")
                elif len(std_sorted_indices) > 1 and i == std_sorted_indices[-2]:
                    notes.append("2nd lowest std")
                
                # Check range rankings
                if i == range_sorted_indices[0]:
                    notes.append("biggest range")
                elif len(range_sorted_indices) > 1 and i == range_sorted_indices[1]:
                    notes.append("2nd biggest range")
                elif i == range_sorted_indices[-1]:
                    notes.append("smallest range")
                elif len(range_sorted_indices) > 1 and i == range_sorted_indices[-2]:
                    notes.append("2nd smallest range")
                
                # Update with objective notes (blank if no extremes)
                item['Notes'] = ', '.join(notes) if notes else ""
                # Remove temporary metrics before DataFrame creation
                item.pop('temp_metrics', None)
                # Also update detailed results
                model_name = item['Model']
                if model_name in all_model_details:
                    all_model_details[model_name]['notes'] = ', '.join(notes) if notes else ""
                    all_model_details[model_name].pop('temp_metrics', None)
        
        # Create summary DataFrame
        summary_df = pd.DataFrame(summary_data)
        summary_df = summary_df.sort_values('Mean_Acc', ascending=False)
        
        # Print compact table
        print(f"{'Rank':<4} {'Model':<15} {'Accuracy±Std':<15} {'Key Notes'}")
        print("-" * 65)
        
        for i, (_, row) in enumerate(summary_df.iterrows(), 1):
            mean_std = f"{row['Mean_Acc']:.3f}±{row['Std_Acc']:.3f}"
            # Use the full notes (single line, objective)
            key_note = row['Notes']
            print(f"{i:<4} {row['Model']:<15} {mean_std:<12} {key_note}")
        
        print(f"\n✓ Best: {summary_df.iloc[0]['Model']} | Most Stable: {summary_df.loc[summary_df['Std_Acc'].idxmin(), 'Model']}")
        print(f"Full analysis with heatmap available in generated reports")
        
        # Save summary to CSV
        output_path = os.path.join(folder_path, "cv_summary_analysis.csv")
        summary_df.to_csv(output_path, index=False)
        print(f"\nSummary saved to: {output_path}")
        
        # Generate comprehensive reports
        print("\nGenerating detailed reports...")
        
        # Create heatmap visualization first (needed for PDF)
        heatmap_path = create_heatmap_table(summary_df, folder_path)
        print(f"Heatmap saved to: {heatmap_path}")
        
        # Generate text report
        text_report_path = generate_text_report(summary_df, folder_path, all_model_details)
        print(f"Text report saved to: {text_report_path}")
        
        # Generate PDF report (with heatmap included)
        pdf_report_path = generate_pdf_report(summary_df, folder_path, all_model_details, heatmap_path)
        if pdf_report_path:
            print(f"PDF report saved to: {pdf_report_path}")
        
        # Final summary
        print("\n" + "=" * 80)
        print("REPORTS GENERATED")
        print("=" * 80)
        
        print(f"Text Report: {text_report_path}")
        if pdf_report_path:
            print(f"PDF Report:  {pdf_report_path}")
        print(f"CSV Summary: {output_path}")
        print(f"Heatmap:     {heatmap_path}")
        print(f"\n✓ All reports include comprehensive analysis with visual summaries")
        
        return summary_df
    
    return None

def generate_text_report(summary_df, folder_path, all_model_details):
    """Generate a comprehensive text report"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    report_path = os.path.join(folder_path, "cv_analysis_report.txt")
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("CROSS-VALIDATION ANALYSIS REPORT\n")
        f.write("=" * 80 + "\n")
        f.write(f"Generated: {timestamp}\n")
        f.write(f"Folder: {folder_path}\n")
        f.write(f"Models Analyzed: {len(summary_df)}\n\n")
        
        # Detailed results for each model
        f.write("DETAILED MODEL RESULTS\n")
        f.write("=" * 50 + "\n\n")
        
        for model_name, details in all_model_details.items():
            f.write(f"{model_name}\n")
            f.write(f"   Mean Accuracy: {details['mean']:.3f} ± {details['std']:.3f}\n")
            
            # Add F1 statistics if available
            if details['f1_stats']:
                f1_stats = details['f1_stats']
                f.write(f"   Mean F1 Score: {f1_stats['mean']:.3f} ± {f1_stats['std']:.3f}\n")
                f.write(f"   F1 Range: {f1_stats['min']:.3f} - {f1_stats['max']:.3f} (span: {f1_stats['range']:.3f})\n")
            
            # Add Recall statistics if available
            if details['recall_stats']:
                recall_stats = details['recall_stats']
                f.write(f"   Mean Recall: {recall_stats['mean']:.3f} ± {recall_stats['std']:.3f}\n")
                f.write(f"   Recall Range: {recall_stats['min']:.3f} - {recall_stats['max']:.3f} (span: {recall_stats['range']:.3f})\n")
            
            f.write(f"   Accuracy Range: {details['min']:.3f} - {details['max']:.3f} (span: {details['range']:.3f})\n")
            f.write(f"   Individual Folds: {details['folds']}\n")
            f.write(f"   Notes: {details['notes']}\n\n")
        
        # Summary table
        f.write("\nSUMMARY TABLE\n")
        f.write("=" * 50 + "\n")
        
        # Check if F1 data is available
        has_f1_data = any(row['Mean_F1'] > 0 for _, row in summary_df.iterrows())
        has_recall_data = any(row['Mean_Recall'] > 0 for _, row in summary_df.iterrows())
        
        if has_f1_data or has_recall_data:
            # Build header dynamically based on available metrics
            header_parts = ['Model', 'Acc±Std']
            if has_f1_data:
                header_parts.append('F1±Std')
            if has_recall_data:
                header_parts.append('Recall±Std')
            header_parts.extend(['Acc Range'])
            if has_f1_data:
                header_parts.append('F1 Range')
            if has_recall_data:
                header_parts.append('Recall Range')
            header_parts.append('Notes')
            
            # Format header with appropriate spacing
            header_line = f"{'Model':<15} {'Accuracy±Std':<15} {'Accuracy Range':<13}"
            if has_f1_data:
                header_line += f" {'F1±Std':<12} {'F1 Range':<10}"
            if has_recall_data:
                header_line += f" {'Recall±Std':<12} {'Recall Range':<12}"
            header_line += " Notes"
            
            f.write(header_line + "\n")
            f.write("-" * (130 + (12 if has_recall_data else 0)) + "\n")
            
            for _, row in summary_df.iterrows():
                acc_std = f"{row['Mean_Acc']:.3f}±{row['Std_Acc']:.3f}"
                f1_std = f"{row['Mean_F1']:.3f}±{row['Std_F1']:.3f}" if has_f1_data and row['Mean_F1'] > 0 else "N/A"
                recall_std = f"{row['Mean_Recall']:.3f}±{row['Std_Recall']:.3f}" if has_recall_data and row['Mean_Recall'] > 0 else "N/A"
                acc_range = f"{row['Min_Acc']:.3f}-{row['Max_Acc']:.3f}"
                
                # Calculate F1 range if available
                f1_range = "N/A"
                recall_range = "N/A"
                model_details = all_model_details.get(row['Model'], {})
                if has_f1_data and model_details.get('f1_stats'):
                    f1_stats = model_details['f1_stats']
                    f1_range = f"{f1_stats['min']:.3f}-{f1_stats['max']:.3f}"
                if has_recall_data and model_details.get('recall_stats'):
                    recall_stats = model_details['recall_stats']
                    recall_range = f"{recall_stats['min']:.3f}-{recall_stats['max']:.3f}"
                
                # Build data line dynamically
                data_line = f"{row['Model']:<15} {acc_std:<15} {acc_range:<13}"
                if has_f1_data:
                    data_line += f" {f1_std:<12} {f1_range:<10}"
                if has_recall_data:
                    data_line += f" {recall_std:<12} {recall_range:<12}"
                data_line += f" {row['Notes']}"
                
                f.write(data_line + "\n")
        else:
            f.write(f"{'Model':<15} {'Accuracy±Std':<15} {'Min':<7} {'Max':<7} {'Range':<7} Notes\n")
            f.write("-" * 100 + "\n")
            
            for _, row in summary_df.iterrows():
                mean_std = f"{row['Mean_Acc']:.3f}±{row['Std_Acc']:.3f}"
                f.write(f"{row['Model']:<15} {mean_std:<12} {row['Min_Acc']:<7.3f} {row['Max_Acc']:<7.3f} {row['Range']:<7.3f} {row['Notes']}\n")
        
        # Key insights
        f.write("\n" + "=" * 50 + "\n")
        f.write("KEY INSIGHTS\n")
        f.write("=" * 50 + "\n")
        
        best_model = summary_df.iloc[0]
        most_stable = summary_df.loc[summary_df['Std_Acc'].idxmin()]
        least_stable = summary_df.loc[summary_df['Std_Acc'].idxmax()]
        
        f.write(f"Best Performance: {best_model['Model']} ({best_model['Mean_Acc']:.3f})\n")
        
        # Add F1 and Recall insights if available
        if has_f1_data or has_recall_data:
            if has_f1_data:
                best_f1_model = summary_df.loc[summary_df['Mean_F1'].idxmax()]
                most_stable_f1 = summary_df.loc[summary_df['Std_F1'].idxmin()]
                f.write(f"Best F1 Score: {best_f1_model['Model']} ({best_f1_model['Mean_F1']:.3f})\n")
                f.write(f"Most Stable (F1): {most_stable_f1['Model']} (std: {most_stable_f1['Std_F1']:.3f})\n")
            
            if has_recall_data:
                best_recall_model = summary_df.loc[summary_df['Mean_Recall'].idxmax()]
                most_stable_recall = summary_df.loc[summary_df['Std_Recall'].idxmin()]
                f.write(f"Best Recall: {best_recall_model['Model']} ({best_recall_model['Mean_Recall']:.3f})\n")
                f.write(f"Most Stable (Recall): {most_stable_recall['Model']} (std: {most_stable_recall['Std_Recall']:.3f})\n")
            
            f.write(f"Most Stable (Accuracy): {most_stable['Model']} (std: {most_stable['Std_Acc']:.3f})\n")
            f.write(f"Least Stable: {least_stable['Model']} (std: {least_stable['Std_Acc']:.3f})\n\n")
        else:
            f.write(f"Most Stable: {most_stable['Model']} (std: {most_stable['Std_Acc']:.3f})\n")
            f.write(f"Least Stable: {least_stable['Model']} (std: {least_stable['Std_Acc']:.3f})\n\n")
        
        # Recommendations
        f.write("RECOMMENDATIONS\n")
        f.write("-" * 20 + "\n")
        reliable_models = summary_df[(summary_df['Std_Acc'] < 0.03) & (summary_df['Mean_Acc'] > 0.8)]
        if len(reliable_models) > 0:
            f.write(f"✓ Most Reliable Models: {', '.join(reliable_models['Model'].tolist())}\n")
    
    return report_path

def generate_pdf_report(summary_df, folder_path, all_model_details, heatmap_path):
    """Generate a comprehensive PDF report with heatmap visualization"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    report_path = os.path.join(folder_path, "cv_analysis_report.pdf")
    
    try:
        doc = SimpleDocTemplate(report_path, pagesize=letter, topMargin=0.5*inch)
        styles = getSampleStyleSheet()
        story = []
        
        # Title
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=16,
            alignment=1,  # Center alignment
            spaceAfter=30
        )
        story.append(Paragraph("Cross-Validation Analysis Report", title_style))
        
        # Header info
        header_style = styles['Normal']
        story.append(Paragraph(f"<b>Generated:</b> {timestamp}", header_style))
        story.append(Paragraph(f"<b>Folder:</b> {folder_path}", header_style))
        story.append(Paragraph(f"<b>Models Analyzed:</b> {len(summary_df)}", header_style))
        story.append(Spacer(1, 20))
        
        # Summary table
        story.append(Paragraph("Summary Table", styles['Heading2']))
        
        # Check if F1 data is available
        has_f1_data = any(row['Mean_F1'] > 0 for _, row in summary_df.iterrows())
        has_recall_data = any(row['Mean_Recall'] > 0 for _, row in summary_df.iterrows())
        
        # Prepare table data with F1 and/or Recall if available
        if has_f1_data or has_recall_data:
            # Build header dynamically
            header = ['Model', 'Accuracy±Std', 'Accuracy Range']
            if has_f1_data:
                header.extend(['F1±Std', 'F1 Range'])
            if has_recall_data:
                header.extend(['Recall±Std', 'Recall Range'])
            header.append('Notes')
            
            table_data = [header]
            
            for _, row in summary_df.iterrows():
                acc_std = f"{row['Mean_Acc']:.3f}±{row['Std_Acc']:.3f}"
                f1_std = f"{row['Mean_F1']:.3f}±{row['Std_F1']:.3f}" if has_f1_data and row['Mean_F1'] > 0 else "N/A"
                recall_std = f"{row['Mean_Recall']:.3f}±{row['Std_Recall']:.3f}" if has_recall_data and row['Mean_Recall'] > 0 else "N/A"
                acc_range = f"{row['Min_Acc']:.3f}-{row['Max_Acc']:.3f}"
                
                # Calculate F1 and Recall ranges if available
                f1_range = "N/A"
                recall_range = "N/A"
                model_details = all_model_details.get(row['Model'], {})
                if has_f1_data and model_details.get('f1_stats'):
                    f1_stats = model_details['f1_stats']
                    f1_range = f"{f1_stats['min']:.3f}-{f1_stats['max']:.3f}"
                if has_recall_data and model_details.get('recall_stats'):
                    recall_stats = model_details['recall_stats']
                    recall_range = f"{recall_stats['min']:.3f}-{recall_stats['max']:.3f}"
                
                # Clean notes for PDF compatibility
                notes_clean = row['Notes']
                if len(notes_clean) > 50:
                    notes_clean = notes_clean[:50] + "..."
                
                # Build row dynamically
                row_data = [str(row['Model']), str(acc_std), str(acc_range)]
                if has_f1_data:
                    row_data.extend([str(f1_std), str(f1_range)])
                if has_recall_data:
                    row_data.extend([str(recall_std), str(recall_range)])
                row_data.append(str(notes_clean))
                
                table_data.append(row_data)
            
            # Calculate column widths based on number of metrics
            base_width = 1.0  # Model column
            metric_width = 0.8  # Standard metric columns
            range_width = 0.7   # Range columns
            notes_width = 8.5 - base_width - metric_width - range_width  # Start with base space
            
            col_widths = [base_width*inch, metric_width*inch, range_width*inch]  # Model, Acc±Std, Acc Range
            if has_f1_data:
                col_widths.extend([metric_width*inch, range_width*inch])  # F1±Std, F1 Range
                notes_width -= (metric_width + range_width)
            if has_recall_data:
                col_widths.extend([metric_width*inch, range_width*inch])  # Recall±Std, Recall Range
                notes_width -= (metric_width + range_width)
            col_widths.append(notes_width*inch)  # Notes
            
            table = Table(table_data, colWidths=col_widths)
        else:
            table_data = [['Model', 'Accuracy±Std', 'Min', 'Max', 'Range', 'Primary Notes']]
            
            for _, row in summary_df.iterrows():
                mean_std = f"{row['Mean_Acc']:.3f}±{row['Std_Acc']:.3f}"
                # Clean notes for PDF compatibility
                notes_clean = row['Notes']
                if len(notes_clean) > 100:
                    notes_clean = notes_clean[:100] + "..."
                
                table_data.append([
                    str(row['Model']),
                    str(mean_std),
                    f"{row['Min_Acc']:.3f}",
                    f"{row['Max_Acc']:.3f}",
                    f"{row['Range']:.3f}",
                    str(notes_clean)
                ])
            
            # Create table with original column widths
            table = Table(table_data, colWidths=[1.1*inch, 0.9*inch, 0.5*inch, 0.5*inch, 0.5*inch, 3.0*inch])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('ALIGN', (-1, 1), (-1, -1), 'LEFT'),  # Left align the notes column
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 9),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('FONTSIZE', (0, 1), (-1, -1), 8),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('WORDWRAP', (-1, 1), (-1, -1), True),  # Enable word wrap for notes column
        ]))
        
        story.append(table)
        story.append(Spacer(1, 15))
        
        # Add heatmap visualization
        if heatmap_path and os.path.exists(heatmap_path):
            story.append(Paragraph("Visual Performance Analysis", styles['Heading2']))
            try:
                # Add the heatmap image to the PDF with optimized size
                img = Image(heatmap_path, width=6.5*inch, height=4*inch)
                story.append(img)
                story.append(Spacer(1, 15))
            except Exception as e:
                story.append(Paragraph(f"Could not include heatmap: {str(e)}", styles['Normal']))
                story.append(Spacer(1, 15))
        
        # Build PDF
        doc.build(story)
        return report_path
        
    except Exception as e:
        print(f"PDF generation failed: {str(e)}")
        return None

def create_heatmap_table(summary_df, folder_path):
    """Create a comprehensive heatmap-style visualization of the CV results"""
    
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("viridis")
    
    # Prepare data for heatmap
    models = summary_df['Model'].tolist()
    
    # Create figure with better layout
    fig = plt.figure(figsize=(18, 10))
    gs = fig.add_gridspec(2, 2, height_ratios=[3, 1], width_ratios=[2, 1], hspace=0.3, wspace=0.3)
    
    # Main performance heatmap
    ax1 = fig.add_subplot(gs[0, :])
    
    # Select key metrics for the main heatmap (include F1 and Recall if available)
    has_f1_data = any(summary_df['Mean_F1'] > 0)
    has_recall_data = any(summary_df['Mean_Recall'] > 0)
    
    # Build metrics list dynamically
    metrics = ['Mean_Acc', 'Std_Acc', 'Range']
    metric_labels = ['Accuracy (Mean)', 'Accuracy (Std)', 'Accuracy (Range)']
    
    if has_f1_data:
        metrics.extend(['Mean_F1', 'Std_F1'])
        metric_labels.extend(['F1 (Mean)', 'F1 (Std)'])
    
    if has_recall_data:
        metrics.extend(['Mean_Recall', 'Std_Recall'])
        metric_labels.extend(['Recall (Mean)', 'Recall (Std)'])
    
    data_matrix = summary_df[metrics].values
    
    # Create custom colormap: Green for good, Red for bad
    # For accuracy/F1 metrics: higher is better (green)
    # For std dev and range: lower is better (green)
    
    # Normalize data for better color mapping
    normalized_data = data_matrix.copy()
    
    # Dynamic normalization based on available metrics
    col_idx = 0
    
    # Mean Accuracy (higher is better)
    normalized_data[:, col_idx] = (data_matrix[:, col_idx] - data_matrix[:, col_idx].min()) / (data_matrix[:, col_idx].max() - data_matrix[:, col_idx].min() + 1e-10)
    col_idx += 1
    
    # Accuracy Std (lower is better - flip)
    normalized_data[:, col_idx] = 1 - (data_matrix[:, col_idx] - data_matrix[:, col_idx].min()) / (data_matrix[:, col_idx].max() - data_matrix[:, col_idx].min() + 1e-10)
    col_idx += 1
    
    # Accuracy Range (lower is better - flip)
    normalized_data[:, col_idx] = 1 - (data_matrix[:, col_idx] - data_matrix[:, col_idx].min()) / (data_matrix[:, col_idx].max() - data_matrix[:, col_idx].min() + 1e-10)
    col_idx += 1
    
    if has_f1_data:
        # Mean F1 (higher is better)
        normalized_data[:, col_idx] = (data_matrix[:, col_idx] - data_matrix[:, col_idx].min()) / (data_matrix[:, col_idx].max() - data_matrix[:, col_idx].min() + 1e-10)
        col_idx += 1
        # F1 Std (lower is better - flip)
        normalized_data[:, col_idx] = 1 - (data_matrix[:, col_idx] - data_matrix[:, col_idx].min()) / (data_matrix[:, col_idx].max() - data_matrix[:, col_idx].min() + 1e-10)
        col_idx += 1
    
    if has_recall_data:
        # Mean Recall (higher is better)
        normalized_data[:, col_idx] = (data_matrix[:, col_idx] - data_matrix[:, col_idx].min()) / (data_matrix[:, col_idx].max() - data_matrix[:, col_idx].min() + 1e-10)
        col_idx += 1
        # Recall Std (lower is better - flip)
        normalized_data[:, col_idx] = 1 - (data_matrix[:, col_idx] - data_matrix[:, col_idx].min()) / (data_matrix[:, col_idx].max() - data_matrix[:, col_idx].min() + 1e-10)
        col_idx += 1
    
    im1 = ax1.imshow(normalized_data, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
    
    # Set ticks and labels
    ax1.set_xticks(range(len(metrics)))
    ax1.set_xticklabels(metric_labels, fontsize=12, fontweight='bold')
    ax1.set_yticks(range(len(models)))
    ax1.set_yticklabels([f"{i+1}. {model}" for i, model in enumerate(models)], fontsize=10)
    
    # Add value annotations
    for i in range(len(models)):
        for j in range(len(metrics)):
            value = data_matrix[i, j]
            # Choose text color based on background
            bg_intensity = normalized_data[i, j]
            text_color = 'black'
            ax1.text(j, i, f'{value:.3f}', ha="center", va="center", 
                    color=text_color, fontweight='bold', fontsize=9)
    
    ax1.set_title('Cross-Validation Performance Overview', fontsize=16, fontweight='bold', pad=20)
    
    # Add colorbar
    cbar1 = plt.colorbar(im1, ax=ax1, shrink=0.8)
    cbar1.set_label('Performance Quality\n(Green = Better)', rotation=270, labelpad=20, fontsize=11)
    
    # Model ranking subplot
    ax2 = fig.add_subplot(gs[1, 0])
    
    # Create ranking bars
    mean_accs = summary_df['Mean_Acc'].values
    colors_rank = plt.cm.RdYlGn(mean_accs / mean_accs.max())
    
    bars = ax2.barh(range(len(models)), mean_accs, color=colors_rank)
    ax2.set_yticks(range(len(models)))
    ax2.set_yticklabels([f"{i+1}. {model}" for i, model in enumerate(models)], fontsize=9)
    ax2.set_xlabel('Accuracy (Mean)', fontsize=11, fontweight='bold')
    ax2.set_title('Performance Ranking', fontsize=12, fontweight='bold')
    ax2.grid(axis='x', alpha=0.3)
    
    # Add value labels on bars
    for i, (bar, acc) in enumerate(zip(bars, mean_accs)):
        ax2.text(acc - 0.01, bar.get_y() + bar.get_height()/2, 
                f'{acc:.3f}', ha='right', va='center', fontweight='bold', fontsize=8)
    
    # Stability analysis subplot
    ax3 = fig.add_subplot(gs[1, 1])
    
    std_accs = summary_df['Std_Acc'].values
    # Invert colors for std dev (lower is better)
    colors_std = plt.cm.RdYlGn(1 - std_accs / std_accs.max())
    
    scatter = ax3.scatter(mean_accs, std_accs, c=colors_std, s=100, alpha=0.8, edgecolors='black')
    ax3.set_xlabel('Accuracy (Mean)', fontsize=10, fontweight='bold')
    ax3.set_ylabel('Accuracy (Std)', fontsize=10, fontweight='bold')
    ax3.set_title('Accuracy vs Stability', fontsize=11, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    # Add quadrant labels
    ax3.axhline(y=np.median(std_accs), color='gray', linestyle='--', alpha=0.5)
    ax3.axvline(x=np.median(mean_accs), color='gray', linestyle='--', alpha=0.5)
    
    # Add annotations for best models
    for i, (acc, std, model) in enumerate(zip(mean_accs[:3], std_accs[:3], models[:3])):
        ax3.annotate(f'{i+1}', (acc, std), xytext=(5, 5), textcoords='offset points',
                    fontsize=8, fontweight='bold')
    
    # Add summary statistics text box
    summary_text = f"""Performance Summary:
Best Model: {models[0]} ({mean_accs[0]:.3f})
Most Stable: {summary_df.loc[summary_df['Std_Acc'].idxmin(), 'Model']} (σ={summary_df['Std_Acc'].min():.3f})
Average Performance: {mean_accs.mean():.3f} ± {mean_accs.std():.3f}"""
    
    fig.text(0.02, 0.02, summary_text, fontsize=10, 
             bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
            
    # Save the heatmap
    heatmap_path = os.path.join(folder_path, "cv_results_heatmap.png")
    plt.savefig(heatmap_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    return heatmap_path

if __name__ == "__main__":
    main()
