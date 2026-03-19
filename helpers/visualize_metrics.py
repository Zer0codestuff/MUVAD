import pandas as pd
import matplotlib.pyplot as plt
import os
import glob
import numpy as np

# Chart style configuration
try:
    plt.style.use('seaborn-v0_8-whitegrid')
except:
    plt.style.use('default')
    plt.rcParams['axes.grid'] = True
    plt.rcParams['grid.alpha'] = 0.3
    
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 13

# Find all metrics.csv files in the results folder
metrics_files = glob.glob('results/*/metrics.csv')

if not metrics_files:
    print("No metrics.csv files found in results/ folder")
    exit(1)

# Show available files
print("Available metrics.csv files:")
print("-" * 60)
for i, file_path in enumerate(metrics_files, 1):
    # Extract experiment name from path
    experiment_name = file_path.split('/')[1]
    print(f"{i}. {experiment_name}")
    print(f"   {file_path}")

print("-" * 60)

# Ask user to choose
while True:
    try:
        choice = input(f"\nChoose file to visualize (1-{len(metrics_files)}): ")
        choice_idx = int(choice) - 1
        if 0 <= choice_idx < len(metrics_files):
            csv_path = metrics_files[choice_idx]
            break
        else:
            print(f"Error: enter a number between 1 and {len(metrics_files)}")
    except ValueError:
        print("Error: enter a valid number")
    except KeyboardInterrupt:
        print("\n\nOperation cancelled.")
        exit(0)

print(f"\nSelected file: {csv_path}\n")

# Read CSV file (use '|' as separator)
df = pd.read_csv(csv_path, sep='|')

# Calculate negative metrics (Specificity/TNR, FPR, NPV, FOR)
df['tnr'] = np.where((df['tn'] + df['fp']) > 0, df['tn'] / (df['tn'] + df['fp']), np.nan)
df['fpr'] = np.where((df['tn'] + df['fp']) > 0, df['fp'] / (df['tn'] + df['fp']), np.nan)
df['npv'] = np.where((df['tn'] + df['fn']) > 0, df['tn'] / (df['tn'] + df['fn']), np.nan)
df['for'] = np.where((df['tn'] + df['fn']) > 0, df['fn'] / (df['tn'] + df['fn']), np.nan)

# Remove only the "overall" row for category charts
df_categories = df[df['subset'] != 'overall'].copy()

# Create a folder to save charts directly in plots/
experiment_dir = os.path.dirname(csv_path)
output_dir = os.path.join(experiment_dir, 'plots')
os.makedirs(output_dir, exist_ok=True)

# 1. Bar chart for Accuracy per category
plt.figure(figsize=(14, 6))
bars = plt.bar(df_categories['subset'], df_categories['accuracy'], color='#9b59b6', edgecolor='#8e44ad', linewidth=1.5)
plt.xlabel('Category', fontsize=12)
plt.ylabel('Accuracy', fontsize=12)
plt.title('Accuracy per Category', fontsize=14, fontweight='bold')
plt.xticks(rotation=45, ha='right')
plt.ylim(0, 1.1)
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 0.02,
             f'{height:.3f}', ha='center', va='bottom', fontsize=9)
plt.tight_layout()
plt.savefig(f'{output_dir}/accuracy_per_category.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"Saved: {output_dir}/accuracy_per_category.png")

# 2. Bar chart for Recall (excluding Normal_Videos)
df_anomalous = df_categories[df_categories['subset'] != 'Normal_Videos'].copy()
plt.figure(figsize=(14, 6))
bars = plt.bar(df_anomalous['subset'], df_anomalous['recall'], color='#2ecc71', edgecolor='#27ae60', linewidth=1.5)
plt.xlabel('Category', fontsize=12)
plt.ylabel('Recall', fontsize=12)
plt.title('Recall per Category', fontsize=14, fontweight='bold')
plt.xticks(rotation=45, ha='right')
plt.ylim(0, 1.1)
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 0.02,
             f'{height:.3f}', ha='center', va='bottom', fontsize=9)
plt.tight_layout()
plt.savefig(f'{output_dir}/recall_per_category.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"Saved: {output_dir}/recall_per_category.png")

# 3. Chart for Confusion Matrix: TP, FN for anomalous categories (excluding Normal_Videos)
fig, ax = plt.subplots(figsize=(14, 6))
x = range(len(df_anomalous))
width = 0.35

plt.bar([i - width/2 for i in x], df_anomalous['tp'], width, label='True Positives', color='#27ae60', edgecolor='#1e8449', linewidth=1.2)
plt.bar([i + width/2 for i in x], df_anomalous['fn'], width, label='False Negatives', color='#e74c3c', edgecolor='#c0392b', linewidth=1.2)

plt.xlabel('Category', fontsize=12)
plt.ylabel('Number of Samples', fontsize=12)
plt.title('Confusion Matrix - Anomalous Categories (TP, FN)', fontsize=14, fontweight='bold')
plt.xticks(x, df_anomalous['subset'], rotation=45, ha='right')
plt.legend(frameon=True, shadow=True)
plt.tight_layout()
plt.savefig(f'{output_dir}/confusion_matrix_anomalous.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"Saved: {output_dir}/confusion_matrix_anomalous.png")

# 6. Dedicated chart for Normal_Videos: TN and FP
nv_row_df = df_categories[df_categories['subset'] == 'Normal_Videos']
if not nv_row_df.empty:
    nv_row = nv_row_df.iloc[0]
    labels = ['True Negatives', 'False Positives']
    values = [nv_row['tn'], nv_row['fp']]
    
    plt.figure(figsize=(8, 6))
    bars = plt.bar(labels, values, color=['#3498db', '#e67e22'], edgecolor=['#2980b9', '#d35400'], linewidth=1.5)
    plt.ylabel('Number of Samples', fontsize=12)
    plt.title('Normal_Videos: Confusion Matrix (TN, FP)', fontsize=14, fontweight='bold')
    for bar, val in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width()/2, val + 1, f'{int(val)}', ha='center', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/confusion_matrix_normal.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir}/confusion_matrix_normal.png")

# 7. Improved Heatmap of main metrics (added TNR and FPR)
metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1_score', 'tnr', 'fpr']
metric_labels = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'TNR', 'FPR']
heatmap_data = df_categories[['subset'] + metrics_to_plot].set_index('subset')
categories = heatmap_data.index.tolist()

data = heatmap_data[metrics_to_plot].T.values  # shape: (num_metrics, num_categories)

# Mask NaN and set color for cells without value
masked_data = np.ma.masked_invalid(data)
cmap = plt.get_cmap('RdYlGn').copy()  # Red-Yellow-Green colormap (better for metrics)
cmap.set_bad(color='#d3d3d3')

fig, ax = plt.subplots(figsize=(max(14, len(categories) * 0.9), 9))
im = ax.imshow(masked_data, cmap=cmap, aspect='auto', vmin=0, vmax=1)

# Set ticks and labels
ax.set_xticks(np.arange(len(categories)))
ax.set_yticks(np.arange(len(metrics_to_plot)))
ax.set_xticklabels(categories, rotation=45, ha='right', fontsize=11)
ax.set_yticklabels(metric_labels, fontsize=11)

# Add grid lines
ax.set_xticks(np.arange(len(categories)) - 0.5, minor=True)
ax.set_yticks(np.arange(len(metrics_to_plot)) - 0.5, minor=True)
ax.grid(which='minor', color='white', linestyle='-', linewidth=2)

# Add values in cells (empty if NaN)
for i in range(masked_data.shape[0]):  # metrics
    for j in range(masked_data.shape[1]):  # categories
        val = data[i, j]
        if isinstance(val, float) and np.isnan(val):
            label = 'N/A'
            text_color = 'gray'
        else:
            label = f'{val:.3f}'
            # Choose text color based on background
            text_color = 'white' if val < 0.5 else 'black'
        ax.text(j, i, label, ha='center', va='center', color=text_color, fontsize=10, fontweight='bold')

# Add colorbar
cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label('Score', rotation=270, labelpad=25, fontsize=12)
cbar.ax.tick_params(labelsize=10)

plt.title('Metrics Heatmap per Category', fontsize=15, fontweight='bold', pad=20)
plt.xlabel('Category', fontsize=12, fontweight='bold')
plt.ylabel('Metric', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{output_dir}/heatmap_metriche.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"Saved: {output_dir}/heatmap_metriche.png")

# 8. Improved Pie chart for sample distribution (TP + FN, or TN + FP for Normal_Videos)
# Calculate total samples per category consistently (positive for classes, negative for Normal_Videos)
df_categories['total_samples'] = np.where(
    df_categories['subset'] == 'Normal_Videos',
    df_categories['tn'] + df_categories['fp'],
    df_categories['tp'] + df_categories['fn']
)

# Remove any categories with 0 samples to avoid null slices
df_pie = df_categories[df_categories['total_samples'] > 0].copy()

# Define a beautiful color palette
beautiful_colors = [
    '#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8',
    '#F7DC6F', '#BB8FCE', '#85C1E2', '#F8B195', '#C06C84',
    '#6C5B7B', '#355C7D', '#99B898', '#FECEAB', '#F38181',
    '#AA96DA', '#FCBAD3', '#FFFFD2', '#A8D8EA', '#FFCBC1'
]

fig, ax = plt.subplots(figsize=(12, 12))

# Autopct that shows percentages only if >= 1%
def autopct_format(values):
    def _fmt(pct):
        return f'{pct:.1f}%' if pct >= 1 else ''
    return _fmt

wedges, texts, autotexts = ax.pie(
    df_pie['total_samples'],
    labels=df_pie['subset'],
    autopct=autopct_format(df_pie['total_samples']),
    startangle=140,
    colors=beautiful_colors[:len(df_pie)],
    pctdistance=0.85,
    labeldistance=1.05,
    wedgeprops=dict(edgecolor='white', linewidth=2),
    textprops={'fontsize': 11, 'weight': 'bold'}
)

ax.axis('equal')  # ensure pie is circular

# Make percentage text more visible
plt.setp(autotexts, size=11, weight='bold', color='white')
plt.setp(texts, size=11, weight='bold')

plt.title('Sample Distribution per Category', fontsize=16, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig(f'{output_dir}/distribuzione_campioni.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"Saved: {output_dir}/distribuzione_campioni.png")

# 9. Summary chart of overall metrics
overall_row = df[df['subset'] == 'overall'].iloc[0]
metrics_overall = {
    'Accuracy': overall_row['accuracy'],
    'Precision': overall_row['precision'],
    'Recall': overall_row['recall'],
    'F1-Score': overall_row['f1_score'],
    'AUC': overall_row['auc']
}

plt.figure(figsize=(10, 6))
colors = ['#3498db', '#e74c3c', '#2ecc71', '#9b59b6', '#f39c12']
edge_colors = ['#2980b9', '#c0392b', '#27ae60', '#8e44ad', '#e67e22']
bars = plt.bar(metrics_overall.keys(), metrics_overall.values(), color=colors, edgecolor=edge_colors, linewidth=2)
plt.xlabel('Metric', fontsize=12)
plt.ylabel('Score', fontsize=12)
plt.title('Overall Model Metrics', fontsize=14, fontweight='bold')
plt.ylim(0, 1.1)
for i, (key, value) in enumerate(metrics_overall.items()):
    plt.text(i, value + 0.02, f'{value:.3f}', ha='center', fontsize=11, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{output_dir}/overall_metrics.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"Saved: {output_dir}/overall_metrics.png")

# 10. F1-Score chart sorted (excluding Normal_Videos)
df_f1_sorted = df_categories[df_categories['subset'] != 'Normal_Videos'].copy()
df_f1_sorted = df_f1_sorted.sort_values('f1_score', ascending=True)
plt.figure(figsize=(10, 8))
colors_gradient = plt.cm.viridis(np.linspace(0.3, 0.9, len(df_f1_sorted)))
bars = plt.barh(df_f1_sorted['subset'], df_f1_sorted['f1_score'], color=colors_gradient, edgecolor='black', linewidth=0.8)
plt.xlabel('F1-Score', fontsize=12, fontweight='bold')
plt.ylabel('Category', fontsize=12, fontweight='bold')
plt.title('F1-Score per Category (Sorted)', fontsize=14, fontweight='bold')
plt.xlim(0, 1.1)
for i, (cat, score) in enumerate(zip(df_f1_sorted['subset'], df_f1_sorted['f1_score'])):
    plt.text(score + 0.02, i, f'{score:.3f}', va='center', fontsize=10, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{output_dir}/f1_score_sorted.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"Saved: {output_dir}/f1_score_sorted.png")

# 11. Dedicated chart for Normal_Videos (TNR, FPR, Accuracy)
nv_row_df = df_categories[df_categories['subset'] == 'Normal_Videos']
if not nv_row_df.empty:
    nv_row = nv_row_df.iloc[0]
    labels = ['TNR (Specificity)', 'FPR (False Alarm)', 'Accuracy']
    values = [nv_row['tnr'], nv_row['fpr'], nv_row['accuracy']]

    plt.figure(figsize=(10, 6))
    colors = ['#3498db', '#e67e22', '#2ecc71']
    edge_colors = ['#2980b9', '#d35400', '#27ae60']
    bars = plt.bar(labels, values, color=colors, edgecolor=edge_colors, linewidth=2)
    plt.ylim(0, 1.1)
    plt.ylabel('Score', fontsize=12)
    plt.title('Normal_Videos: Key Metrics', fontsize=14, fontweight='bold')
    for bar, val in zip(bars, values):
        if isinstance(val, float) and not np.isnan(val):
            plt.text(bar.get_x() + bar.get_width()/2, val + 0.02, f'{val:.3f}', ha='center', fontsize=11, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/normal_videos_metrics.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir}/normal_videos_metrics.png")

print(f"\n✓ All charts have been generated in folder: {output_dir}")
