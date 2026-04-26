"""
Complete Evaluation Pipeline for ECG Denoiser Manuscript
Computes all metrics, generates figures, and exports results.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving figures
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MaxNLocator
import math
import os
import warnings
warnings.filterwarnings('ignore')

# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────
RESULTS_FILE = 'results/y_pred_360/results_64_bi_drop0_test_set_db.txt.txt'
FIG_DIR = 'paper_figures'
SAMPLING_RATE = 360
N_SAMPLES = 3267
TIMESTEPS = 3600  # 10 seconds at 360 Hz

os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(f'{FIG_DIR}/individual', exist_ok=True)

# Plotting style
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
})

# ──────────────────────────────────────────────
# Helper Functions (from tools/compute_metrics_with_GT.py)
# ──────────────────────────────────────────────
def signaltonoise(sig_ori, sig_hat):
    sum_top = sum(sig_ori**2)
    sum_bot = sum((sig_hat - sig_ori)**2)
    if sum_bot == 0:
        return float('inf')
    quo = sum_top / sum_bot
    if quo <= 0:
        return float('-inf')
    snr = 10 * math.log(quo, 10)
    return snr

def signaltonoise_imp(sig, sig_noisy, sig_pred):
    return signaltonoise(sig, sig_pred) - signaltonoise(sig, sig_noisy)

def prd(sig_ori, sig_pred):
    sum_bot = sum(sig_ori ** 2)
    if sum_bot == 0:
        return float('inf')
    sum_top = sum((sig_ori - sig_pred) ** 2)
    return math.sqrt(sum_top / sum_bot) * 100

def rmse(y, y_hat):
    MSE = np.square(np.subtract(y, y_hat)).mean()
    return math.sqrt(MSE)

# ──────────────────────────────────────────────
# Load prediction results
# ──────────────────────────────────────────────
print("Loading prediction results file ...")
res_file = pd.read_csv(RESULTS_FILE)
print(f"Loaded {len(res_file)} rows")

# ──────────────────────────────────────────────
# Extract signals for each sample and compute metrics
# ──────────────────────────────────────────────
print("\nComputing evaluation metrics for all 3,267 test samples...")

all_results = []
noise_class_names = {
    (1, 0, 0): 'MA',
    (0, 1, 0): 'EM',
    (0, 0, 1): 'BW',
    (1, 0, 1): 'BW+MA',
    (0, 1, 1): 'BW+EM',
    (1, 1, 0): 'MA+EM',
    (1, 1, 1): 'MA+EM+BW'
}

for i in range(N_SAMPLES):
    if i % 500 == 0:
        print(f"  Processing sample {i}/{N_SAMPLES}...")
    
    try:
        # Extract signals (3601 rows per sample: 1 header + 3600 data)
        start = i * (TIMESTEPS + 1) + 1
        end = (i + 1) * (TIMESTEPS + 1) - 1
        
        ecg_noisy = np.array(res_file['Noisy'][start:end], dtype=float)
        ecg_real = np.array(res_file['Real'][start:end], dtype=float)
        ecg_pred = np.array(res_file['Predicted'][start:end], dtype=float)
        
        if len(ecg_real) < 100:
            continue
        
        # Load noise class and SNR info
        class_arr = np.load(f'data/Y_class/Y_test2/{i}.npy')
        noise_info = np.load(f'data/Y_test_noise_timesteps/{i}.npy')
        
        noise_type = noise_class_names.get(tuple(class_arr.astype(int)), 'Unknown')
        snr_in_target = int(noise_info[2])
        
        # Compute metrics
        snr_in = signaltonoise(ecg_real, ecg_noisy)
        snr_out = signaltonoise(ecg_real, ecg_pred)
        snr_imp = signaltonoise_imp(ecg_real, ecg_noisy, ecg_pred)
        prd_val = prd(ecg_real, ecg_pred)
        rmse_val = rmse(ecg_real, ecg_pred)
        
        all_results.append({
            'sample_id': i,
            'noise_type': noise_type,
            'snr_in_target': snr_in_target,
            'SNR_in': snr_in,
            'SNR_out': snr_out,
            'SNR_imp': snr_imp,
            'PRD': prd_val,
            'RMSE': rmse_val
        })
    except Exception as e:
        pass

df_results = pd.DataFrame(all_results)

# Filter extreme outliers for cleaner statistics
df_clean = df_results[
    (df_results['SNR_in'].abs() < 100) & 
    (df_results['SNR_out'].abs() < 100) & 
    (df_results['PRD'] < 500)
].copy()

print(f"\nComputed metrics for {len(df_clean)} valid samples out of {N_SAMPLES}")

# ──────────────────────────────────────────────
# TABLE 1: Overall Metrics Summary
# ──────────────────────────────────────────────
print("\n" + "="*70)
print("TABLE 1: Overall Evaluation Metrics")
print("="*70)
summary = df_clean[['SNR_in', 'SNR_out', 'SNR_imp', 'PRD', 'RMSE']].describe()
print(summary.round(4))
summary.to_csv(f'{FIG_DIR}/table1_overall_metrics.csv')

# ──────────────────────────────────────────────
# TABLE 2: Metrics by Noise Type
# ──────────────────────────────────────────────
print("\n" + "="*70)
print("TABLE 2: Evaluation Metrics by Noise Type")
print("="*70)
by_noise = df_clean.groupby('noise_type')[['SNR_in', 'SNR_out', 'SNR_imp', 'PRD', 'RMSE']].agg(['mean', 'std', 'count'])
print(by_noise.round(4))
by_noise.to_csv(f'{FIG_DIR}/table2_metrics_by_noise.csv')

# ──────────────────────────────────────────────
# TABLE 3: Metrics by Target SNR
# ──────────────────────────────────────────────
print("\n" + "="*70)
print("TABLE 3: Evaluation Metrics by Target SNR Input")
print("="*70)
by_snr = df_clean.groupby('snr_in_target')[['SNR_out', 'SNR_imp', 'PRD', 'RMSE']].agg(['mean', 'std'])
print(by_snr.round(4))
by_snr.to_csv(f'{FIG_DIR}/table3_metrics_by_snr.csv')

# ──────────────────────────────────────────────
# Save full results
# ──────────────────────────────────────────────
df_results.to_csv(f'{FIG_DIR}/full_results.csv', index=False)
print(f"\nFull results saved to {FIG_DIR}/full_results.csv")

# ╔════════════════════════════════════════════════╗
# ║           FIGURE GENERATION                     ║
# ╚════════════════════════════════════════════════╝

# ──────────────────────────────────────────────
# FIGURE 1: Example Denoising Visualizations (3 noise types)
# ──────────────────────────────────────────────
print("\nGenerating Figure 1: Denoising examples for each noise type...")

# Find good examples for each noise type
target_types = ['MA', 'EM', 'BW']
example_indices = {}
for nt in target_types:
    candidates = df_clean[df_clean['noise_type'] == nt].sort_values('SNR_imp', ascending=False)
    if len(candidates) > 10:
        example_indices[nt] = candidates.iloc[5]['sample_id']  # Take a representative (not best/worst)
    elif len(candidates) > 0:
        example_indices[nt] = candidates.iloc[0]['sample_id']

fig, axes = plt.subplots(len(example_indices), 3, figsize=(14, 3.2 * len(example_indices)), 
                          gridspec_kw={'hspace': 0.45, 'wspace': 0.15})

for row, (nt, idx) in enumerate(example_indices.items()):
    idx = int(idx)
    start = idx * (TIMESTEPS + 1) + 1
    end = (idx + 1) * (TIMESTEPS + 1) - 1
    
    ecg_noisy = np.array(res_file['Noisy'][start:end], dtype=float)
    ecg_real = np.array(res_file['Real'][start:end], dtype=float)
    ecg_pred = np.array(res_file['Predicted'][start:end], dtype=float)
    time_sec = np.arange(len(ecg_real)) / SAMPLING_RATE
    
    # Column 1: Clean
    axes[row, 0].plot(time_sec, ecg_real, color='#2ecc71', linewidth=0.6)
    axes[row, 0].set_ylabel(f'{nt} Noise', fontweight='bold')
    if row == 0:
        axes[row, 0].set_title('Clean Ground Truth', fontweight='bold')
    axes[row, 0].grid(True, alpha=0.2)
    
    # Column 2: Noisy Input
    axes[row, 1].plot(time_sec, ecg_noisy, color='#e74c3c', linewidth=0.6)
    if row == 0:
        axes[row, 1].set_title('Noisy Input', fontweight='bold')
    axes[row, 1].grid(True, alpha=0.2)
    
    # Column 3: Denoised Output
    axes[row, 2].plot(time_sec, ecg_pred, color='#3498db', linewidth=0.6)
    if row == 0:
        axes[row, 2].set_title('biGRU Denoised Output', fontweight='bold')
    axes[row, 2].grid(True, alpha=0.2)
    
    if row == len(example_indices) - 1:
        for c in range(3):
            axes[row, c].set_xlabel('Time (s)')

fig.suptitle('Figure 1: ECG Denoising Examples by Noise Type', fontsize=14, fontweight='bold', y=1.02)
plt.savefig(f'{FIG_DIR}/fig1_denoising_examples.png', dpi=300)
plt.savefig(f'{FIG_DIR}/fig1_denoising_examples.pdf')
plt.close()
print("  ✓ Saved fig1_denoising_examples.png/pdf")

# ──────────────────────────────────────────────
# FIGURE 2: Box Plot of RMSE by Noise Type
# ──────────────────────────────────────────────
print("Generating Figure 2: RMSE distribution by noise type...")

fig, ax = plt.subplots(figsize=(8, 5))
noise_types_ordered = ['MA', 'EM', 'BW', 'MA+EM', 'BW+MA', 'BW+EM', 'MA+EM+BW']
data_for_box = [df_clean[df_clean['noise_type'] == nt]['RMSE'].values 
                for nt in noise_types_ordered if nt in df_clean['noise_type'].values]
labels = [nt for nt in noise_types_ordered if nt in df_clean['noise_type'].values]

bp = ax.boxplot(data_for_box, labels=labels, patch_artist=True, 
                showfliers=True, flierprops={'markersize': 2, 'alpha': 0.3})

colors = ['#3498db', '#e74c3c', '#2ecc71', '#9b59b6', '#f39c12', '#1abc9c', '#e67e22']
for patch, color in zip(bp['boxes'], colors[:len(labels)]):
    patch.set_facecolor(color)
    patch.set_alpha(0.6)

ax.set_xlabel('Noise Type')
ax.set_ylabel('RMSE')
ax.set_title('Figure 2: RMSE Distribution by Noise Type', fontweight='bold')
ax.grid(True, axis='y', alpha=0.2)
plt.savefig(f'{FIG_DIR}/fig2_rmse_by_noise.png', dpi=300)
plt.savefig(f'{FIG_DIR}/fig2_rmse_by_noise.pdf')
plt.close()
print("  ✓ Saved fig2_rmse_by_noise.png/pdf")

# ──────────────────────────────────────────────
# FIGURE 3: SNR Improvement by Target SNR Input
# ──────────────────────────────────────────────
print("Generating Figure 3: SNR improvement by input SNR level...")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Panel A: SNR output by target SNR
snr_groups = [0, 5, 7, 10]
snr_data_out = [df_clean[df_clean['snr_in_target'] == s]['SNR_out'].values for s in snr_groups]
snr_data_imp = [df_clean[df_clean['snr_in_target'] == s]['SNR_imp'].values for s in snr_groups]

bp1 = axes[0].boxplot(snr_data_out, labels=[f'{s} dB' for s in snr_groups], patch_artist=True,
                       showfliers=False)
for patch in bp1['boxes']:
    patch.set_facecolor('#3498db')
    patch.set_alpha(0.6)
axes[0].set_xlabel('Target Input SNR')
axes[0].set_ylabel('Output SNR (dB)')
axes[0].set_title('(a) Output SNR by Input Level', fontweight='bold')
axes[0].grid(True, axis='y', alpha=0.2)

# Panel B: SNR improvement
bp2 = axes[1].boxplot(snr_data_imp, labels=[f'{s} dB' for s in snr_groups], patch_artist=True,
                       showfliers=False)
for patch in bp2['boxes']:
    patch.set_facecolor('#2ecc71')
    patch.set_alpha(0.6)
axes[1].set_xlabel('Target Input SNR')
axes[1].set_ylabel('SNR Improvement (dB)')
axes[1].set_title('(b) SNR Improvement by Input Level', fontweight='bold')
axes[1].grid(True, axis='y', alpha=0.2)
axes[1].axhline(y=0, color='red', linestyle='--', alpha=0.5, label='No improvement')
axes[1].legend()

fig.suptitle('Figure 3: Denoising Performance vs. Input Noise Level', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{FIG_DIR}/fig3_snr_by_input.png', dpi=300)
plt.savefig(f'{FIG_DIR}/fig3_snr_by_input.pdf')
plt.close()
print("  ✓ Saved fig3_snr_by_input.png/pdf")

# ──────────────────────────────────────────────
# FIGURE 4: Histogram of RMSE and SNR Improvement
# ──────────────────────────────────────────────
print("Generating Figure 4: Metric distributions...")

fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

axes[0].hist(df_clean['RMSE'], bins=50, color='#3498db', alpha=0.7, edgecolor='white')
axes[0].axvline(df_clean['RMSE'].mean(), color='red', linestyle='--', label=f"Mean: {df_clean['RMSE'].mean():.4f}")
axes[0].axvline(df_clean['RMSE'].median(), color='orange', linestyle='--', label=f"Median: {df_clean['RMSE'].median():.4f}")
axes[0].set_xlabel('RMSE')
axes[0].set_ylabel('Count')
axes[0].set_title('(a) RMSE Distribution', fontweight='bold')
axes[0].legend(fontsize=8)
axes[0].grid(True, alpha=0.2)

axes[1].hist(df_clean['SNR_imp'], bins=50, color='#2ecc71', alpha=0.7, edgecolor='white')
axes[1].axvline(df_clean['SNR_imp'].mean(), color='red', linestyle='--', label=f"Mean: {df_clean['SNR_imp'].mean():.2f} dB")
axes[1].set_xlabel('SNR Improvement (dB)')
axes[1].set_ylabel('Count')
axes[1].set_title('(b) SNR Improvement Distribution', fontweight='bold')
axes[1].legend(fontsize=8)
axes[1].grid(True, alpha=0.2)

axes[2].hist(df_clean['PRD'], bins=50, color='#e74c3c', alpha=0.7, edgecolor='white')
axes[2].axvline(df_clean['PRD'].mean(), color='red', linestyle='--', label=f"Mean: {df_clean['PRD'].mean():.2f}%")
axes[2].set_xlabel('PRD (%)')
axes[2].set_ylabel('Count')
axes[2].set_title('(c) PRD Distribution', fontweight='bold')
axes[2].legend(fontsize=8)
axes[2].grid(True, alpha=0.2)

fig.suptitle('Figure 4: Evaluation Metrics Distributions', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{FIG_DIR}/fig4_metric_distributions.png', dpi=300)
plt.savefig(f'{FIG_DIR}/fig4_metric_distributions.pdf')
plt.close()
print("  ✓ Saved fig4_metric_distributions.png/pdf")

# ──────────────────────────────────────────────
# FIGURE 5: Scatter — SNR Input vs SNR Output
# ──────────────────────────────────────────────
print("Generating Figure 5: Input vs Output SNR scatter...")

fig, ax = plt.subplots(figsize=(7, 6))
scatter = ax.scatter(df_clean['SNR_in'], df_clean['SNR_out'], 
                     c=df_clean['RMSE'], cmap='viridis_r', 
                     alpha=0.4, s=8, edgecolors='none')
# Identity line
lims = [min(df_clean['SNR_in'].min(), df_clean['SNR_out'].min()),
        max(df_clean['SNR_in'].max(), df_clean['SNR_out'].max())]
ax.plot(lims, lims, 'r--', alpha=0.5, label='No improvement line')
ax.set_xlabel('Input SNR (dB)')
ax.set_ylabel('Output SNR (dB)')
ax.set_title('Figure 5: SNR Before vs. After Denoising', fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.2)
cbar = plt.colorbar(scatter, ax=ax)
cbar.set_label('RMSE')
plt.savefig(f'{FIG_DIR}/fig5_snr_scatter.png', dpi=300)
plt.savefig(f'{FIG_DIR}/fig5_snr_scatter.pdf')
plt.close()
print("  ✓ Saved fig5_snr_scatter.png/pdf")

# ──────────────────────────────────────────────
# FIGURE 6: Model Architecture Diagram (text-based)
# ──────────────────────────────────────────────
print("Generating Figure 6: Model architecture...")

fig, ax = plt.subplots(figsize=(12, 3))
ax.set_xlim(0, 10)
ax.set_ylim(0, 2)
ax.axis('off')

# Draw boxes
boxes = [
    (0.5, 0.7, 'Noisy ECG\n(3600×1)', '#e74c3c'),
    (2.5, 0.7, 'biGRU\nhidden=64\n2 directions', '#3498db'),
    (4.5, 0.7, 'Dropout\np=0.3', '#f39c12'),
    (6.5, 0.7, 'GRU Decoder\n128→1', '#9b59b6'),
    (8.5, 0.7, 'Clean ECG\n(3600×1)', '#2ecc71'),
]

for x, y, text, color in boxes:
    rect = plt.Rectangle((x, y), 1.5, 1, fill=True, facecolor=color, 
                          edgecolor='black', alpha=0.8, linewidth=1.5)
    ax.add_patch(rect)
    ax.text(x + 0.75, y + 0.5, text, ha='center', va='center', 
            fontsize=8, fontweight='bold', color='white')

# Arrows
for i in range(len(boxes) - 1):
    x1 = boxes[i][0] + 1.5
    x2 = boxes[i+1][0]
    ax.annotate('', xy=(x2, 1.2), xytext=(x1, 1.2),
                arrowprops=dict(arrowstyle='->', color='black', lw=1.5))

# Dimension labels
dim_labels = ['', '→ (3600, 128)', '→ (3600, 128)', '→ (3600, 1)', '']
for i, (x, y, _, _) in enumerate(boxes):
    if dim_labels[i]:
        ax.text(x - 0.1, 0.45, dim_labels[i], fontsize=7, color='gray', style='italic')

ax.set_title('Figure 6: biGRU Denoiser Architecture', fontweight='bold', fontsize=13, pad=10)
plt.savefig(f'{FIG_DIR}/fig6_model_architecture.png', dpi=300)
plt.savefig(f'{FIG_DIR}/fig6_model_architecture.pdf')
plt.close()
print("  ✓ Saved fig6_model_architecture.png/pdf")

# ──────────────────────────────────────────────
# FIGURE 7: Detailed before/after comparison (zoomed)
# ──────────────────────────────────────────────
print("Generating Figure 7: Detailed overlay comparison...")

# Pick a good MA example
ma_samples = df_clean[df_clean['noise_type'] == 'MA'].sort_values('SNR_imp', ascending=False)
if len(ma_samples) > 5:
    idx = int(ma_samples.iloc[3]['sample_id'])
else:
    idx = int(ma_samples.iloc[0]['sample_id'])

start = idx * (TIMESTEPS + 1) + 1
end = (idx + 1) * (TIMESTEPS + 1) - 1
ecg_noisy = np.array(res_file['Noisy'][start:end], dtype=float)
ecg_real = np.array(res_file['Real'][start:end], dtype=float)
ecg_pred = np.array(res_file['Predicted'][start:end], dtype=float)
time_sec = np.arange(len(ecg_real)) / SAMPLING_RATE

fig, axes = plt.subplots(3, 1, figsize=(12, 7), sharex=True)

axes[0].plot(time_sec, ecg_real, color='#2ecc71', linewidth=0.8)
axes[0].set_title('Clean Ground Truth', fontweight='bold')
axes[0].set_ylabel('Amplitude')
axes[0].grid(True, alpha=0.2)

axes[1].plot(time_sec, ecg_noisy, color='#e74c3c', linewidth=0.8)
axes[1].set_title('Noisy Input (MA Artifact)', fontweight='bold')
axes[1].set_ylabel('Amplitude')
axes[1].grid(True, alpha=0.2)

axes[2].plot(time_sec, ecg_real, color='#2ecc71', linewidth=0.8, alpha=0.5, label='Ground Truth')
axes[2].plot(time_sec, ecg_pred, color='#3498db', linewidth=0.8, label='Denoised')
axes[2].set_title('Overlay: Ground Truth vs. Denoised Output', fontweight='bold')
axes[2].set_ylabel('Amplitude')
axes[2].set_xlabel('Time (s)')
axes[2].legend()
axes[2].grid(True, alpha=0.2)

sample_metrics = df_clean[df_clean['sample_id'] == idx].iloc[0]
fig.suptitle(f'Figure 7: Detailed Denoising Example (RMSE={sample_metrics["RMSE"]:.4f}, SNR Imp={sample_metrics["SNR_imp"]:.2f} dB)', 
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{FIG_DIR}/fig7_detailed_comparison.png', dpi=300)
plt.savefig(f'{FIG_DIR}/fig7_detailed_comparison.pdf')
plt.close()
print("  ✓ Saved fig7_detailed_comparison.png/pdf")


# ──────────────────────────────────────────────
# Final Summary
# ──────────────────────────────────────────────
print("\n" + "="*70)
print("EVALUATION COMPLETE — SUMMARY")
print("="*70)
print(f"Total samples evaluated: {len(df_clean)}")
print(f"\nOverall Metrics:")
print(f"  RMSE:            {df_clean['RMSE'].mean():.4f} ± {df_clean['RMSE'].std():.4f}")
print(f"  SNR Input:       {df_clean['SNR_in'].mean():.2f} ± {df_clean['SNR_in'].std():.2f} dB")
print(f"  SNR Output:      {df_clean['SNR_out'].mean():.2f} ± {df_clean['SNR_out'].std():.2f} dB")
print(f"  SNR Improvement: {df_clean['SNR_imp'].mean():.2f} ± {df_clean['SNR_imp'].std():.2f} dB")
print(f"  PRD:             {df_clean['PRD'].mean():.2f} ± {df_clean['PRD'].std():.2f}%")
print(f"\nAll figures saved to: {FIG_DIR}/")
print(f"All CSV tables saved to: {FIG_DIR}/")
print("\nGenerated files:")
for f in sorted(os.listdir(FIG_DIR)):
    if not os.path.isdir(f'{FIG_DIR}/{f}'):
        size = os.path.getsize(f'{FIG_DIR}/{f}')
        print(f"  {f:45s} ({size/1024:.1f} KB)")
