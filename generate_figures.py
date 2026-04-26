"""
Generate figures with visually recognizable ECG morphology.
Uses good leads from the 12-lead data + runs the pre-trained model for new predictions.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import math
import os
import sys
import warnings
warnings.filterwarnings('ignore')
import sklearn.preprocessing as pp
from scipy.signal import butter, sosfilt
from scipy.stats import zscore

# Add project root to path
sys.path.insert(0, '.')
from gru_denoiser import GRU

# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────
FIG_DIR = 'paper_figures'
SAMPLING_RATE = 360
os.makedirs(FIG_DIR, exist_ok=True)

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
# Load the pre-trained model
# ──────────────────────────────────────────────
print("Loading pre-trained biGRU model...")
model = GRU(n_features=1, hid_dim=64, n_layers=1, dropout=0, bidirectional=True, gpu_id=None)
model.load_state_dict(torch.load('best_gru_denoiser_360Hz', map_location=torch.device('cpu')))
model.eval()
print("  ✓ Model loaded")

# ──────────────────────────────────────────────
# Load noise data for injection
# ──────────────────────────────────────────────
import pickle
import scipy.signal as si
from scipy.stats import zscore as zsc

print("Loading noise data...")
pickle_in = open('data_noise.pickle', 'rb')
noise_in = pickle.load(pickle_in)
pickle_in.close()

ma_raw = np.concatenate((np.array(noise_in[-1][:, 0]), np.array(noise_in[-1][:, 1])))
em_raw = np.concatenate((np.array(noise_in[-2][:, 0]), np.array(noise_in[-2][:, 1])))
bw_raw = np.concatenate((np.array(noise_in[-3][:, 0]), np.array(noise_in[-3][:, 1])))

num_samples_360 = 360 * 30 * 60 * 2
ma = zsc(si.resample(ma_raw, num_samples_360))
em = zsc(si.resample(em_raw, num_samples_360))
bw = zsc(si.resample(bw_raw, num_samples_360))
print("  ✓ Noise data loaded")

# ──────────────────────────────────────────────
# Helper: find best lead from a 12-lead record
# ──────────────────────────────────────────────
def find_best_lead(ecg_all):
    """Find the lead with the most prominent, clean QRS morphology."""
    best_lead = 0
    best_score = 0
    for lead in range(ecg_all.shape[1]):
        sig = ecg_all[:, lead]
        peaks = si.find_peaks(sig, distance=250, prominence=1.0)[0]
        # Score: number of peaks * mean prominence
        if len(peaks) > 5:
            prominences = si.peak_prominences(sig, peaks)[0]
            score = len(peaks) * np.mean(prominences)
            if score > best_score:
                best_score = score
                best_lead = lead
    return best_lead

def denoise_signal(clean_zscore, noise_signal, noise_factor=1.0):
    """Add noise to clean signal, run model, return all three versions."""
    noisy = clean_zscore + noise_factor * noise_signal
    
    # MinMax normalize for model input (model was trained on minmax-scaled data)
    clean_mm = pp.minmax_scale(clean_zscore)
    noisy_mm = pp.minmax_scale(noisy)
    
    # Reshape for model: (1, timesteps, 1)
    noisy_tensor = torch.from_numpy(noisy_mm.reshape(1, -1, 1)).float()
    
    with torch.no_grad():
        pred_tensor = model(noisy_tensor)
    pred_mm = pred_tensor.numpy().reshape(-1)
    
    return clean_zscore, noisy, clean_mm, noisy_mm, pred_mm

# ──────────────────────────────────────────────
# Find good samples with clear QRS from best leads
# ──────────────────────────────────────────────
print("\nSearching for samples with clear QRS morphology...")

good_samples = []
for idx in range(0, 500):
    try:
        ecg_all = np.load(f'data/Y_test_all_leads/{idx}.npy')
        best_lead = find_best_lead(ecg_all)
        sig = ecg_all[:, best_lead]
        peaks = si.find_peaks(sig, distance=250, prominence=1.0)[0]
        if len(peaks) >= 8:
            prominences = si.peak_prominences(sig, peaks)[0]
            score = len(peaks) * np.mean(prominences)
            good_samples.append((idx, best_lead, score))
    except:
        pass

good_samples.sort(key=lambda x: x[2], reverse=True)
print(f"  Found {len(good_samples)} samples with clear QRS. Top 5 scores: {[f'{s[2]:.1f}' for s in good_samples[:5]]}")

# ──────────────────────────────────────────────
# FIGURE 1 (REVISED): Denoising examples with clear QRS
# ──────────────────────────────────────────────
print("\nGenerating revised Figure 1: Denoising with clear ECG morphology...")

noise_configs = [
    ('Muscle Activation (MA)', 'ma'),
    ('Electrode Motion (EM)', 'em'),
    ('Baseline Wander (BW)', 'bw'),
]

fig, axes = plt.subplots(3, 3, figsize=(14, 9), gridspec_kw={'hspace': 0.5, 'wspace': 0.2})

np.random.seed(42)

for row, (noise_label, noise_type) in enumerate(noise_configs):
    # Pick a different good sample for each noise type
    sample_idx, lead_idx, _ = good_samples[row * 3]
    ecg_all = np.load(f'data/Y_test_all_leads/{sample_idx}.npy')
    clean_sig = zsc(ecg_all[:, lead_idx])
    signal_length = len(clean_sig)
    time_sec = np.arange(signal_length) / SAMPLING_RATE
    
    # Create noise
    if noise_type == 'ma':
        noise_arr = np.zeros_like(clean_sig)
        burst_len = 4 * 360  # 4 seconds
        st_noise = np.random.randint(0, len(ma) - burst_len)
        st_ecg = np.random.randint(0, signal_length - burst_len)
        noise_arr[st_ecg:st_ecg + burst_len] = ma[st_noise:st_noise + burst_len]
        factor = 1.0
    elif noise_type == 'em':
        noise_arr = np.zeros_like(clean_sig)
        burst_len = 3 * 360  # 3 seconds
        st_noise = np.random.randint(0, len(em) - burst_len)
        st_ecg = np.random.randint(0, signal_length - burst_len)
        noise_arr[st_ecg:st_ecg + burst_len] = em[st_noise:st_noise + burst_len]
        factor = 1.2
    elif noise_type == 'bw':
        st_bw = np.random.randint(0, len(bw) - signal_length)
        noise_arr = bw[st_bw:st_bw + signal_length]
        factor = 1.0
    
    clean_z, noisy_z, clean_mm, noisy_mm, pred_mm = denoise_signal(clean_sig, noise_arr, factor)
    
    # Plot clean (z-scored for visual clarity)
    axes[row, 0].plot(time_sec, clean_mm, color='#27ae60', linewidth=0.7)
    axes[row, 0].set_ylabel(f'{noise_label}', fontweight='bold', fontsize=9)
    if row == 0:
        axes[row, 0].set_title('Clean Ground Truth', fontweight='bold')
    axes[row, 0].grid(True, alpha=0.15)
    axes[row, 0].set_xlim(0, 10)
    
    # Plot noisy
    axes[row, 1].plot(time_sec, noisy_mm, color='#c0392b', linewidth=0.7)
    if row == 0:
        axes[row, 1].set_title('Noisy Input', fontweight='bold')
    axes[row, 1].grid(True, alpha=0.15)
    axes[row, 1].set_xlim(0, 10)
    
    # Plot denoised
    axes[row, 2].plot(time_sec, pred_mm, color='#2980b9', linewidth=0.7)
    if row == 0:
        axes[row, 2].set_title('biGRU Denoised Output', fontweight='bold')
    axes[row, 2].grid(True, alpha=0.15)
    axes[row, 2].set_xlim(0, 10)
    
    # Compute quality metric for annotation
    rmse_val = math.sqrt(np.mean((clean_mm - pred_mm)**2))
    axes[row, 2].text(0.98, 0.95, f'RMSE: {rmse_val:.4f}', transform=axes[row, 2].transAxes,
                       ha='right', va='top', fontsize=8, 
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    
    if row == 2:
        for c in range(3):
            axes[row, c].set_xlabel('Time (s)')

fig.suptitle('Figure 1: ECG Denoising Examples — Clean 12-Lead Records with Injected Noise', 
             fontsize=13, fontweight='bold', y=1.02)
plt.savefig(f'{FIG_DIR}/fig1_denoising_examples.png', dpi=300)
plt.savefig(f'{FIG_DIR}/fig1_denoising_examples.pdf')
plt.close()
print("   Saved revised fig1_denoising_examples.png/pdf")

# ──────────────────────────────────────────────
# FIGURE 7 (REVISED): Detailed overlay with real QRS
# ──────────────────────────────────────────────
print("Generating revised Figure 7: Detailed overlay with clear QRS...")

sample_idx, lead_idx, _ = good_samples[0]  # Best sample
ecg_all = np.load(f'data/Y_test_all_leads/{sample_idx}.npy')
clean_sig = zsc(ecg_all[:, lead_idx])
signal_length = len(clean_sig)
time_sec = np.arange(signal_length) / SAMPLING_RATE

# Add MA noise burst
noise_arr = np.zeros_like(clean_sig)
burst_len = 4 * 360
st_noise = 50000
st_ecg = 1000
noise_arr[st_ecg:st_ecg + burst_len] = ma[st_noise:st_noise + burst_len]

clean_z, noisy_z, clean_mm, noisy_mm, pred_mm = denoise_signal(clean_sig, noise_arr, 1.2)
rmse_val = math.sqrt(np.mean((clean_mm - pred_mm)**2))

fig, axes = plt.subplots(4, 1, figsize=(13, 10), gridspec_kw={'hspace': 0.4})

# Panel 1: Clean
axes[0].plot(time_sec, clean_mm, color='#27ae60', linewidth=0.8)
axes[0].set_title('(a) Clean Ground Truth ECG', fontweight='bold')
axes[0].set_ylabel('Amplitude')
axes[0].grid(True, alpha=0.15)
axes[0].set_xlim(0, 10)

# Panel 2: Noisy
axes[1].plot(time_sec, noisy_mm, color='#c0392b', linewidth=0.8)
axes[1].set_title('(b) Noisy Input — Muscle Activation Artifact Injected', fontweight='bold')
axes[1].set_ylabel('Amplitude')
axes[1].grid(True, alpha=0.15)
axes[1].set_xlim(0, 10)
# Highlight noise region
axes[1].axvspan(st_ecg/360, (st_ecg + burst_len)/360, alpha=0.1, color='red', label='Noise region')
axes[1].legend(loc='upper right', fontsize=8)

# Panel 3: Denoised
axes[2].plot(time_sec, pred_mm, color='#2980b9', linewidth=0.8)
axes[2].set_title(f'(c) biGRU Denoised Output (RMSE = {rmse_val:.4f})', fontweight='bold')
axes[2].set_ylabel('Amplitude')
axes[2].grid(True, alpha=0.15)
axes[2].set_xlim(0, 10)

# Panel 4: Overlay
axes[3].plot(time_sec, clean_mm, color='#27ae60', linewidth=0.8, alpha=0.6, label='Ground Truth')
axes[3].plot(time_sec, pred_mm, color='#2980b9', linewidth=0.8, alpha=0.8, label='Denoised Output')
axes[3].set_title('(d) Overlay: Ground Truth vs. Denoised', fontweight='bold')
axes[3].set_ylabel('Amplitude')
axes[3].set_xlabel('Time (s)')
axes[3].legend(loc='upper right')
axes[3].grid(True, alpha=0.15)
axes[3].set_xlim(0, 10)

fig.suptitle('Figure 7: Detailed Denoising — Muscle Activation Artifact Removal', 
             fontsize=13, fontweight='bold', y=1.01)
plt.savefig(f'{FIG_DIR}/fig7_detailed_comparison.png', dpi=300)
plt.savefig(f'{FIG_DIR}/fig7_detailed_comparison.pdf')
plt.close()
print("  ✓ Saved revised fig7_detailed_comparison.png/pdf")

# ──────────────────────────────────────────────
# FIGURE 8 (NEW): Zoomed 3-second window comparison
# ──────────────────────────────────────────────
print("Generating Figure 8: Zoomed QRS comparison...")

fig, axes = plt.subplots(1, 3, figsize=(14, 3.5))

# Zoom into 3 seconds around a QRS complex
zoom_start = 2.0  # seconds
zoom_end = 5.0
mask = (time_sec >= zoom_start) & (time_sec <= zoom_end)

axes[0].plot(time_sec[mask], clean_mm[mask], color='#27ae60', linewidth=1.2)
axes[0].set_title('Clean', fontweight='bold')
axes[0].set_ylabel('Amplitude')
axes[0].set_xlabel('Time (s)')
axes[0].grid(True, alpha=0.15)

axes[1].plot(time_sec[mask], noisy_mm[mask], color='#c0392b', linewidth=1.2)
axes[1].set_title('Noisy (MA)', fontweight='bold')
axes[1].set_xlabel('Time (s)')
axes[1].grid(True, alpha=0.15)

axes[2].plot(time_sec[mask], clean_mm[mask], color='#27ae60', linewidth=1.0, alpha=0.5, label='Truth')
axes[2].plot(time_sec[mask], pred_mm[mask], color='#2980b9', linewidth=1.2, label='Denoised')
axes[2].set_title('Overlay (Zoomed)', fontweight='bold')
axes[2].set_xlabel('Time (s)')
axes[2].legend(fontsize=8)
axes[2].grid(True, alpha=0.15)

fig.suptitle('Figure 8: Zoomed QRS Complex — Before and After Denoising', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{FIG_DIR}/fig8_zoomed_qrs.png', dpi=300)
plt.savefig(f'{FIG_DIR}/fig8_zoomed_qrs.pdf')
plt.close()
print("  ✓ Saved fig8_zoomed_qrs.png/pdf")

# ──────────────────────────────────────────────
# FIGURE 9 (NEW): Mixed noise (all 3 types combined)
# ──────────────────────────────────────────────
print("Generating Figure 9: Mixed noise denoising...")

sample_idx, lead_idx, _ = good_samples[1]
ecg_all = np.load(f'data/Y_test_all_leads/{sample_idx}.npy')
clean_sig = zsc(ecg_all[:, lead_idx])
signal_length = len(clean_sig)
time_sec = np.arange(signal_length) / SAMPLING_RATE

# Add all three noise types
mixed_noise = np.zeros_like(clean_sig)
# MA burst
burst_len = 3 * 360
st = np.random.randint(0, len(ma) - burst_len)
mixed_noise[500:500+burst_len] += 0.8 * ma[st:st+burst_len]
# EM burst
st = np.random.randint(0, len(em) - burst_len)
mixed_noise[2000:2000+burst_len] += 1.2 * em[st:st+burst_len]
# BW continuous
st_bw = np.random.randint(0, len(bw) - signal_length)
mixed_noise += 0.8 * bw[st_bw:st_bw + signal_length]

clean_z, noisy_z, clean_mm, noisy_mm, pred_mm = denoise_signal(clean_sig, mixed_noise, 1.0)
rmse_val = math.sqrt(np.mean((clean_mm - pred_mm)**2))

fig, axes = plt.subplots(3, 1, figsize=(13, 7), gridspec_kw={'hspace': 0.4})

axes[0].plot(time_sec, clean_mm, color='#27ae60', linewidth=0.8)
axes[0].set_title('(a) Clean Ground Truth', fontweight='bold')
axes[0].set_ylabel('Amplitude')
axes[0].grid(True, alpha=0.15)

axes[1].plot(time_sec, noisy_mm, color='#c0392b', linewidth=0.8)
axes[1].set_title('(b) Mixed Noise: MA + EM + BW Combined', fontweight='bold')
axes[1].set_ylabel('Amplitude')
axes[1].grid(True, alpha=0.15)

axes[2].plot(time_sec, clean_mm, color='#27ae60', linewidth=0.8, alpha=0.5, label='Ground Truth')
axes[2].plot(time_sec, pred_mm, color='#2980b9', linewidth=0.8, label=f'Denoised (RMSE={rmse_val:.4f})')
axes[2].set_title('(c) Overlay: Ground Truth vs. Denoised', fontweight='bold')
axes[2].set_ylabel('Amplitude')
axes[2].set_xlabel('Time (s)')
axes[2].legend(loc='upper right')
axes[2].grid(True, alpha=0.15)

fig.suptitle('Figure 9: Denoising Under Combined MA + EM + BW Noise', fontsize=13, fontweight='bold', y=1.01)
plt.savefig(f'{FIG_DIR}/fig9_mixed_noise.png', dpi=300)
plt.savefig(f'{FIG_DIR}/fig9_mixed_noise.pdf')
plt.close()
print("  ✓ Saved fig9_mixed_noise.png/pdf")

print("\n" + "="*60)
print("ALL REVISED FIGURES GENERATED SUCCESSFULLY")
print("="*60)
print(f"\nFiles in {FIG_DIR}/:")
for f in sorted(os.listdir(FIG_DIR)):
    if not os.path.isdir(f'{FIG_DIR}/{f}') and (f.startswith('fig1') or f.startswith('fig7') or f.startswith('fig8') or f.startswith('fig9')):
        size = os.path.getsize(f'{FIG_DIR}/{f}')
        print(f"  {f:45s} ({size/1024:.1f} KB)")
