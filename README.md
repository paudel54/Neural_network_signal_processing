# ECG Signal Noise Removal with Bidirectional GRU

This repository contains a PyTorch implementation of a gated recurrent unit (GRU) model for the task of removing noise from electrocardiogram (ECG) signals.

## Abstract
As the popularity of wearables continues to scale, a substantial portion of the population now has access to (self-)monitoring of cardiovascular activity. In particular, the use of ECG wearables is growing in the realm of occupational health assessment, but one common issue that is encountered is the presence of noise which hinders the reliability of the acquired data.

In this work, we propose an ECG denoiser based on bidirectional Gated Recurrent Units (biGRU). This model was trained on noisy ECG samples that were created by adding noise from the MIT-BIH Noise Stress Test database to clean ECG samples from the PTB-XL database. The model was trained and tested on data corrupted with the three most common sources of noise:
* Muscle Activation (MA)
* Electrode Motion artifacts (EM) 
* Baseline Wander (BW)

After training, the model is able to beautifully reconstruct severely corrupted signals, achieving overall average **Root-Mean-Square Error (RMSE) of 0.029** and improving the **Signal-to-Noise Ratio (SNR) by ~12.5 dB** on average. Its compact size (~104 KB) allows real-time inference on edge devices.

## Performance & Visual Results

The model has been rigorously evaluated on a held-out test set of 3,267 signals across various noise intensities (down to 0 dB SNR).

### 1. Denoising Examples by Artifact Type
The biGRU successfully handles various types of noise while preserving the true QRS morphology.
![Denoising Examples](figures_/fig1_denoising_examples.png)

### 2. High-Fidelity Artifact Removal
Even when a localized burst of muscle activation completely obscures the QRS complexes, the denoiser reconstructs the baseline and peaks accurately.
![Detailed Comparison](figures_/fig7_detailed_comparison.png)

### 3. Model Architecture
A compact Bidirectional GRU encoder-decoder structure.
![Model Architecture](figures_/fig6_model_architecture.png)

### 4. Overall Evaluation Metrics
![Metric Distributions](figures_/fig4_metric_distributions.png)
*(Evaluations computed over the full 3,267 test set samples).*

## Features
* GRU-based model for time series analysis
* Bidirectional GRU for better context capturing
* Comprehensive data preprocessing and synthetic noise injection pipelines
* Pre-trained model weights are provided for immediate inference
* Full mathematical evaluation suite (SNR, RMSE, PRD)

## Installation
```bash
git clone https://github.com/paudel54/Neural_network_signal_processing.git
cd ECG_Denoiser
pip install -r requirements.txt
```

## Databases
For the development of the ECG noise removal model, we created a dataset with noisy and clean versions of ECG signals using two publicly available datasets from PhysioNet:
* **PTB-XL**: https://physionet.org/content/ptb-xl/1.0.3/
* **MIT-BIH Noise Stress Test**: https://physionet.org/content/nstdb/1.0.0/

## Repository Structure
* `tools/`: package including the core functions for evaluation and metrics computation.
* `create_dataset/`: Scripts to create train/val/test splits and inject MIT-BIH noise into PTB-XL records.
  * `1_get_save_data_from_db.py`: Load data from databases
  * `2_train_val_test_split.py`: Split and preprocess the clean ground truth
  * `3_create_noisy_data2_360hz.py`: Synthetically inject noise
* `gru_denoiser.py` and `utils_denoiser.py`: PyTorch model implementation.
* `evaluation.py` and `generate_figures.py`: Generates evaluation metrics and figures.


## References

1. U. Satija, B. Ramkumar, and M. S. Manikandan, "A review of signal processing techniques for electrocardiogram signal quality assessment," *IEEE Reviews in Biomedical Engineering*, vol. 11, pp. 36–52, 2018.
2. H. Limaye and V. Deshmukh, "Ecg noise sources and various noise removal techniques: A survey," *International Journal of Application or Innovation in Engineering & Management*, vol. 5, no. 2, pp. 86–92, 2016.
3. S. Chatterjee, R. S. Thakur, R. N. Yadav, L. Gupta, and D. K. Raghuvanshi, "Review of noise removal techniques in ecg signals," *IET Signal Processing*, vol. 14, no. 9, pp. 569–590, 2020.
4. H.-T. Chiang, Y.-Y. Hsieh, S.-W. Fu, K.-H. Hung, Y. Tsao, and S.-Y. Chien, "Noise reduction in ecg signals using fully convolutional denoising autoencoders," *Ieee Access*, vol. 7, pp. 60 806–60 813, 2019.
5. K. Antczak, "Deep recurrent neural networks for ecg signal denoising," *arXiv preprint arXiv:1807.11551*, 2018.
6. P. Wagner, N. Strodthoff, R.-D. Bousseljot, D. Kreiseler, F. I. Lunze, W. Samek, and T. Schaeffter, "Ptb-xl, a large publicly available electrocardiography dataset," *Scientific data*, vol. 7, no. 1, p. 154, 2020.
7. G. B. Moody, W. Muldrow, and R. G. Mark, "A noise stress test for arrhythmia detectors," *Computers in cardiology*, vol. 11, no. 3, pp. 381–384, 1984.
