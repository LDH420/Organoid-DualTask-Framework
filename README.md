# Organoid-DualTask-Framework
A comprehensive deep learning framework for standardized organoid detection and semantic segmentation across multiple species and modalities.
The baseline model code uses existing public code，evaluated under a standardized input resolution of 256×256 pixels
Environment：
 NVIDIA RTX A6000 GPU using CUDA 12.6
Scripts usage instructions：
 data processing (data_process.py), annotation format conversion and label mapping (instance_mask_tran.py), dataset loading (dataset.py), data validation, baseline model training (train_baseline.py), model inference (predict.py and inference.py), and organoid mask size distribution analysis (analyze_mask_size_distribution.py and size_distribution.py). 
Data availability:
  The data can be obtained from Zenodo： https://doi.org/10.5281/zenodo.18978542
