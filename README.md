# 💎 Image Segmentation 🔬  
*Thanks to Howsam AI Academy*

---

## 🔴 Environment Setup  
1. **Connect Colab to your local Jupyter server** (if using Colab + local machine):  
   ```bash
   jupyter notebook \
     --NotebookApp.allow_origin='https://colab.research.google.com' \
     --port=4000 \
     --NotebookApp.port_retries=0
   ```
2. **Install required libraries** (run each in its own cell or prefix with `!` in Colab):  
   ```bash
   pip install torchmetrics
   pip install portalocker
   pip install wandb
   ```
3. **Restart the runtime** after installing new packages to ensure they load correctly.

---

## 🔴 Import Libraries  
- **System & Utilities**: `os`, `sys`, `time`, `random`, `glob`, `tqdm`, `typing`  
- **Data & Arrays**: `numpy as np`, `pandas as pd`  
- **Visualization**: `matplotlib.pyplot as plt`, `PIL.Image`, `cv2`  
- **PyTorch Core**:  
  - `torch`, `torch.nn as nn`, `torch.nn.functional as F`  
  - `torch.optim`  
  - `torch.utils.data.Dataset`, `DataLoader`, `random_split`, `ConcatDataset`  
- **Torchvision**:  
  - `transforms`, `tv_tensors`, `VisionDataset`  
- **Segmentation Models**: `segmentation_models_pytorch as smp`  
- **Metrics**: `torchmetrics` (`MeanMetric`, `Dice`)

---

## 🔴 Change Output Font Size (Optional)  
- Adjust matplotlib font sizes globally for readability:  
  ```python
  matplotlib.rcParams.update({'font.size': 12})
  ```

---

## 🔴 Load the Dataset  
1. **Load & display a sample image**  
2. **Load & display its corresponding segmentation mask**  
   - Convert mask IDs to color overlays  
   - Visualize image-mask pairs side by side  

---

## 🔴 Dataset Preparation  
1. **Load metadata CSV** containing `image_id`, `mask_id`, `class_id`, etc.  
2. **Clean the DataFrame**  
   - Drop missing values (`NaN`)  
   - Identify & remove duplicate rows  
3. **Reshape & pivot**  
   - Create a pivot table so each row = one image, columns = one-hot or list of mask IDs  
4. **Merge file paths**  
   - Glob filesystem for image and mask directories  
   - Build a DataFrame mapping `image_id` → `file_path`  
   - Merge with metadata DataFrame  
5. **Feature augmentation**  
   - Add derived columns (e.g., image height/width, number of objects per image)  
6. **Split into Train / Validation / Test**  
   - Use `random_split` or stratified split based on class distribution  
   - Typical splits: 70% train, 15% val, 15% test

---

## 🔴 Exploratory Data Analysis (EDA)  
- **Show random image + mask overlays** to inspect quality  
- **Animate a time-series** (if dataset has temporal dimension)  
- **Plot distributions**:  
  - Number of samples per class  
  - Number of objects per image  
  - Image dimensions (height × width)  
- **Estimate RAM footprint** of entire dataset in-memory

---

## 🔴 Custom Dataset Class  
1. **Version 1**: Inherit from `torch.utils.data.Dataset`  
   - `__init__`: accept DataFrame, transforms  
   - `__len__`: return dataset size  
   - `__getitem__`: load image & mask, apply preprocessing  
2. **Version 2**: Leverage `torchvision.datasets.VisionDataset` or `tv_tensors` for more efficiency  
3. **Memory-saving option**:  
   - Load masks on-the-fly  
   - Cache frequently used items  
   - Use smaller data types (`uint8` for masks)

---

## 🔴 DataLoader  
- Instantiate `DataLoader` for train/val/test sets  
  - Set `batch_size`, `shuffle`, `num_workers`, `pin_memory`  
- Optionally wrap in `ConcatDataset` for combining multiple subsets

---

## 🔴 Finding Hyperparameters  
1. **Step 1**: Compute initial loss on an untrained model over a few batches  
2. **Step 2**: Overfit a small subset to verify model can learn (sanity check)  
3. **Step 3**: Grid-search learning rates:  
   - Train for a few epochs at different `lr` values  
   - Plot loss curves to pick optimal `lr`  
4. **Step 4**: Grid-search weight decay around best `lr`  
5. **Step 5**: Retrain for more epochs using best `lr` & `weight_decay`

---

## 🔴 Main Training Loop  
- **Epoch loop**: for each epoch:  
  1. **Training phase**  
     - Set `model.train()`  
     - Loop over train `DataLoader`  
     - Forward pass → compute loss → backward → optimizer step  
     - Track metrics (loss, pixel accuracy, Dice score)  
  2. **Validation phase**  
     - Set `model.eval()`, disable gradients  
     - Loop over val `DataLoader`  
     - Compute loss & metrics  
     - Save best model checkpoints based on validation Dice

---

## 🔴 Visualization & Metrics  
1. **Plot training & validation loss curves** over epochs  
2. **Plot Dice score** (global) over epochs  
3. **Compute class-wise Dice scores**  
   - Evaluate per-class performance  
   - Bar chart of Dice per class  
4. **Confusion Matrix**  
   - Compute pixel-wise confusion matrix across classes  
   - Visualize as heatmap  

---

## 🔴 Tips & Next Steps  
- Experiment with different backbone architectures in `smp` (e.g., ResNet34, EfficientNet)  
- Use advanced augmentations (`albumentations`) for robustness  
- Integrate with **Weights & Biases** (`wandb`) for experiment tracking  
- Deploy your best model using TorchScript or ONNX for production inference  

---
