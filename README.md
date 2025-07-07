# Deployed Face Mask Detection (Interactive Demo)

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/HabiAshourichoshali/Capstone_binder/main?filepath=binder_notebook-interface.ipynb)
This repository provides an interactive demo of a deployed face mask detection model using classical machine learning. It includes a pre-trained model and a browser-based Jupyter notebook interface that requires no installation to run.
## 🎯 Project Overview

This project detects whether individuals are wearing face masks using a trained machine learning model. It processes image inputs and classifies them into:
- `face_mask`
- `face_no_mask`

Built using `scikit-learn`, `dill`, and utility functions, this version is optimized for deployment in the cloud for quick testing and demonstration.

---
## 📁 Contents

| File/Folder | Description |
|-------------|-------------|
| `binder_notebook-interface.ipynb` | Interactive notebook for uploading and testing images |
| `model_combined_datasets.dill` | Trained model using combined datasets |
| `pred_valid.dill`, `y_valid.dill` | Stored validation results |
| `requirements.txt` | Python dependencies |
| `util/` | Helper functions for plotting and prediction |
| `test/subimages/` | Example input images for testing |

---

## 🚀 How to Use

1. Click the **Binder badge** above.
2. Wait for the environment to launch (approx. 30–60 seconds).
3. Open and run the notebook to test images or review model results.

---
