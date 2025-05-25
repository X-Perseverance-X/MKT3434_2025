# Machine Learning Course GUI

A comprehensive GUI application for machine learning experimentation and education, featuring classical ML algorithms, deep learning, GANs, and more.

## Features

- Classical Machine Learning algorithms (Regression, Classification)
- Deep Learning with customizable architectures
- Generative Adversarial Networks (GANs)
- Dimensionality Reduction techniques
- Feature Extraction
- Real-time training visualization
- Comprehensive logging system

## Setup Instructions

1. Create a virtual environment (recommended):

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the application:

```bash
python v18.py
```

## Usage Guide

### Data Loading

1. Select a dataset from the dropdown menu (Iris, Boston Housing, Breast Cancer, or Custom CSV)
2. Choose data preprocessing options (scaling, missing value handling)
3. Click "Load Data" to load the dataset

### Training Models

1. Navigate to the appropriate tab for your desired model type
2. Configure model parameters
3. Click "Train" to start training
4. Monitor progress in the status bar and training log
5. View results in the visualization panel

### GAN Training

1. Load a dataset
2. Go to the "GAN" tab
3. Configure GAN parameters:
   - Latent dimension
   - Number of epochs
   - Batch size
4. Click "Train GAN" to start training
5. Monitor training progress in the log window
6. View generated samples in the visualization panel

### Real-time Logging

- Training logs are saved to `ml_gui_YYYYMMDD_HHMMSS.log`
- View real-time training progress in the GUI
- Monitor model performance metrics

## Requirements

- Python 3.7+
- PyQt6
- TensorFlow 2.4+
- PyTorch 1.7+
- scikit-learn
- matplotlib
- numpy
- pandas

## Notes

- For GAN training, ensure your dataset is properly normalized
- The application supports both CPU and GPU training
- Training progress can be monitored in real-time through the GUI
- Generated samples are visualized using PCA for high-dimensional data

## Troubleshooting

If you encounter any issues:

1. Ensure all dependencies are correctly installed
2. Check the log file for detailed error messages
3. Verify your dataset format when using custom data
4. For GPU acceleration, ensure CUDA is properly installed
