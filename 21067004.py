import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import sys
import numpy as np
import pandas as pd
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                           QHBoxLayout, QTabWidget, QPushButton, QLabel, 
                           QComboBox, QFileDialog, QSpinBox, QDoubleSpinBox,
                           QGroupBox, QScrollArea, QTextEdit, QStatusBar,
                           QProgressBar, QCheckBox, QGridLayout, QMessageBox,
                           QDialog, QLineEdit, QListWidget, QListWidgetItem)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from sklearn import datasets, preprocessing, model_selection
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, mean_squared_error, confusion_matrix
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD
from sklearn.manifold import TSNE
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics import silhouette_score
from sklearn.metrics import pairwise_distances
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
import logging
from datetime import datetime
import threading
import queue

# GAN: Generator ve Discriminator sınıfları
class Generator(tf.keras.Model):
    def __init__(self, latent_dim, output_dim):
        super().__init__()
        self.model = tf.keras.Sequential([
            layers.Input(shape=(latent_dim,)),
            layers.Dense(256, use_bias=False),
            layers.BatchNormalization(),
            layers.LeakyReLU(),
            layers.Dense(512),
            layers.BatchNormalization(),
            layers.LeakyReLU(),
            layers.Dense(1024),
            layers.BatchNormalization(),
            layers.LeakyReLU(),
            layers.Dense(output_dim, activation='tanh')
        ])

    def call(self, inputs, training=False):
        return self.model(inputs, training=training)

class Discriminator(tf.keras.Model):
    def __init__(self, input_dim):
        super().__init__()
        self.model = tf.keras.Sequential([
            layers.Input(shape=(input_dim,)),
            layers.Dense(512),
            layers.LeakyReLU(),
            layers.Dropout(0.3),
            layers.Dense(256),
            layers.LeakyReLU(),
            layers.Dropout(0.3),
            layers.Dense(1, activation='sigmoid')
        ])

    def call(self, inputs, training=False):
        return self.model(inputs, training=training)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'ml_gui_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Custom logging handler for GUI
class GUIHandler(logging.Handler):
    def __init__(self, text_widget):
        super().__init__()
        self.text_widget = text_widget
        self.queue = queue.Queue()
        self.thread = threading.Thread(target=self._process_queue, daemon=True)
        self.thread.start()

    def emit(self, record):
        self.queue.put(record)

    def _process_queue(self):
        while True:
            record = self.queue.get()
            msg = self.format(record)
            self.text_widget.append(msg)
            self.text_widget.verticalScrollBar().setValue(
                self.text_widget.verticalScrollBar().maximum()
            )

class MLCourseGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Machine Learning Course GUI")
        self.setGeometry(100, 100, 1400, 800)
        self.main_widget = QWidget()
        self.setCentralWidget(self.main_widget)
        self.layout = QVBoxLayout(self.main_widget)
        self.data = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.current_model = None
        self.current_loss = "mse"
        self.class_weights_dict = None
        self.layer_config = []
        self.create_data_section()
        self.create_tabs()
        self.create_visualization()
        self.create_status_bar()
        self.setWindowFlags(self.windowFlags() & ~Qt.WindowType.WindowStaysOnTopHint)

    def create_gan_tab(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # GAN Configuration

    def start_gan_training(self):
        """Start GAN training"""
        if self.X_train is None:
            self.show_error("Please load a dataset first")
            return
        try:
            input_dim = self.X_train.shape[1]
            latent_dim = self.latent_dim_spin.value()
            self.generator = Generator(latent_dim, input_dim)
            self.discriminator = Discriminator(input_dim)
            dataset = tf.data.Dataset.from_tensor_slices(self.X_train)
            self.gan_thread = GANTrainingThread(
                self.generator,
                self.discriminator,
                dataset,
                latent_dim,
                self.gan_epochs_spin.value(),
                self.gan_batch_size_spin.value()
            )
            self.gan_thread.progress_updated.connect(self.update_gan_progress)
            self.gan_thread.training_log.connect(self.update_gan_log)
            self.gan_thread.training_finished.connect(self.gan_training_finished)
            self.train_gan_btn.setEnabled(False)
            self.stop_gan_btn.setEnabled(True)
            self.gan_thread.start()
        except Exception as e:
            self.show_error(f"Error starting GAN training: {str(e)}")
            logger.error(f"GAN training error: {str(e)}")

    def stop_gan_training(self):
        """Stop GAN training"""
        if hasattr(self, 'gan_thread'):
            self.gan_thread.stop()
            self.gan_thread.wait()
            self.gan_training_finished()

    def update_gan_progress(self, value):
        """Update progress bar during GAN training"""
        if hasattr(self, 'progress_bar'):
            self.progress_bar.setValue(value)

    def update_gan_log(self, message):
        """Update training log during GAN training"""
        if hasattr(self, 'gan_log_text'):
            self.gan_log_text.append(message)
        logger.info(message)

    def gan_training_finished(self):
        """Handle GAN training completion"""
        self.train_gan_btn.setEnabled(True)
        self.stop_gan_btn.setEnabled(False)
        if hasattr(self, 'progress_bar'):
            self.progress_bar.setValue(0)
        self.generate_gan_samples()

    def generate_gan_samples(self):
        """Generate and display samples from trained GAN"""
        if not hasattr(self, 'generator'):
            return
        try:
            noise = tf.random.normal([16, self.latent_dim_spin.value()])
            samples = self.generator(noise, training=False).numpy()
            self.gan_figure.clear()
            ax = self.gan_figure.add_subplot(111)
            if samples.shape[1] > 2:
                pca = PCA(n_components=2)
                samples_2d = pca.fit_transform(samples)
                ax.scatter(samples_2d[:, 0], samples_2d[:, 1])
                ax.set_title("Generated Samples (PCA projection)")
            else:
                ax.scatter(samples[:, 0], samples[:, 1])
                ax.set_title("Generated Samples")
            self.gan_canvas.draw()
        except Exception as e:
            self.show_error(f"Error generating samples: {str(e)}")
            logger.error(f"Sample generation error: {str(e)}")

        # ... rest of your code ...
        
        # Initialize main widget and layout
        self.main_widget = QWidget()
        self.setCentralWidget(self.main_widget)
        self.layout = QVBoxLayout(self.main_widget)
        
        # Initialize data containers
        self.data = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.current_model = None
        self.current_loss = "mse"  # Default loss
        self.class_weights_dict = None
        
        # Neural network configuration
        self.layer_config = []
        
        # Create components
        self.create_data_section()
        self.create_tabs()
        self.create_visualization()
        self.create_status_bar()
        
        # Set window to stay on top temporarily when updating plots
        self.setWindowFlags(self.windowFlags() & ~Qt.WindowType.WindowStaysOnTopHint)

    def create_data_section(self):
        """Create the data management section"""
        data_group = QGroupBox("Data Management")
        data_group.setMaximumHeight(250)  # Reduced vertical height
        layout = QHBoxLayout()
        
        # Left section for dataset selection and loading
        left_section = QVBoxLayout()
        left_section.setSpacing(5)  # Reduce vertical spacing
        
        # Dataset selection in a more compact layout
        dataset_layout = QHBoxLayout()
        dataset_layout.setSpacing(5)  # Reduce horizontal spacing
        dataset_label = QLabel("Dataset:")
        dataset_label.setFixedWidth(50)  # Fixed width for label
        self.dataset_combo = QComboBox()
        self.dataset_combo.addItems(["Iris", "Boston Housing", "Breast Cancer", "Custom CSV"])
        dataset_layout.addWidget(dataset_label)
        dataset_layout.addWidget(self.dataset_combo)
        
        # Load button
        load_btn = QPushButton("Load Data")
        load_btn.setFixedHeight(25)  # Smaller height
        load_btn.clicked.connect(self.load_dataset)

        # --- MISSING DATA HANDLING ---
        missing_group = QGroupBox("Missing Data Handling")
        missing_layout = QVBoxLayout()
        self.missing_combo = QComboBox()
        self.missing_combo.addItems([
            "No Action",
            "Mean Imputation"
        ])
        missing_layout.addWidget(QLabel("Handle missing values:"))
        missing_layout.addWidget(self.missing_combo)
        missing_group.setLayout(missing_layout)
        
        # Scaling options in a compact layout
        scaling_layout = QHBoxLayout()
        scaling_layout.setSpacing(5)
        scaling_label = QLabel("Scaling:")
        scaling_label.setFixedWidth(50)
        self.scaling_combo = QComboBox()
        self.scaling_combo.addItems(["No Scaling", "Standard Scaling", "Min-Max Scaling", "Robust Scaling"])
        scaling_layout.addWidget(scaling_label)
        scaling_layout.addWidget(self.scaling_combo)
        
        # Split ratio and k-fold options in a compact layout
        split_section = QHBoxLayout()
        split_section.setSpacing(5)
        # Split ratio ComboBox
        self.split_combo = QComboBox()
        self.split_combo.addItems([
            "80-20 (Train-Test)",
            "70-15-15 (Train-Val-Test)",
            "60-20-20 (Train-Val-Test)"
        ])
        self.split_combo.setCurrentIndex(0)
        split_section.addWidget(QLabel("Split Ratio:"))
        split_section.addWidget(self.split_combo)
        # K-Fold SpinBox
        self.kfold_spin = QSpinBox()
        self.kfold_spin.setRange(2, 20)
        self.kfold_spin.setValue(5)
        split_section.addWidget(QLabel("K-Fold:"))
        split_section.addWidget(self.kfold_spin)
        
        # Add to left section
        left_section.addLayout(dataset_layout)
        left_section.addWidget(load_btn)
        left_section.addWidget(missing_group)
        left_section.addLayout(scaling_layout)
        left_section.addLayout(split_section)
        
        # Right section for loss function selection
        right_section = QVBoxLayout()
        right_section.setSpacing(5)
        
        # Classification loss options in a more compact form
        class_loss_group = QGroupBox("Classification Loss")
        class_loss_layout = QVBoxLayout()
        class_loss_layout.setSpacing(5)
        
        self.class_loss_combo = QComboBox()
        self.class_loss_combo.addItems(["Cross Entropy", "Binary Cross Entropy", "Hinge Loss"])
        
        # Class weights in a compact layout
        weight_layout = QHBoxLayout()
        weight_layout.setSpacing(5)
        weight_label = QLabel("Weights:")
        weight_label.setFixedWidth(50)
        self.class_weights = QComboBox()
        self.class_weights.addItems(["None", "Balanced", "Custom"])
        weight_layout.addWidget(weight_label)
        weight_layout.addWidget(self.class_weights)
        
        class_loss_layout.addWidget(self.class_loss_combo)
        class_loss_layout.addLayout(weight_layout)
        class_loss_group.setLayout(class_loss_layout)
        
        # Regression loss options in a compact form
        reg_loss_group = QGroupBox("Regression Loss")
        reg_loss_layout = QVBoxLayout()
        reg_loss_layout.setSpacing(5)
        
        self.reg_loss_combo = QComboBox()
        self.reg_loss_combo.addItems(["Mean Squared Error (MSE)", "Mean Absolute Error (MAE)", "Huber Loss"])
        
        # Huber loss delta in a compact layout
        huber_layout = QHBoxLayout()
        huber_layout.setSpacing(5)
        huber_label = QLabel("Huber δ:")
        huber_label.setFixedWidth(50)
        self.huber_delta = QDoubleSpinBox()
        self.huber_delta.setRange(0.1, 10.0)
        self.huber_delta.setValue(1.0)
        self.huber_delta.setSingleStep(0.1)
        huber_layout.addWidget(huber_label)
        huber_layout.addWidget(self.huber_delta)
        
        reg_loss_layout.addWidget(self.reg_loss_combo)
        reg_loss_layout.addLayout(huber_layout)
        reg_loss_group.setLayout(reg_loss_layout)
        
        # Apply button
        apply_btn = QPushButton("Apply Loss Settings")
        apply_btn.setFixedHeight(25)
        apply_btn.clicked.connect(self.apply_loss_settings)
        
        # Add to right section
        right_section.addWidget(class_loss_group)
        right_section.addWidget(reg_loss_group)
        right_section.addWidget(apply_btn)
        
        # Add sections to main layout with stretch factors
        layout.addLayout(left_section, stretch=1)
        layout.addLayout(right_section, stretch=1)
        
        data_group.setLayout(layout)
        self.layout.addWidget(data_group)

    def select_target_column(self, columns):
        """Dialog to select target column"""
        dialog = QDialog(self)
        dialog.setWindowTitle("Select Target Column")
        layout = QVBoxLayout(dialog)
        
        combo = QComboBox()
        combo.addItems(columns)
        layout.addWidget(combo)
        
        btn = QPushButton("Select")
        btn.clicked.connect(dialog.accept)
        layout.addWidget(btn)
        
        if dialog.exec() == QDialog.DialogCode.Accepted:
            return combo.currentText()
        return None
    
    def create_visualization(self):
        """Create the visualization section"""
        viz_group = QGroupBox("Visualization")
        
        layout = QHBoxLayout()  # Changed back to HBox for side-by-side plots
        
        # Left side: Raw data plot with axis selection
        raw_data_group = QGroupBox("Raw Data")
        raw_data_layout = QVBoxLayout()
        raw_data_layout.setSpacing(5)
        
        # Axis selection in a compact form
        axis_layout = QHBoxLayout()
        axis_layout.setSpacing(5)
        
        # Add axis labels and combos with reduced size
        x_label = QLabel("X:")
        x_label.setFixedWidth(10)
        y_label = QLabel("Y:")
        y_label.setFixedWidth(10)
        z_label = QLabel("Z:")
        z_label.setFixedWidth(10)
        
        # Set fixed sizes for combo boxes (even smaller)
        self.x_axis_combo = QComboBox()
        self.x_axis_combo.setFixedWidth(100)
        self.y_axis_combo = QComboBox()
        self.y_axis_combo.setFixedWidth(100)
        self.z_axis_combo = QComboBox()
        self.z_axis_combo.setFixedWidth(100)
        
        # Set fixed height for combo boxes
        self.x_axis_combo.setFixedHeight(22)
        self.y_axis_combo.setFixedHeight(22)
        self.z_axis_combo.setFixedHeight(22)
        
        axis_layout.addWidget(x_label)
        axis_layout.addWidget(self.x_axis_combo)
        axis_layout.addWidget(y_label)
        axis_layout.addWidget(self.y_axis_combo)
        axis_layout.addWidget(z_label)
        axis_layout.addWidget(self.z_axis_combo)
        axis_layout.addStretch()
        
        raw_data_layout.addLayout(axis_layout)
        
        # Plot with smaller figure size
        self.figure = plt.figure(figsize=(6, 4))
        self.canvas = FigureCanvas(self.figure)
        raw_data_layout.addWidget(self.canvas)
        raw_data_group.setLayout(raw_data_layout)
        
        # Right side: Predictions plot
        predictions_group = QGroupBox("Model Predictions")
        predictions_layout = QVBoxLayout()
        self.prediction_figure = plt.figure(figsize=(7, 5))  # Increased size
        self.prediction_canvas = FigureCanvas(self.prediction_figure)
        predictions_layout.addWidget(self.prediction_canvas)
        predictions_group.setLayout(predictions_layout)
        
        # Metrics display below the plots
        metrics_group = QGroupBox("Metrics")
        metrics_layout = QVBoxLayout()  # Back to vertical layout
        
        # Single text box with reduced height
        self.metrics_text = QTextEdit()
        self.metrics_text.setReadOnly(True)
        self.metrics_text.setMaximumHeight(80)  # Reduced height for metrics
        metrics_layout.addWidget(self.metrics_text)
        metrics_group.setLayout(metrics_layout)
        metrics_group.setMinimumHeight(200)
        
        # Create a vertical layout for the right side
        right_layout = QVBoxLayout()
        right_layout.addWidget(predictions_group, stretch=3)  # Give more space to predictions
        right_layout.addWidget(metrics_group, stretch=1)  # Less space for metrics
        
        # Add components to main layout with proper stretching
        layout.addWidget(raw_data_group, stretch=1)
        layout.addLayout(right_layout, stretch=1)
        
        viz_group.setLayout(layout)
        viz_group.setMinimumHeight(500)  # Increased overall height
        self.layout.addWidget(viz_group)
        
        # Connect axis selection signals
        self.x_axis_combo.currentIndexChanged.connect(self.update_plot)
        self.y_axis_combo.currentIndexChanged.connect(self.update_plot)
        self.z_axis_combo.currentIndexChanged.connect(self.update_plot)

    def update_axis_options(self):
        """Update axis selection combo boxes with available features"""
        if self.X_train is not None:
            features = []
            if isinstance(self.X_train, pd.DataFrame):
                features = self.X_train.columns.tolist()
            else:
                features = [f"Feature {i+1}" for i in range(self.X_train.shape[1])]
            
            self.x_axis_combo.clear()
            self.y_axis_combo.clear()
            self.z_axis_combo.clear()
            
            self.x_axis_combo.addItems(features)
            self.y_axis_combo.addItems(features)
            self.z_axis_combo.addItems(features + ["Target"])
            
            # Set default selections
            self.x_axis_combo.setCurrentIndex(0)
            if len(features) > 1:
                self.y_axis_combo.setCurrentIndex(1)
            if len(features) > 2:
                self.z_axis_combo.setCurrentIndex(2)
            else:
                self.z_axis_combo.setCurrentIndex(len(features))  # Select "Target"

    def plot_raw_data(self):
        """Plot the raw training data with histogram"""
        try:
            if self.X_train is None or self.y_train is None:
                return
                
            self.figure.clear()
            
            # Create subplot layout with custom size ratio and spacing
            gs = self.figure.add_gridspec(1, 4, wspace=0.3)  # Increased spacing between subplots
            ax_3d = self.figure.add_subplot(gs[0, :3], projection='3d')  # Use first 3 columns for 3D plot
            
            # Get selected features
            x_idx = self.x_axis_combo.currentIndex()
            y_idx = self.y_axis_combo.currentIndex()
            z_idx = self.z_axis_combo.currentIndex()
            
            # Extract data based on selection
            if isinstance(self.X_train, pd.DataFrame):
                x_data = self.X_train.iloc[:, x_idx].values
                y_data = self.X_train.iloc[:, y_idx].values
            else:
                x_data = self.X_train[:, x_idx]
                y_data = self.X_train[:, y_idx]
            
            # Handle Z-axis data
            if z_idx < self.X_train.shape[1]:
                z_data = self.X_train[:, z_idx] if not isinstance(self.X_train, pd.DataFrame) else self.X_train.iloc[:, z_idx].values
            else:
                z_data = self.y_train
            
            # Create 3D scatter plot with adjusted position
            scatter = ax_3d.scatter(x_data, y_data, z_data, c=z_data, cmap='viridis')
            
            # Set labels
            ax_3d.set_xlabel(self.x_axis_combo.currentText())
            ax_3d.set_ylabel(self.y_axis_combo.currentText())
            ax_3d.set_zlabel(self.z_axis_combo.currentText())
            
            # Add colorbar with adjusted position
            self.figure.colorbar(scatter, ax=ax_3d, pad=0.1)
            
            # Add histogram if available
            if self.X_train.shape[1] > 3:
                ax_hist = self.figure.add_subplot(gs[0, 3])
                remaining_features = self.X_train[:, 3:].mean(axis=1)
                
                # Create histogram with adjusted position
                n, bins, patches = ax_hist.hist(remaining_features, bins=30, 
                                              orientation='horizontal')
                
                # Color the histogram bars
                fracs = n / n.max()
                norm = plt.Normalize(fracs.min(), fracs.max())
                for thisfrac, thispatch in zip(fracs, patches):
                    color = plt.cm.viridis(norm(thisfrac))
                    thispatch.set_facecolor(color)
                
                ax_hist.set_xlabel('Count')
                ax_hist.set_ylabel('Additional Features\n(Mean)')
            
            # Adjust layout to center plots
            self.figure.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
            
            # Add extra space around plots
            self.figure.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
            
            self.canvas.draw()
            
        except Exception as e:
            self.show_error(f"Error plotting raw data: {str(e)}")

    def plot_predictions(self):
        """Plot model predictions"""
        try:
            if self.current_model is None:
                return
                
            self.prediction_figure.clear()
            
            # Create main plot with adjusted position
            ax = self.prediction_figure.add_subplot(111, projection='3d')
            
            # Get current axis selections
            x_idx = self.x_axis_combo.currentIndex()
            y_idx = self.y_axis_combo.currentIndex()
            
            # Get the data
            if isinstance(self.X_test, pd.DataFrame):
                x_data = self.X_test.iloc[:, x_idx].values
                y_data = self.X_test.iloc[:, y_idx].values
            else:
                x_data = self.X_test[:, x_idx]
                y_data = self.X_test[:, y_idx]
            
            # Get predictions
            z_pred = self.current_model.predict(self.X_test)
            
            # Create scatter plot with predictions
            scatter = ax.scatter(x_data, y_data, z_pred, 
                               c=z_pred, cmap='viridis', 
                               label='Predictions')
            
            # Add actual values if available
            if self.y_test is not None:
                ax.scatter(x_data, y_data, self.y_test, 
                          c='red', marker='x', 
                          label='Actual Values')
            
            # Set labels
            ax.set_xlabel(self.x_axis_combo.currentText())
            ax.set_ylabel(self.y_axis_combo.currentText())
            ax.set_zlabel('Predictions')
            
            # Add legend
            ax.legend()
            
            # Add colorbar with adjusted position
            self.prediction_figure.colorbar(scatter, ax=ax, pad=0.1)
            
            # Adjust layout to center plot
            self.prediction_figure.tight_layout()
            
            # Add extra space around plot
            self.prediction_figure.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
            
            self.prediction_canvas.draw()
            
        except Exception as e:
            self.show_error(f"Error plotting predictions: {str(e)}")
    
    def update_plot(self):
        """Update plots when axis selection changes"""
        try:
            if self.X_train is not None:
                self.plot_raw_data()
                if self.current_model is not None:
                    self.plot_predictions()
        except Exception as e:
            self.show_error(f"Error updating plots: {str(e)}")
    
    def train_model(self, name, param_widgets):
        """Train the selected model with parameters"""
        try:
            if self.X_train is None or self.y_train is None:
                self.show_error("Please load a dataset first")
                return
                
            # Get current loss function
            loss_func = self.current_loss
            
            # Handle target variable shape for classification
            if loss_func in ["categorical_crossentropy", "binary_crossentropy", "hinge"]:
                y_train = np.argmax(self.y_train, axis=1) if len(self.y_train.shape) > 1 else self.y_train
                y_test = np.argmax(self.y_test, axis=1) if len(self.y_test.shape) > 1 else self.y_test
            else:
                y_train = self.y_train.ravel() if len(self.y_train.shape) > 1 else self.y_train
                y_test = self.y_test.ravel() if len(self.y_test.shape) > 1 else self.y_test
            
            # K-Fold cross-validation
            k = self.kfold_spin.value() if hasattr(self, 'kfold_spin') else 5
            from sklearn.model_selection import cross_val_score
            from sklearn.metrics import make_scorer, mean_squared_error
            model = None
            if name == "Linear Regression":
                model = LinearRegression(
                    fit_intercept=param_widgets['fit_intercept'].isChecked(),
                    normalize=param_widgets['normalize'].isChecked()
                )
            elif name == "Logistic Regression":
                max_iter = 1000
                model = LogisticRegression(
                    max_iter=max_iter,
                    class_weight=self.class_weights_dict
                )
            elif name == "Naive Bayes":
                var_smoothing = param_widgets['var_smoothing'].value()
                priors_text = param_widgets['priors'].text() if 'priors' in param_widgets else ""
                priors = None
                if priors_text:
                    try:
                        priors = [float(x) for x in priors_text.strip('[]').split(',')]
                    except Exception:
                        priors = None
                model = GaussianNB(var_smoothing=var_smoothing, priors=priors)
            elif name == "Support Vector Machine":
                model = SVC(
                    C=param_widgets['C'].value(),
                    kernel=param_widgets['kernel'].currentText(),
                    degree=param_widgets['degree'].value(),
                    class_weight=self.class_weights_dict
                )
            elif name == "Decision Tree":
                model = DecisionTreeClassifier(
                    max_depth=param_widgets['max_depth'].value(),
                    min_samples_split=param_widgets['min_samples_split'].value(),
                    criterion=param_widgets['criterion'].currentText(),
                    class_weight=self.class_weights_dict
                )
            elif name == "Random Forest":
                model = RandomForestClassifier(
                    n_estimators=param_widgets['n_estimators'].value(),
                    max_depth=param_widgets['max_depth'].value(),
                    min_samples_split=param_widgets['min_samples_split'].value(),
                    class_weight=self.class_weights_dict
                )
            elif name == "K-Nearest Neighbors":
                model = KNeighborsClassifier(
                    n_neighbors=param_widgets['n_neighbors'].value(),
                    weights=param_widgets['weights'].currentText(),
                    metric=param_widgets['metric'].currentText()
                )

            if model is None:
                self.show_error("Model not implemented")
                return

            # Cross-validation metrics (only if k > 1 and enough samples)
            if k > 1 and self.X_train.shape[0] >= k:
                accuracy = cross_val_score(model, self.X_train, y_train, cv=k, scoring='accuracy')
                mse = cross_val_score(model, self.X_train, y_train, cv=k, scoring='neg_mean_squared_error')
                rmse = np.sqrt(-mse)
                msg = f"K-Fold Accuracy: {accuracy.mean():.3f} ± {accuracy.std():.3f}\nK-Fold RMSE: {rmse.mean():.3f}"
                QMessageBox.information(self, "K-Fold Cross-Validation Results", msg)
            # Train model on all data
            model.fit(self.X_train, y_train)
            self.current_model = model
            
            # Get predictions
            y_pred = model.predict(self.X_test)
            
            # Update metrics display
            self.update_metrics(y_pred)
            
            # Update visualization
            self.plot_predictions()
            self.status_bar.showMessage(f"{name} training complete")
            
        except Exception as e:
            self.show_error(f"Error training {name}: {str(e)}")

    def load_dataset(self):
        """Load selected dataset"""
        try:
            dataset_name = self.dataset_combo.currentText()
            
            if dataset_name == "Custom CSV":
                self.load_custom_data()
                return
            
            # Load selected dataset
            if dataset_name == "Iris":
                data = datasets.load_iris()
                X, y = data.data, data.target
            elif dataset_name == "Boston Housing":
                data = datasets.load_boston()
                X, y = data.data, data.target
            elif dataset_name == "Breast Cancer":
                data = datasets.load_breast_cancer()
                X, y = data.data, data.target
            
            # Apply missing data handling
            method = self.missing_combo.currentText()
            if method == "Mean Imputation":
                imputer = SimpleImputer(strategy="mean")
                X = imputer.fit_transform(X)
            # (No Action: do nothing)

            # Split data
            train_r, val_r, test_r = self.get_split_values() if hasattr(self, 'get_split_values') else (0.8, 0.0, 0.2)
            if val_r > 0:
                X_train, X_tmp, y_train, y_tmp = model_selection.train_test_split(X, y, test_size=(1-train_r), random_state=42)
                val_size = val_r/(val_r+test_r)
                self.X_val, self.X_test, self.y_val, self.y_test = model_selection.train_test_split(X_tmp, y_tmp, test_size=(test_r/(val_r+test_r)), random_state=42)
                self.X_train, self.y_train = X_train, y_train
            else:
                self.X_train, self.X_test, self.y_train, self.y_test = model_selection.train_test_split(X, y, test_size=test_r, random_state=42)
                self.X_val, self.y_val = None, None
            
            # Apply scaling if selected
            self.apply_scaling()
            
            # Update axis options
            self.update_axis_options()
            
            # Update max values for dimred spinboxes
            self.update_dimred_spinboxes()
            
            self.status_bar.showMessage(f"Loaded {dataset_name}")
            self.plot_raw_data()
            
        except Exception as e:
            self.show_error(f"Error loading dataset: {str(e)}")

    def load_custom_data(self):
        """Load custom dataset from CSV file"""
        try:
            file_name, _ = QFileDialog.getOpenFileName(
                self,
                "Load Dataset",
                "",
                "CSV files (*.csv)"
            )
            
            if file_name:
                # Load data
                data = pd.read_csv(file_name)
                
                # Ask user to select target column
                target_col = self.select_target_column(data.columns)
                
                if target_col:
                    X = data.drop(target_col, axis=1).values
                    y = data[target_col].values

                    # Apply missing data handling
                    method = self.missing_combo.currentText()
                    if method == "Mean Imputation":
                        imputer = SimpleImputer(strategy="mean")
                        X = imputer.fit_transform(X)
                    # (No Action: do nothing)

                    # Split data
                    train_r, val_r, test_r = self.get_split_values() if hasattr(self, 'get_split_values') else (0.8, 0.0, 0.2)
                    if val_r > 0:
                        X_train, X_tmp, y_train, y_tmp = model_selection.train_test_split(X, y, test_size=(1-train_r), random_state=42)
                        val_size = val_r/(val_r+test_r)
                        self.X_val, self.X_test, self.y_val, self.y_test = model_selection.train_test_split(X_tmp, y_tmp, test_size=(test_r/(val_r+test_r)), random_state=42)
                        self.X_train, self.y_train = X_train, y_train
                    else:
                        self.X_train, self.X_test, self.y_train, self.y_test = model_selection.train_test_split(X, y, test_size=test_r, random_state=42)
                        self.X_val, self.y_val = None, None
                    
                    # Apply scaling if selected
                    self.apply_scaling()
                    
                    # Update axis options
                    self.update_axis_options()
                    
                    # Update max values for dimred spinboxes
                    self.update_dimred_spinboxes()
                    
                    self.status_bar.showMessage(f"Loaded custom dataset: {file_name}")
                    self.plot_raw_data()
                    
        except Exception as e:
            self.show_error(f"Error loading custom dataset: {str(e)}")

    def train_neural_network(self):
        """Train neural network with current configuration"""
        # Ensure loss settings and label format are applied before training
        self.apply_loss_settings()
        if not self.layer_config:
            self.show_error("Please add at least one layer to the network")
            return
            
        try:
            # Create model
            model = self.create_neural_network()
            
            # Get training parameters
            batch_size = self.batch_size_spin.value()
            epochs = self.epochs_spin.value()
            learning_rate = self.lr_spin.value()
            
            # Compile model with selected loss
            optimizer = optimizers.Adam(learning_rate=learning_rate)
            model.compile(
                optimizer=optimizer,
                loss=self.current_loss,
                metrics=['accuracy'] if self.current_loss in [
                    'categorical_crossentropy', 
                    'binary_crossentropy', 
                    'hinge'
                ] else ['mse', 'mae']
            )
            
            # Train model
            history = model.fit(
                self.X_train, self.y_train,
                batch_size=batch_size,
                epochs=epochs,
                validation_split=0.2,
                class_weight=self.class_weights_dict,
                callbacks=[self.create_progress_callback()]
            )
            
            self.current_model = model
            self.plot_training_history(history)
        except Exception as e:
            self.show_error(f"Error training neural network: {str(e)}")

    def plot_training_history(self, history):
        """Plot training/validation loss and accuracy curves after training."""
        import matplotlib.pyplot as plt
        hist = history.history
        fig, ax1 = plt.subplots()
        ax1.plot(hist.get('loss', []), label='Train Loss')
        if 'val_loss' in hist:
            ax1.plot(hist['val_loss'], label='Val Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend(loc='upper left')
        ax2 = ax1.twinx()
        if 'accuracy' in hist:
            ax2.plot(hist['accuracy'], color='g', label='Train Acc', linestyle='dashed')
        if 'val_accuracy' in hist:
            ax2.plot(hist['val_accuracy'], color='r', label='Val Acc', linestyle='dashed')
        ax2.set_ylabel('Accuracy')
        lines, labels = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax2.legend(lines + lines2, labels + labels2, loc='upper right')
        plt.title('Training History')
        plt.tight_layout()
        plt.show()

    def apply_loss_settings(self):
        """Apply selected loss function settings"""
        try:
            if self.X_train is None or self.y_train is None:
                self.show_error("Please load a dataset first")
                return
                
            # Determine if we're doing classification or regression
            n_unique = len(np.unique(self.y_train))
            is_classification = n_unique <= 10  # Arbitrary threshold
            
            # Get selected loss function
            if is_classification:
                loss_func = self.class_loss_combo.currentText()
                # Handle classification loss settings
                if loss_func == "Cross Entropy":
                    self.current_loss = "categorical_crossentropy"
                    if self.y_train.ndim == 1:
                        self.y_train = tf.keras.utils.to_categorical(self.y_train)
                        self.y_test = tf.keras.utils.to_categorical(self.y_test)
                elif loss_func == "Binary Cross Entropy":
                    self.current_loss = "binary_crossentropy"
                elif loss_func == "Hinge Loss":
                    self.current_loss = "hinge"
                
                # Handle class weights
                weight_option = self.class_weights.currentText()
                if weight_option == "Balanced":
                    from sklearn.utils.class_weight import compute_class_weight
                    classes = np.unique(self.y_train)
                    self.class_weights_dict = dict(zip(
                        classes,
                        compute_class_weight('balanced', classes=classes, y=self.y_train)
                    ))
                elif weight_option == "Custom":
                    # Could add a dialog for custom weights here
                    pass
                else:
                    self.class_weights_dict = None
            else:
                loss_func = self.reg_loss_combo.currentText()
                # Handle regression loss settings
                if loss_func == "Mean Squared Error (MSE)":
                    self.current_loss = "mse"
                elif loss_func == "Mean Absolute Error (MAE)":
                    self.current_loss = "mae"
                elif loss_func == "Huber Loss":
                    self.current_loss = tf.keras.losses.Huber(
                        delta=self.huber_delta.value()
                    )
            
            self.status_bar.showMessage(f"Applied {loss_func} loss function")
            
        except Exception as e:
            self.show_error(f"Error applying loss settings: {str(e)}")

    def create_status_bar(self):
        """Create the status bar"""
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        
        # Add progress bar
        self.progress_bar = QProgressBar()
        self.status_bar.addPermanentWidget(self.progress_bar)

    def create_tabs(self):
        """Create tabs for different ML topics"""
        self.tab_widget = QTabWidget()
        # Create individual tabs
        tabs = [
            ("Classical ML", self.create_classical_ml_tab),
            ("Deep Learning", self.create_deep_learning_tab),
            ("Advanced Deep Learning", self.create_advanced_deep_learning_tab),
            ("Dimensionality Reduction", self.create_dim_reduction_tab),
            ("Reinforcement Learning", self.create_rl_tab),
            ("Advanced Dim Reduction", self.create_advanced_dim_reduction_tab),
            ("Feature Extraction", self.create_feature_extraction_tab),
            ("GAN", self.create_gan_tab)  # Add GAN tab
        ]
        
        for tab_name, create_func in tabs:
            scroll = QScrollArea()
            tab_widget = create_func()
            scroll.setWidget(tab_widget)
            scroll.setWidgetResizable(True)
            self.tab_widget.addTab(scroll, tab_name)
        
        self.layout.addWidget(self.tab_widget)
    
    def create_classical_ml_tab(self):
        """Create the classical machine learning algorithms tab"""
        widget = QWidget()
        layout = QGridLayout(widget)
        
        # Regression section
        regression_group = QGroupBox("Regression")
        regression_layout = QVBoxLayout()
        
        # Linear Regression
        lr_group = self.create_algorithm_group(
            "Linear Regression",
            {"fit_intercept": "checkbox",
             "normalize": "checkbox"}
        )
        regression_layout.addWidget(lr_group)
        
        # Logistic Regression
        logistic_group = self.create_algorithm_group(
            "Logistic Regression",
            {"C": "double",
             "max_iter": "int",
             "multi_class": ["ovr", "multinomial"]}
        )
        regression_layout.addWidget(logistic_group)
        
        regression_group.setLayout(regression_layout)
        layout.addWidget(regression_group, 0, 0)
        
        # Classification section
        classification_group = QGroupBox("Classification")
        classification_layout = QVBoxLayout()
        
        # Naive Bayes
        nb_group = self.create_algorithm_group(
            "Naive Bayes",
            {"var_smoothing": "double",
             "priors": "text"}
        )
        classification_layout.addWidget(nb_group)
        
        # SVM
        svm_group = self.create_algorithm_group(
            "Support Vector Machine",
            {"C": "double",
             "kernel": ["linear", "rbf", "poly"],
             "degree": "int"}
        )
        classification_layout.addWidget(svm_group)
        
        # Decision Trees
        dt_group = self.create_algorithm_group(
            "Decision Tree",
            {"max_depth": "int",
             "min_samples_split": "int",
             "criterion": ["gini", "entropy"]}
        )
        classification_layout.addWidget(dt_group)
        
        # Random Forest
        rf_group = self.create_algorithm_group(
            "Random Forest",
            {"n_estimators": "int",
             "max_depth": "int",
             "min_samples_split": "int"}
        )
        classification_layout.addWidget(rf_group)
        
        # KNN
        knn_group = self.create_algorithm_group(
            "K-Nearest Neighbors",
            {"n_neighbors": "int",
             "weights": ["uniform", "distance"],
             "metric": ["euclidean", "manhattan"]}
        )
        classification_layout.addWidget(knn_group)
        
        classification_group.setLayout(classification_layout)
        layout.addWidget(classification_group, 0, 1)
        
        return widget
    
    def create_dim_reduction_tab(self):
        """Create the dimensionality reduction tab"""
        widget = QWidget()
        layout = QGridLayout(widget)
        
        # K-Means section
        kmeans_group = QGroupBox("K-Means Clustering")
        kmeans_layout = QVBoxLayout()
        
        kmeans_params = self.create_algorithm_group(
            "K-Means Parameters",
            {"n_clusters": "int",
             "max_iter": "int",
             "n_init": "int"}
        )
        kmeans_layout.addWidget(kmeans_params)
        
        kmeans_group.setLayout(kmeans_layout)
        layout.addWidget(kmeans_group, 0, 0)
        
        # PCA section
        pca_group = QGroupBox("Principal Component Analysis")
        pca_layout = QVBoxLayout()
        
        pca_params = self.create_algorithm_group(
            "PCA Parameters",
            {"n_components": "int",
             "whiten": "checkbox"}
        )
        pca_layout.addWidget(pca_params)
        
        pca_group.setLayout(pca_layout)
        layout.addWidget(pca_group, 0, 1)
        
        return widget
    
    def create_rl_tab(self):
        """Create the reinforcement learning tab"""
        widget = QWidget()
        layout = QGridLayout(widget)
        
        # Environment selection
        env_group = QGroupBox("Environment")
        env_layout = QVBoxLayout()
        
        self.env_combo = QComboBox()
        self.env_combo.addItems([
            "CartPole-v1",
            "MountainCar-v0",
            "Acrobot-v1"
        ])
        env_layout.addWidget(self.env_combo)
        
        env_group.setLayout(env_layout)
        layout.addWidget(env_group, 0, 0)
        
        # RL Algorithm selection
        algo_group = QGroupBox("RL Algorithm")
        algo_layout = QVBoxLayout()
        
        self.rl_algo_combo = QComboBox()
        self.rl_algo_combo.addItems([
            "Q-Learning",
            "SARSA",
            "DQN"
        ])
        algo_layout.addWidget(self.rl_algo_combo)
        
        algo_group.setLayout(algo_layout)
        layout.addWidget(algo_group, 0, 1)
        
        return widget
    
    def create_algorithm_group(self, name, params):
        """Helper method to create algorithm parameter groups"""
        group = QGroupBox(name)
        layout = QVBoxLayout()
        
        # Create parameter inputs
        param_widgets = {}
        for param_name, param_type in params.items():
            param_layout = QHBoxLayout()
            param_layout.addWidget(QLabel(f"{param_name}:"))
            
            if param_type == "int":
                widget = QSpinBox()
                widget.setRange(1, 1000)
            elif param_type == "double":
                widget = QDoubleSpinBox()
                widget.setRange(0.0001, 1000.0)
                widget.setSingleStep(0.1)
            elif param_type == "checkbox":
                widget = QCheckBox()
            elif param_type == "text":
                widget = QLineEdit()
            elif isinstance(param_type, list):
                widget = QComboBox()
                widget.addItems(param_type)
            
            param_layout.addWidget(widget)
            param_widgets[param_name] = widget
            layout.addLayout(param_layout)

        # Add custom priors field for Naive Bayes
        if name == "Naive Bayes":
            priors_layout = QHBoxLayout()
            priors_layout.addWidget(QLabel("Priors (e.g. [0.3, 0.7]):"))
            priors_widget = QLineEdit()
            param_widgets["priors"] = priors_widget
            priors_layout.addWidget(priors_widget)
            layout.addLayout(priors_layout)

        # Add train button
        train_btn = QPushButton(f"Train {name}")
        train_btn.clicked.connect(lambda: self.train_model(name, param_widgets))
        layout.addWidget(train_btn)
        
        group.setLayout(layout)
        return group

    def show_error(self, message):
        """Show error message dialog"""
        QMessageBox.critical(self, "Error", message)
       
    def create_deep_learning_tab(self):
        """Create the deep learning tab"""
        widget = QWidget()
        layout = QGridLayout(widget)
        
        # MLP section
        mlp_group = QGroupBox("Multi-Layer Perceptron")
        mlp_layout = QVBoxLayout()
        
        # Layer configuration
        self.layer_config = []
        layer_btn = QPushButton("Add Layer")
        layer_btn.clicked.connect(self.add_layer_dialog)
        mlp_layout.addWidget(layer_btn)
        
        # Training parameters
        training_params_group = self.create_training_params_group()
        mlp_layout.addWidget(training_params_group)
        
        # Train button
        train_btn = QPushButton("Train Neural Network")
        train_btn.clicked.connect(self.train_neural_network)
        mlp_layout.addWidget(train_btn)
        
        mlp_group.setLayout(mlp_layout)
        layout.addWidget(mlp_group, 0, 0)
        
        # CNN section
        cnn_group = QGroupBox("Convolutional Neural Network")
        cnn_layout = QVBoxLayout()
        
        # CNN architecture controls
        cnn_controls = self.create_cnn_controls()
        cnn_layout.addWidget(cnn_controls)
        
        cnn_group.setLayout(cnn_layout)
        layout.addWidget(cnn_group, 0, 1)
        
        # RNN section
        rnn_group = QGroupBox("Recurrent Neural Network")
        rnn_layout = QVBoxLayout()
        
        # RNN architecture controls
        rnn_controls = self.create_rnn_controls()
        rnn_layout.addWidget(rnn_controls)
        
        rnn_group.setLayout(rnn_layout)
        layout.addWidget(rnn_group, 1, 0)
        
        return widget
    
    def add_layer_dialog(self):
        """Open a dialog to add a neural network layer"""
        dialog = QDialog(self)
        dialog.setWindowTitle("Add Neural Network Layer")
        layout = QVBoxLayout(dialog)
        
        # Layer type selection
        type_layout = QHBoxLayout()
        type_label = QLabel("Layer Type:")
        type_combo = QComboBox()
        type_combo.addItems(["Dense", "Conv2D", "MaxPooling2D", "Flatten", "Dropout"])
        type_layout.addWidget(type_label)
        type_layout.addWidget(type_combo)
        layout.addLayout(type_layout)
        
        # Parameters input
        params_group = QGroupBox("Layer Parameters")
        params_layout = QVBoxLayout()
        
        # Dynamic parameter inputs based on layer type
        self.layer_param_inputs = {}
        
        def update_params():
            # Clear existing parameter inputs
            for widget in list(self.layer_param_inputs.values()):
                params_layout.removeWidget(widget)
                widget.deleteLater()
            self.layer_param_inputs.clear()
            
            layer_type = type_combo.currentText()
            if layer_type == "Dense":
                units_label = QLabel("Units:")
                units_input = QSpinBox()
                units_input.setRange(1, 1000)
                units_input.setValue(32)
                self.layer_param_inputs["units"] = units_input
                
                activation_label = QLabel("Activation:")
                activation_combo = QComboBox()
                activation_combo.addItems(["relu", "sigmoid", "tanh", "softmax"])
                self.layer_param_inputs["activation"] = activation_combo
                
                params_layout.addWidget(units_label)
                params_layout.addWidget(units_input)
                params_layout.addWidget(activation_label)
                params_layout.addWidget(activation_combo)
            
            elif layer_type == "Conv2D":
                filters_label = QLabel("Filters:")
                filters_input = QSpinBox()
                filters_input.setRange(1, 1000)
                filters_input.setValue(32)
                self.layer_param_inputs["filters"] = filters_input
                
                kernel_label = QLabel("Kernel Size:")
                kernel_input = QLineEdit()
                kernel_input.setText("3, 3")
                self.layer_param_inputs["kernel_size"] = kernel_input
                
                params_layout.addWidget(filters_label)
                params_layout.addWidget(filters_input)
                params_layout.addWidget(kernel_label)
                params_layout.addWidget(kernel_input)
            
            elif layer_type == "Dropout":
                rate_label = QLabel("Dropout Rate:")
                rate_input = QDoubleSpinBox()
                rate_input.setRange(0.0, 1.0)
                rate_input.setValue(0.5)
                rate_input.setSingleStep(0.1)
                self.layer_param_inputs["rate"] = rate_input
                
                params_layout.addWidget(rate_label)
                params_layout.addWidget(rate_input)
        
        type_combo.currentIndexChanged.connect(update_params)
        update_params()  # Initial update
        
        params_group.setLayout(params_layout)
        layout.addWidget(params_group)
        
        # Buttons
        btn_layout = QHBoxLayout()
        add_btn = QPushButton("Add Layer")
        cancel_btn = QPushButton("Cancel")
        btn_layout.addWidget(add_btn)
        btn_layout.addWidget(cancel_btn)
        layout.addLayout(btn_layout)
        
        def add_layer():
            layer_type = type_combo.currentText()
            
            # Collect parameters
            layer_params = {}
            for param_name, widget in self.layer_param_inputs.items():
                if isinstance(widget, QSpinBox):
                    layer_params[param_name] = widget.value()
                elif isinstance(widget, QDoubleSpinBox):
                    layer_params[param_name] = widget.value()
                elif isinstance(widget, QComboBox):
                    layer_params[param_name] = widget.currentText()
                elif isinstance(widget, QLineEdit):
                    # Handle kernel size or other tuple-like inputs
                    if param_name == "kernel_size":
                        layer_params[param_name] = tuple(map(int, widget.text().split(',')))
            
            self.layer_config.append({
                "type": layer_type,
                "params": layer_params
            })
            
            dialog.accept()
        
        add_btn.clicked.connect(add_layer)
        cancel_btn.clicked.connect(dialog.reject)
        
        dialog.exec()
    
    def create_training_params_group(self):
        """Create group for neural network training parameters"""
        group = QGroupBox("Training Parameters")
        layout = QVBoxLayout()
        
        # Batch size
        batch_layout = QHBoxLayout()
        batch_layout.addWidget(QLabel("Batch Size:"))
        self.batch_size_spin = QSpinBox()
        self.batch_size_spin.setRange(1, 1000)
        self.batch_size_spin.setValue(32)
        batch_layout.addWidget(self.batch_size_spin)
        layout.addLayout(batch_layout)
        
        # Epochs
        epochs_layout = QHBoxLayout()
        epochs_layout.addWidget(QLabel("Epochs:"))
        self.epochs_spin = QSpinBox()
        self.epochs_spin.setRange(1, 1000)
        self.epochs_spin.setValue(10)
        epochs_layout.addWidget(self.epochs_spin)
        layout.addLayout(epochs_layout)
        
        # Learning rate
        lr_layout = QHBoxLayout()
        lr_layout.addWidget(QLabel("Learning Rate:"))
        self.lr_spin = QDoubleSpinBox()
        self.lr_spin.setRange(0.0001, 1.0)
        self.lr_spin.setValue(0.001)
        self.lr_spin.setSingleStep(0.001)
        lr_layout.addWidget(self.lr_spin)
        layout.addLayout(lr_layout)
        
        group.setLayout(layout)
        return group
    
    def create_cnn_controls(self):
        """Create controls for Convolutional Neural Network"""
        group = QGroupBox("CNN Architecture")
        layout = QVBoxLayout()

        # Filtre sayısı
        filters_label = QLabel("Filters:")
        filters_spin = QSpinBox()
        filters_spin.setRange(1, 512)
        filters_spin.setValue(32)
        layout.addWidget(filters_label)
        layout.addWidget(filters_spin)

        # Kernel boyutu
        kernel_label = QLabel("Kernel Size:")
        kernel_spin = QSpinBox()
        kernel_spin.setRange(1, 10)
        kernel_spin.setValue(3)
        layout.addWidget(kernel_label)
        layout.addWidget(kernel_spin)

        # Aktivasyon fonksiyonu
        activation_label = QLabel("Activation:")
        activation_combo = QComboBox()
        activation_combo.addItems(["relu", "sigmoid", "tanh", "softmax"])
        layout.addWidget(activation_label)
        layout.addWidget(activation_combo)

        # Katman ekle butonu
        add_btn = QPushButton("Add Conv2D Layer")
        def add_cnn_layer():
            params = {
                "filters": filters_spin.value(),
                "kernel_size": kernel_spin.value(),
                "activation": activation_combo.currentText(),
                # input_shape ilk katmanda ekleniyor, burada eklenmiyor
            }
            self.layer_config.append({"type": "Conv2D", "params": params})
            QMessageBox.information(self, "Layer Added", f"Conv2D layer added: {params}")
        add_btn.clicked.connect(add_cnn_layer)
        layout.addWidget(add_btn)

        # MaxPooling ekle
        pool_btn = QPushButton("Add MaxPooling2D")
        def add_pool_layer():
            self.layer_config.append({"type": "MaxPooling2D", "params": {}})
            QMessageBox.information(self, "Layer Added", "MaxPooling2D layer added.")
        pool_btn.clicked.connect(add_pool_layer)
        layout.addWidget(pool_btn)

        # Flatten ekle
        flatten_btn = QPushButton("Add Flatten")
        def add_flatten_layer():
            self.layer_config.append({"type": "Flatten", "params": {}})
            QMessageBox.information(self, "Layer Added", "Flatten layer added.")
        flatten_btn.clicked.connect(add_flatten_layer)
        layout.addWidget(flatten_btn)

        group.setLayout(layout)
        return group

    
    def create_rnn_controls(self):
        """Create controls for Recurrent Neural Network"""
        group = QGroupBox("RNN Architecture")
        layout = QVBoxLayout()

        # RNN tipi
        rnn_type_label = QLabel("RNN Type:")
        rnn_type_combo = QComboBox()
        rnn_type_combo.addItems(["LSTM", "GRU"])
        layout.addWidget(rnn_type_label)
        layout.addWidget(rnn_type_combo)

        # Ünite sayısı
        units_label = QLabel("Units:")
        units_spin = QSpinBox()
        units_spin.setRange(1, 512)
        units_spin.setValue(32)
        layout.addWidget(units_label)
        layout.addWidget(units_spin)

        # Aktivasyon fonksiyonu
        activation_label = QLabel("Activation:")
        activation_combo = QComboBox()
        activation_combo.addItems(["tanh", "relu", "sigmoid"])
        layout.addWidget(activation_label)
        layout.addWidget(activation_combo)

        # Return sequences
        return_seq_chk = QCheckBox("Return Sequences")
        layout.addWidget(return_seq_chk)

        # Katman ekle butonu
        add_btn = QPushButton("Add RNN Layer")
        def add_rnn_layer():
            params = {
                "units": units_spin.value(),
                "activation": activation_combo.currentText(),
                "return_sequences": return_seq_chk.isChecked()
            }
            rnn_type = rnn_type_combo.currentText()
            self.layer_config.append({"type": rnn_type, "params": params})
            QMessageBox.information(self, "Layer Added", f"{rnn_type} layer added: {params}")
        add_btn.clicked.connect(add_rnn_layer)
        layout.addWidget(add_btn)

        group.setLayout(layout)
        return group

    
    def create_neural_network(self):
        """Create neural network based on current configuration"""
        model = models.Sequential()
        first_layer = True
        for layer_config in self.layer_config:
            layer_type = layer_config["type"]
            params = layer_config["params"].copy()  # Orijinali bozma
            # Dense
            if layer_type == "Dense":
                if first_layer and 'input_shape' not in params:
                    params['input_shape'] = self.X_train.shape[1:]
                    first_layer = False
                model.add(layers.Dense(**params))
            # Conv2D
            elif layer_type == "Conv2D":
                if first_layer and 'input_shape' not in params:
                    params['input_shape'] = self.X_train.shape[1:]
                    first_layer = False
                model.add(layers.Conv2D(**params))
            # MaxPooling2D
            elif layer_type == "MaxPooling2D":
                model.add(layers.MaxPooling2D())
            # Flatten
            elif layer_type == "Flatten":
                model.add(layers.Flatten())
            # Dropout
            elif layer_type == "Dropout":
                model.add(layers.Dropout(**params))
            # LSTM
            elif layer_type == "LSTM":
                if first_layer and 'input_shape' not in params:
                    params['input_shape'] = self.X_train.shape[1:]
                    first_layer = False
                model.add(layers.LSTM(**params))
            # GRU
            elif layer_type == "GRU":
                if first_layer and 'input_shape' not in params:
                    params['input_shape'] = self.X_train.shape[1:]
                    first_layer = False
                model.add(layers.GRU(**params))
        # Eğer Conv2D varsa ve Flatten yoksa ekle
        if any(isinstance(layer, layers.Conv2D) for layer in model.layers):
            if not any(isinstance(layer, layers.Flatten) for layer in model.layers):
                model.add(layers.Flatten())
        # Çıkış katmanı
        if isinstance(self.y_train, np.ndarray) and self.y_train.ndim > 1:
            num_classes = self.y_train.shape[1]
        else:
            num_classes = len(np.unique(self.y_train))
        model.add(layers.Dense(num_classes, activation='softmax'))
        return model


    def create_progress_callback(self):
        """Create callback for updating progress bar during training"""
        class ProgressCallback(tf.keras.callbacks.Callback):
            def __init__(self, progress_bar):

                super().__init__()
                self.progress_bar = progress_bar
                
            def on_epoch_end(self, epoch, logs=None):
                progress = int(((epoch + 1) / self.params['epochs']) * 100)
                QApplication.processEvents()  # Process pending events
                self.progress_bar.setValue(progress)
                QApplication.processEvents()  # Process pending events
                
        return ProgressCallback(self.progress_bar)
        
    def update_visualization(self, y_pred):
        """Update the visualization with current results"""
        self.figure.clear()
        
        # Create appropriate visualization based on data
        if len(np.unique(self.y_test)) > 10:  # Regression
            ax = self.figure.add_subplot(111)
            ax.scatter(self.y_test, y_pred)
            ax.plot([self.y_test.min(), self.y_test.max()],
                   [self.y_test.min(), self.y_test.max()],
                   'r--', lw=2)
            ax.set_xlabel("Actual Values")
            ax.set_ylabel("Predicted Values")
            
        else:  # Classification
            if self.X_train.shape[1] > 2:  # Use PCA for visualization
                pca = PCA(n_components=2)
                X_test_2d = pca.fit_transform(self.X_test)
                
                ax = self.figure.add_subplot(111)
                scatter = ax.scatter(X_test_2d[:, 0], X_test_2d[:, 1],
                                   c=y_pred, cmap='viridis')
                self.figure.colorbar(scatter)
                
            else:  # Direct 2D visualization
                ax = self.figure.add_subplot(111)
                scatter = ax.scatter(self.X_test[:, 0], self.X_test[:, 1],
                                   c=y_pred, cmap='viridis')
                self.figure.colorbar(scatter)
        
        self.canvas.draw()
        
    def update_metrics(self, y_pred):
        """Update metrics display"""
        metrics_text = "Model Performance Metrics:\n\n"
        
        # Calculate appropriate metrics based on problem type
        if len(np.unique(self.y_test)) > 10:  # Regression
            mse = mean_squared_error(self.y_test, y_pred)
            rmse = np.sqrt(mse)
            r2 = self.current_model.score(self.X_test, self.y_test)
            
            metrics_text += f"Mean Squared Error: {mse:.4f}\n"
            metrics_text += f"Root Mean Squared Error: {rmse:.4f}\n"
            metrics_text += f"R² Score: {r2:.4f}"
            
        else:  # Classification
            accuracy = accuracy_score(self.y_test, y_pred)
            conf_matrix = confusion_matrix(self.y_test, y_pred)
            
            metrics_text += f"Accuracy: {accuracy:.4f}\n\n"
            metrics_text += f"Confusion Matrix:\n{conf_matrix}"
        
        self.metrics_text.setText(metrics_text)
        
    def apply_scaling(self):
        """Apply selected scaling method to the data"""
        scaling_method = self.scaling_combo.currentText()
        
        if scaling_method != "No Scaling":
            try:
                if scaling_method == "Standard Scaling":
                    scaler = preprocessing.StandardScaler()
                elif scaling_method == "Min-Max Scaling":
                    scaler = preprocessing.MinMaxScaler()
                elif scaling_method == "Robust Scaling":
                    scaler = preprocessing.RobustScaler()
                
                self.X_train = scaler.fit_transform(self.X_train)
                self.X_test = scaler.transform(self.X_test)
                
            except Exception as e:
                self.show_error(f"Error applying scaling: {str(e)}")

    def create_feature_extraction_tab(self):
        """Tab for PCA, SVD, t-SNE, LDA: user-selectable components and explained variance/görselleştirme ekle."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # PCA Section
        pca_group = QGroupBox("Principal Component Analysis (PCA)")
        pca_layout = QVBoxLayout()
        self.pca_n_components = QSpinBox()
        self.pca_n_components.setRange(1, 20)
        self.pca_n_components.setValue(2)
        pca_layout.addWidget(QLabel("Number of components:"))
        pca_layout.addWidget(self.pca_n_components)
        self.pca_btn = QPushButton("Run PCA")
        self.pca_btn.clicked.connect(self.run_pca)
        pca_layout.addWidget(self.pca_btn)
        pca_group.setLayout(pca_layout)
        
        # SVD Section
        svd_group = QGroupBox("Truncated SVD")
        svd_layout = QVBoxLayout()
        self.svd_n_components = QSpinBox()
        self.svd_n_components.setRange(1, 20)
        self.svd_n_components.setValue(2)
        svd_layout.addWidget(QLabel("Number of components:"))
        svd_layout.addWidget(self.svd_n_components)
        self.svd_btn = QPushButton("Run SVD")
        self.svd_btn.clicked.connect(self.run_svd)
        svd_layout.addWidget(self.svd_btn)
        svd_group.setLayout(svd_layout)
        
        # t-SNE Section
        tsne_group = QGroupBox("t-SNE")
        tsne_layout = QVBoxLayout()
        self.tsne_n_components = QSpinBox()
        self.tsne_n_components.setRange(2, 3)
        self.tsne_n_components.setValue(2)
        tsne_layout.addWidget(QLabel("Number of components:"))
        tsne_layout.addWidget(self.tsne_n_components)
        self.tsne_btn = QPushButton("Run t-SNE")
        self.tsne_btn.clicked.connect(self.run_tsne_projection)
        tsne_layout.addWidget(self.tsne_btn)
        tsne_group.setLayout(tsne_layout)
        
        # LDA Section
        lda_group = QGroupBox("LDA (Linear Discriminant Analysis)")
        lda_layout = QVBoxLayout()
        self.lda_n_components = QSpinBox()
        self.lda_n_components.setRange(1, 5)
        self.lda_n_components.setValue(1)
        lda_layout.addWidget(QLabel("Number of components:"))
        lda_layout.addWidget(self.lda_n_components)
        self.lda_btn = QPushButton("Run LDA")
        self.lda_btn.clicked.connect(self.run_lda_projection)
        lda_layout.addWidget(self.lda_btn)
        lda_group.setLayout(lda_layout)
        
        # Add all sections
        layout.addWidget(pca_group)
        layout.addWidget(svd_group)
        layout.addWidget(tsne_group)
        layout.addWidget(lda_group)
        widget.setLayout(layout)
        return widget

    def run_pca(self):
        if self.X_train is None:
            self.show_error("Load data first!")
            return
        n = self.pca_n_components.value()
        pca = PCA(n_components=n)
        X_pca = pca.fit_transform(self.X_train)
        plt.figure(figsize=(6,4))
        plt.plot(np.arange(1, len(pca.explained_variance_ratio_)+1), np.cumsum(pca.explained_variance_ratio_), marker="o")
        plt.xlabel("# Components")
        plt.ylabel("Cumulative Explained Variance")
        plt.title("PCA Explained Variance")
        plt.tight_layout()
        plt.show()

    def run_svd(self):
        if self.X_train is None:
            self.show_error("Load data first!")
            return
        n = self.svd_n_components.value()
        svd = TruncatedSVD(n_components=n)
        X_svd = svd.fit_transform(self.X_train)
        plt.figure(figsize=(6,4))
        plt.plot(np.arange(1, len(svd.explained_variance_ratio_)+1), np.cumsum(svd.explained_variance_ratio_), marker="o")
        plt.xlabel("# Components")
        plt.ylabel("Cumulative Explained Variance")
        plt.title("SVD Explained Variance")
        plt.tight_layout()
        plt.show()

    def run_tsne_projection(self):
        if self.X_train is None:
            self.show_error("Load data first!")
            return
        n = self.tsne_n_components.value()
        tsne = TSNE(n_components=n, random_state=0)
        X_tsne = tsne.fit_transform(self.X_train)
        plt.figure(figsize=(6,4))
        plt.scatter(X_tsne[:,0], X_tsne[:,1], c='b', s=8)
        plt.title("t-SNE Projection")
        plt.tight_layout()
        plt.show()

    def run_lda_projection(self):
        if self.X_train is None or self.y_train is None:
            self.show_error("Load data first!")
            return
        n = self.lda_n_components.value()
        lda = LDA(n_components=n)
        try:
            X_lda = lda.fit_transform(self.X_train, np.argmax(self.y_train, axis=1) if len(self.y_train.shape) > 1 else self.y_train)
            plt.figure(figsize=(6,4))
            if n == 1:
                plt.hist(X_lda, bins=30)
            else:
                plt.scatter(X_lda[:,0], X_lda[:,1], c='g', s=8)
            plt.title("LDA Projection")
            plt.tight_layout()
            plt.show()
        except Exception as e:
            self.show_error(f"LDA error: {e}")

    def create_advanced_dim_reduction_tab(self):
        """Tab for PCA, LDA, t-SNE, KMeans: user-selectable components and explained variance/görselleştirme ekle."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # PCA Section
        pca_group = QGroupBox("Principal Component Analysis (PCA)")
        pca_layout = QVBoxLayout()
        self.pca2_n_components = QSpinBox()
        self.pca2_n_components.setRange(1, 20)
        self.pca2_n_components.setValue(2)
        pca_layout.addWidget(QLabel("Number of components:"))
        pca_layout.addWidget(self.pca2_n_components)
        self.pca2_btn = QPushButton("Run PCA")
        self.pca2_btn.clicked.connect(self.run_pca2)
        pca_layout.addWidget(self.pca2_btn)
        pca_group.setLayout(pca_layout)
        
        # LDA Section
        lda_group = QGroupBox("LDA (Linear Discriminant Analysis)")
        lda_layout = QVBoxLayout()
        self.lda2_n_components = QSpinBox()
        self.lda2_n_components.setRange(1, 5)
        self.lda2_n_components.setValue(1)
        lda_layout.addWidget(QLabel("Number of components:"))
        lda_layout.addWidget(self.lda2_n_components)
        self.lda2_btn = QPushButton("Run LDA")
        self.lda2_btn.clicked.connect(self.run_lda2)
        lda_layout.addWidget(self.lda2_btn)
        lda_group.setLayout(lda_layout)
        
        # KMeans Section
        kmeans_group = QGroupBox("K-Means Clustering")
        kmeans_layout = QVBoxLayout()
        self.kmeans_k = QSpinBox()
        self.kmeans_k.setRange(2, 10)
        self.kmeans_k.setValue(3)
        kmeans_layout.addWidget(QLabel("Number of clusters (k):"))
        kmeans_layout.addWidget(self.kmeans_k)
        self.kmeans_btn = QPushButton("Run KMeans")
        self.kmeans_btn.clicked.connect(self.run_kmeans2)
        kmeans_layout.addWidget(self.kmeans_btn)
        self.elbow_btn = QPushButton("Show Elbow Method")
        self.elbow_btn.clicked.connect(self.show_elbow2)
        kmeans_layout.addWidget(self.elbow_btn)
        kmeans_group.setLayout(kmeans_layout)
        
        # t-SNE Section
        tsne_group = QGroupBox("t-SNE (Interactive Projections)")
        tsne_layout = QVBoxLayout()
        self.tsne2_n_components = QSpinBox()
        self.tsne2_n_components.setRange(2, 3)
        self.tsne2_n_components.setValue(2)
        tsne_layout.addWidget(QLabel("Number of components:"))
        tsne_layout.addWidget(self.tsne2_n_components)
        self.tsne2_perplexity = QDoubleSpinBox()
        self.tsne2_perplexity.setRange(5, 50)
        self.tsne2_perplexity.setValue(30)
        tsne_layout.addWidget(QLabel("Perplexity:"))
        tsne_layout.addWidget(self.tsne2_perplexity)
        self.tsne2_btn = QPushButton("Run t-SNE")
        self.tsne2_btn.clicked.connect(self.run_tsne2)
        tsne_layout.addWidget(self.tsne2_btn)
        tsne_group.setLayout(tsne_layout)
        
        # Add all sections
        layout.addWidget(pca_group)
        layout.addWidget(lda_group)
        layout.addWidget(kmeans_group)
        layout.addWidget(tsne_group)
        widget.setLayout(layout)
        return widget

    def run_pca2(self):
        if self.X_train is None:
            self.show_error("Load data first!")
            return
        n = self.pca2_n_components.value()
        pca = PCA(n_components=n)
        X_pca = pca.fit_transform(self.X_train)
        plt.figure(figsize=(6,4))
        plt.plot(np.arange(1, len(pca.explained_variance_ratio_)+1), np.cumsum(pca.explained_variance_ratio_), marker="o")
        plt.xlabel("# Components")
        plt.ylabel("Cumulative Explained Variance")
        plt.title("PCA Explained Variance")
        plt.tight_layout()
        plt.show()

    def run_lda2(self):
        if self.X_train is None or self.y_train is None:
            self.show_error("Load data first!")
            return
        n = self.lda2_n_components.value()
        lda = LDA(n_components=n)
        try:
            y = np.argmax(self.y_train, axis=1) if len(self.y_train.shape) > 1 else self.y_train
            X_lda = lda.fit_transform(self.X_train, y)
            plt.figure(figsize=(6,4))
            if n == 1:
                plt.hist(X_lda, bins=30)
            else:
                plt.scatter(X_lda[:,0], X_lda[:,1], c=y, s=8, cmap='tab10')
            plt.title("LDA Projection")
            plt.tight_layout()
            plt.show()
        except Exception as e:
            self.show_error(f"LDA error: {e}")

    def run_kmeans2(self):
        if self.X_train is None:
            self.show_error("Load data first!")
            return
        k = self.kmeans_k.value()
        from sklearn.cluster import KMeans
        km = KMeans(n_clusters=k, n_init=10)
        labels = km.fit_predict(self.X_train)
        sil = silhouette_score(self.X_train, labels)
        plt.figure(figsize=(6,4))
        if self.X_train.shape[1] >= 2:
            plt.scatter(self.X_train[:,0], self.X_train[:,1], c=labels, cmap='tab10', s=8)
        plt.title("KMeans Clustering (first 2 dims)")
        plt.suptitle(f"Silhouette score: {sil:.3f}")
        plt.tight_layout()
        plt.show()

    def show_elbow2(self):
        if self.X_train is None:
            self.show_error("Load data first!")
            return
        inertias = []
        K = range(1, 11)
        from sklearn.cluster import KMeans
        for k in K:
            km = KMeans(n_clusters=k, n_init=10).fit(self.X_train)
            inertias.append(km.inertia_)
        plt.figure(figsize=(6,4))
        plt.plot(K, inertias, marker="o")
        plt.xlabel("k")
        plt.ylabel("Inertia (Elbow)")
        plt.title("Elbow Method")
        plt.tight_layout()
        plt.show()

    def run_tsne2(self):
        if self.X_train is None:
            self.show_error("Load data first!")
            return
        n = self.tsne2_n_components.value()
        perplexity = self.tsne2_perplexity.value()
        tsne = TSNE(n_components=n, perplexity=perplexity, random_state=0)
        X_tsne = tsne.fit_transform(self.X_train)
        plt.figure(figsize=(6,4))
        if n == 2:
            plt.scatter(X_tsne[:,0], X_tsne[:,1], c='b', s=8)
        else:
            from mpl_toolkits.mplot3d import Axes3D
            ax = plt.axes(projection='3d')
            ax.scatter(X_tsne[:,0], X_tsne[:,1], X_tsne[:,2], c='b', s=8)
        plt.title("t-SNE Projection")
        plt.tight_layout()
        plt.show()
    
    def update_dimred_spinboxes(self):
        """Update max values of all dimension reduction component spinboxes based on current data."""
        if self.X_train is not None:
            n_samples = self.X_train.shape[0]
            n_features = self.X_train.shape[1]
            max_comp = min(n_samples, n_features)
            # PCA
            if hasattr(self, 'pca_n_components'):
                self.pca_n_components.setMaximum(max_comp)
            if hasattr(self, 'pca2_n_components'):
                self.pca2_n_components.setMaximum(max_comp)
            # SVD
            if hasattr(self, 'svd_n_components'):
                self.svd_n_components.setMaximum(max_comp)
            # LDA (max sınıf sayısı - 1)
            if hasattr(self, 'lda_n_components') and self.y_train is not None:
                try:
                    n_classes = len(np.unique(self.y_train))
                    self.lda_n_components.setMaximum(min(max_comp, n_classes-1))
                except Exception:
                    self.lda_n_components.setMaximum(max_comp)
            if hasattr(self, 'lda2_n_components') and self.y_train is not None:
                try:
                    n_classes = len(np.unique(self.y_train))
                    self.lda2_n_components.setMaximum(min(max_comp, n_classes-1))
                except Exception:
                    self.lda2_n_components.setMaximum(max_comp)
            # t-SNE
            if hasattr(self, 'tsne_n_components'):
                self.tsne_n_components.setMaximum(min(3, max_comp))
            if hasattr(self, 'tsne2_n_components'):
                self.tsne2_n_components.setMaximum(min(3, max_comp))

    def get_split_values(self):
        opt = self.split_combo.currentText()
        if opt.startswith("80-20"):
            return 0.8, 0.0, 0.2
        elif opt.startswith("70-15-15"):
            return 0.7, 0.15, 0.15
        elif opt.startswith("60-20-20"):
            return 0.6, 0.2, 0.2
        else:
            return 0.8, 0.0, 0.2

    def create_advanced_deep_learning_tab(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)
        # Layer configuration
        layers_group = QGroupBox("Layer Configuration")
        layers_layout = QVBoxLayout()
        self.adv_layers_list = QListWidget()
        btn_layout = QHBoxLayout()
        add_btn = QPushButton("Add Layer")
        remove_btn = QPushButton("Remove Layer")
        add_btn.clicked.connect(self.add_layer_dialog)
        remove_btn.clicked.connect(lambda: self.adv_layers_list.takeItem(self.adv_layers_list.currentRow()))
        btn_layout.addWidget(add_btn)
        btn_layout.addWidget(remove_btn)
        layers_layout.addWidget(self.adv_layers_list)
        layers_layout.addLayout(btn_layout)
        layers_group.setLayout(layers_layout)
        layout.addWidget(layers_group)
        # Model I/O
        io_group = QGroupBox("Model I/O")
        io_layout = QHBoxLayout()
        save_btn = QPushButton("Save Model")
        load_btn = QPushButton("Load Model")
        save_btn.clicked.connect(self.save_model)
        load_btn.clicked.connect(self.load_model)
        io_layout.addWidget(save_btn)
        io_layout.addWidget(load_btn)
        io_group.setLayout(io_layout)
        layout.addWidget(io_group)
        # Optimizer / LR Schedule
        opt_group = QGroupBox("Optimizer / LR Schedule")
        opt_layout = QHBoxLayout()
        self.optim_combo = QComboBox()
        self.optim_combo.addItems(["Adam", "SGD", "RMSprop"])
        self.lr_sched_combo = QComboBox()
        self.lr_sched_combo.addItems(["None", "Step Decay", "Exponential"])
        opt_layout.addWidget(QLabel("Optimizer:"))
        opt_layout.addWidget(self.optim_combo)
        opt_layout.addWidget(QLabel("LR Schedule:"))
        opt_layout.addWidget(self.lr_sched_combo)
        opt_group.setLayout(opt_layout)
        layout.addWidget(opt_group)
        # Regularization
        reg_group = QGroupBox("Regularization")
        reg_layout = QHBoxLayout()
        self.dropout_spin = QDoubleSpinBox()
        self.dropout_spin.setRange(0.0, 1.0)
        self.dropout_spin.setSingleStep(0.1)
        self.l2_spin = QDoubleSpinBox()
        self.l2_spin.setRange(0.0, 1.0)
        self.l2_spin.setSingleStep(0.001)
        reg_layout.addWidget(QLabel("Dropout:"))
        reg_layout.addWidget(self.dropout_spin)
        reg_layout.addWidget(QLabel("L2:"))
        reg_layout.addWidget(self.l2_spin)
        reg_group.setLayout(reg_layout)
        layout.addWidget(reg_group)
        # Image Augmentation
        aug_group = QGroupBox("Image Augmentation")
        aug_layout = QHBoxLayout()
        self.aug_rot = QDoubleSpinBox()
        self.aug_rot.setRange(0, 180)
        self.aug_rot.setSingleStep(1)
        self.aug_flip = QCheckBox("Flip")
        self.aug_zoom = QDoubleSpinBox()
        self.aug_zoom.setRange(1.0, 3.0)
        self.aug_zoom.setSingleStep(0.1)
        aug_layout.addWidget(QLabel("Rotation:"))
        aug_layout.addWidget(self.aug_rot)
        aug_layout.addWidget(self.aug_flip)
        aug_layout.addWidget(QLabel("Zoom:"))
        aug_layout.addWidget(self.aug_zoom)
        aug_group.setLayout(aug_layout)
        layout.addWidget(aug_group)
        # Pre-trained Model
        pre_group = QGroupBox("Pre-trained Model")
        pre_layout = QHBoxLayout()
        self.pre_combo = QComboBox()
        self.pre_combo.addItems(["None", "VGG16", "ResNet50"])
        self.ft_check = QCheckBox("Fine-tune")
        pre_layout.addWidget(self.pre_combo)
        pre_layout.addWidget(self.ft_check)
        pre_group.setLayout(pre_layout)
        layout.addWidget(pre_group)
        # Training Controls
        ctrl_layout = QHBoxLayout()
        train_btn2 = QPushButton("Train")
        eval_btn = QPushButton("Evaluate")
        curves_btn = QPushButton("Plot Curves")
        grad_btn = QPushButton("Plot Gradients")
        train_btn2.clicked.connect(self.train_advanced_model)
        eval_btn.clicked.connect(self.evaluate_advanced_model)
        curves_btn.clicked.connect(self.plot_training_curves)
        grad_btn.clicked.connect(self.plot_gradient_histograms)
        ctrl_layout.addWidget(train_btn2)
        ctrl_layout.addWidget(eval_btn)
        ctrl_layout.addWidget(curves_btn)
        ctrl_layout.addWidget(grad_btn)
        layout.addLayout(ctrl_layout)
        return widget

    def save_model(self):
        """Save the current model architecture and weights"""
        if self.current_model is None:
            self.show_error("No model to save!")
            return
            
        try:
            file_name, _ = QFileDialog.getSaveFileName(
                self,
                "Save Model",
                "",
                "HDF5 files (*.h5);;JSON files (*.json)"
            )
            
            if file_name:
                if file_name.endswith('.h5'):
                    self.current_model.save(file_name)
                elif file_name.endswith('.json'):
                    model_json = self.current_model.to_json()
                    with open(file_name, 'w') as f:
                        f.write(model_json)
                self.status_bar.showMessage(f"Model saved to {file_name}")
        except Exception as e:
            self.show_error(f"Error saving model: {str(e)}")

    def load_model(self):
        """Load a saved model"""
        try:
            file_name, _ = QFileDialog.getOpenFileName(
                self,
                "Load Model",
                "",
                "HDF5 files (*.h5);;JSON files (*.json)"
            )
            
            if file_name:
                if file_name.endswith('.h5'):
                    self.current_model = models.load_model(file_name)
                elif file_name.endswith('.json'):
                    with open(file_name, 'r') as f:
                        model_json = f.read()
                    self.current_model = models.model_from_json(model_json)
                self.status_bar.showMessage(f"Model loaded from {file_name}")
        except Exception as e:
            self.show_error(f"Error loading model: {str(e)}")

    def train_advanced_model(self):
        """Train the model with advanced settings"""
        if self.X_train is None or self.y_train is None:
            self.show_error("Please load data first!")
            return
            
        try:
            # Create model if not exists
            if self.current_model is None:
                self.current_model = self.create_neural_network()
            
            # Get optimizer settings
            optimizer_name = self.optim_combo.currentText()
            lr = self.lr_spin.value()
            
            if optimizer_name == "Adam":
                optimizer = optimizers.Adam(learning_rate=lr)
            elif optimizer_name == "SGD":
                optimizer = optimizers.SGD(learning_rate=lr)
            elif optimizer_name == "RMSprop":
                optimizer = optimizers.RMSprop(learning_rate=lr)
            
            # Get learning rate schedule
            lr_schedule = self.lr_sched_combo.currentText()
            if lr_schedule == "Step Decay":
                lr_schedule = optimizers.schedules.ExponentialDecay(
                    initial_learning_rate=lr,
                    decay_steps=1000,
                    decay_rate=0.9
                )
                optimizer = optimizers.Adam(learning_rate=lr_schedule)
            elif lr_schedule == "Exponential":
                lr_schedule = optimizers.schedules.ExponentialDecay(
                    initial_learning_rate=lr,
                    decay_steps=1000,
                    decay_rate=0.9
                )
                optimizer = optimizers.Adam(learning_rate=lr_schedule)
            
            # Compile model
            self.current_model.compile(
                optimizer=optimizer,
                loss=self.current_loss,
                metrics=['accuracy'] if self.current_loss in [
                    'categorical_crossentropy', 
                    'binary_crossentropy', 
                    'hinge'
                ] else ['mse', 'mae']
            )
            
            # Create callbacks
            callbacks = []
            
            # Add progress callback
            class ProgressCallback(tf.keras.callbacks.Callback):
                def __init__(self, progress_bar):
                    super().__init__()
                    self.progress_bar = progress_bar
                    
                def on_epoch_end(self, epoch, logs=None):
                    progress = int(((epoch + 1) / self.params['epochs']) * 100)
                    QApplication.processEvents()  # Process pending events
                    self.progress_bar.setValue(progress)
                    QApplication.processEvents()  # Process pending events
            
            callbacks.append(ProgressCallback(self.progress_bar))
            
            # Add early stopping
            callbacks.append(tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=5,
                restore_best_weights=True
            ))
            
            # Train model
            history = self.current_model.fit(
                self.X_train, self.y_train,
                batch_size=self.batch_size_spin.value(),
                epochs=self.epochs_spin.value(),
                validation_split=0.2,
                callbacks=callbacks,
                verbose=1
            )
            
            self.training_history = history.history
            self.status_bar.showMessage("Training complete!")
            self.progress_bar.setValue(100)  # Ensure progress bar reaches 100%
            QApplication.processEvents()  # Process pending events
            
        except Exception as e:
            self.show_error(f"Error training model: {str(e)}")
            logger.error(f"Training error: {str(e)}")
            self.progress_bar.setValue(0)  # Reset progress bar on error
            QApplication.processEvents()  # Process pending events

    def evaluate_advanced_model(self):
        """Evaluate the model on test data"""
        if self.current_model is None:
            self.show_error("No model to evaluate!")
            return
            
        try:
            # Get predictions
            y_pred = self.current_model.predict(self.X_test)
            
            # Calculate metrics
            if len(np.unique(self.y_test)) <= 10:  # Classification
                from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
                y_pred_classes = np.argmax(y_pred, axis=1) if y_pred.ndim > 1 else y_pred
                y_test_classes = np.argmax(self.y_test, axis=1) if self.y_test.ndim > 1 else self.y_test
                
                accuracy = accuracy_score(y_test_classes, y_pred_classes)
                f1 = f1_score(y_test_classes, y_pred_classes, average='weighted')
                conf_matrix = confusion_matrix(y_test_classes, y_pred_classes)
                
                # Show results
                msg = f"Test Accuracy: {accuracy:.4f}\nF1 Score: {f1:.4f}\n\nConfusion Matrix:\n{conf_matrix}"
                QMessageBox.information(self, "Evaluation Results", msg)
            else:  # Regression
                from sklearn.metrics import mean_squared_error, r2_score
                mse = mean_squared_error(self.y_test, y_pred)
                r2 = r2_score(self.y_test, y_pred)
                
                msg = f"Test MSE: {mse:.4f}\nR² Score: {r2:.4f}"
                QMessageBox.information(self, "Evaluation Results", msg)
                
        except Exception as e:
            self.show_error(f"Error evaluating model: {str(e)}")

    def plot_training_curves(self):
        """Plot training and validation curves"""
        if not hasattr(self, 'training_history'):
            self.show_error("No training history available!")
            return
            
        try:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
            
            # Plot loss
            ax1.plot(self.training_history['loss'], label='Training Loss')
            if 'val_loss' in self.training_history:
                ax1.plot(self.training_history['val_loss'], label='Validation Loss')
            ax1.set_title('Loss Curves')
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Loss')
            ax1.legend()
            
            # Plot accuracy/metrics
            if 'accuracy' in self.training_history:
                ax2.plot(self.training_history['accuracy'], label='Training Accuracy')
                if 'val_accuracy' in self.training_history:
                    ax2.plot(self.training_history['val_accuracy'], label='Validation Accuracy')
                ax2.set_title('Accuracy Curves')
                ax2.set_xlabel('Epoch')
                ax2.set_ylabel('Accuracy')
            else:
                ax2.plot(self.training_history['mse'], label='Training MSE')
                if 'val_mse' in self.training_history:
                    ax2.plot(self.training_history['val_mse'], label='Validation MSE')
                ax2.set_title('MSE Curves')
                ax2.set_xlabel('Epoch')
                ax2.set_ylabel('MSE')
            ax2.legend()
            
            plt.tight_layout()
            plt.show()
            
        except Exception as e:
            self.show_error(f"Error plotting training curves: {str(e)}")

    def plot_gradient_histograms(self):
        """Plot histograms of weight gradients during training"""
        if self.current_model is None:
            self.show_error("No model available!")
            return
            
        try:
            # Get gradients for each layer
            gradients = []
            layer_names = []
            
            for layer in self.current_model.layers:
                if layer.trainable_weights:
                    weights = layer.get_weights()
                    if weights:
                        gradients.extend([w.flatten() for w in weights])
                        layer_names.extend([f"{layer.name} - {i}" for i in range(len(weights))])
            
            # Plot histograms
            n_layers = len(gradients)
            n_cols = min(3, n_layers)
            n_rows = (n_layers + n_cols - 1) // n_cols
            
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4*n_rows))
            axes = axes.flatten()
            
            for i, (grad, name) in enumerate(zip(gradients, layer_names)):
                if i < len(axes):
                    axes[i].hist(grad, bins=50)
                    axes[i].set_title(f"Gradients: {name}")
                    axes[i].set_xlabel("Gradient Value")
                    axes[i].set_ylabel("Frequency")
            
            # Hide empty subplots
            for i in range(len(gradients), len(axes)):
                axes[i].set_visible(False)
            
            plt.tight_layout()
            plt.show()
            
        except Exception as e:
            self.show_error(f"Error plotting gradient histograms: {str(e)}")

    # GAN Models
    class Generator(tf.keras.Model):
        def __init__(self, latent_dim, output_dim):
            super(Generator, self).__init__()
            self.model = tf.keras.Sequential([
                layers.Dense(256, use_bias=False, input_shape=(latent_dim,)),
                layers.BatchNormalization(),
                layers.LeakyReLU(),
                layers.Dense(512),
                layers.BatchNormalization(),
                layers.LeakyReLU(),
                layers.Dense(1024),
                layers.BatchNormalization(),
                layers.LeakyReLU(),
                layers.Dense(output_dim, activation='tanh')
            ])

        def call(self, inputs):
            return self.model(inputs)

    class Discriminator(tf.keras.Model):
        def __init__(self, input_dim):
            super(Discriminator, self).__init__()
            self.model = tf.keras.Sequential([
                layers.Dense(512, input_shape=(input_dim,)),
                layers.LeakyReLU(),
                layers.Dropout(0.3),
                layers.Dense(256),
                layers.LeakyReLU(),
                layers.Dropout(0.3),
                layers.Dense(1, activation='sigmoid')
            ])

    def call(self, inputs):
        return self.model(inputs)

class GANTrainingThread(QThread):
    progress_updated = pyqtSignal(int)
    training_log = pyqtSignal(str)
    training_finished = pyqtSignal()

    def __init__(self, generator, discriminator, dataset, latent_dim, num_epochs, batch_size):
        super().__init__()
        self.generator = generator
        self.discriminator = discriminator
        self.dataset = dataset
        self.latent_dim = latent_dim
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.is_running = True

    def stop(self):
        self.is_running = False

    def run(self):
        # Optimizers
        g_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5)
        d_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5)

        # Loss function
        cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=False)

        for epoch in range(self.num_epochs):
            if not self.is_running:
                break

            for batch_idx, batch in enumerate(self.dataset):
                if not self.is_running:
                    break
                batch_size = tf.shape(batch)[0]

                # Train Discriminator
                with tf.GradientTape() as d_tape:
                    # Generate fake data
                    noise = tf.random.normal([batch_size, self.latent_dim])
                    generated_data = self.generator(noise, training=False)

                    # Get discriminator outputs
                    real_output = self.discriminator(batch, training=True)
                    fake_output = self.discriminator(generated_data, training=True)

                    # Calculate losses
                    d_loss_real = cross_entropy(tf.ones_like(real_output), real_output)
                    d_loss_fake = cross_entropy(tf.zeros_like(fake_output), fake_output)
                    d_loss = d_loss_real + d_loss_fake

                # Update discriminator
                d_gradients = d_tape.gradient(d_loss, self.discriminator.trainable_variables)
                d_optimizer.apply_gradients(zip(d_gradients, self.discriminator.trainable_variables))

                # Train Generator
                with tf.GradientTape() as g_tape:
                    # Generate fake data
                    noise = tf.random.normal([batch_size, self.latent_dim])
                    generated_data = self.generator(noise, training=True)
                    fake_output = self.discriminator(generated_data, training=False)

                    # Calculate loss
                    g_loss = cross_entropy(tf.ones_like(fake_output), fake_output)

                # Update generator
                g_gradients = g_tape.gradient(g_loss, self.generator.trainable_variables)
                g_optimizer.apply_gradients(zip(g_gradients, self.generator.trainable_variables))

                # Log progress
                if batch_idx % 100 == 0:
                    log_msg = f'Epoch [{epoch}/{self.num_epochs}] Batch [{batch_idx}] ' \
                             f'D_loss: {d_loss:.4f} G_loss: {g_loss:.4f}'
                    self.training_log.emit(log_msg)
                    logger.info(log_msg)

            progress = int((epoch + 1) / self.num_epochs * 100)
            self.progress_updated.emit(progress)


    def create_gan_tab(self):
        """Create the GAN tab with training controls and visualization"""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # GAN Configuration
        config_group = QGroupBox("GAN Configuration")
        config_layout = QGridLayout()

        # Latent dimension
        config_layout.addWidget(QLabel("Latent Dimension:"), 0, 0)
        self.latent_dim_spin = QSpinBox()
        self.latent_dim_spin.setRange(10, 1000)
        self.latent_dim_spin.setValue(100)
        config_layout.addWidget(self.latent_dim_spin, 0, 1)

        # Number of epochs
        config_layout.addWidget(QLabel("Epochs:"), 1, 0)
        self.gan_epochs_spin = QSpinBox()
        self.gan_epochs_spin.setRange(1, 1000)
        self.gan_epochs_spin.setValue(100)
        config_layout.addWidget(self.gan_epochs_spin, 1, 1)

        # Batch size
        config_layout.addWidget(QLabel("Batch Size:"), 2, 0)
        self.gan_batch_size_spin = QSpinBox()
        self.gan_batch_size_spin.setRange(1, 1000)
        self.gan_batch_size_spin.setValue(64)
        config_layout.addWidget(self.gan_batch_size_spin, 2, 1)

        config_group.setLayout(config_layout)
        layout.addWidget(config_group)

        # Training Controls
        controls_group = QGroupBox("Training Controls")
        controls_layout = QHBoxLayout()

        self.train_gan_btn = QPushButton("Train GAN")
        self.train_gan_btn.clicked.connect(self.start_gan_training)
        controls_layout.addWidget(self.train_gan_btn)

        self.stop_gan_btn = QPushButton("Stop Training")
        self.stop_gan_btn.clicked.connect(self.stop_gan_training)
        self.stop_gan_btn.setEnabled(False)
        controls_layout.addWidget(self.stop_gan_btn)

        controls_group.setLayout(controls_layout)
        layout.addWidget(controls_group)

        # Training Log
        log_group = QGroupBox("Training Log")
        log_layout = QVBoxLayout()
        self.gan_log_text = QTextEdit()
        self.gan_log_text.setReadOnly(True)
        log_layout.addWidget(self.gan_log_text)
        log_group.setLayout(log_layout)
        layout.addWidget(log_group)

        # Visualization
        viz_group = QGroupBox("Generated Samples")
        viz_layout = QVBoxLayout()
        self.gan_figure = plt.figure(figsize=(6, 4))
        self.gan_canvas = FigureCanvas(self.gan_figure)
        viz_layout.addWidget(self.gan_canvas)
        viz_group.setLayout(viz_layout)
        layout.addWidget(viz_group)

        return widget

    def start_gan_training(self):
        """Start GAN training"""
        if self.X_train is None:
            self.show_error("Please load a dataset first")
            return

        try:
            # Initialize models
            input_dim = self.X_train.shape[1]
            latent_dim = self.latent_dim_spin.value()
            # Create generator and discriminator instances
            self.generator = Generator(latent_dim, input_dim)
            self.discriminator = Discriminator(input_dim)
            # Prepare data
            dataset = tf.data.Dataset.from_tensor_slices(self.X_train)
            # Start training thread
            self.gan_thread = MLCourseGUI.GANTrainingThread(
                self.generator,
                self.discriminator,
                dataset,
                latent_dim,
                self.gan_epochs_spin.value(),
                self.gan_batch_size_spin.value()
            )
            self.gan_thread.progress_updated.connect(self.update_gan_progress)
            self.gan_thread.training_log.connect(self.update_gan_log)
            self.gan_thread.training_finished.connect(self.gan_training_finished)
            self.train_gan_btn.setEnabled(False)
            self.stop_gan_btn.setEnabled(True)
            self.gan_thread.start()
        except Exception as e:
            self.show_error(f"Error starting GAN training: {str(e)}")
            logger.error(f"GAN training error: {str(e)}")

    def stop_gan_training(self):
        """Stop GAN training"""
        if hasattr(self, 'gan_thread'):
            self.gan_thread.stop()
            self.gan_thread.wait()
            self.gan_training_finished()

    def update_gan_progress(self, value):
        """Update progress bar during GAN training"""
        self.progress_bar.setValue(value)

    def update_gan_log(self, message):
        """Update training log during GAN training"""
        self.gan_log_text.append(message)
        logger.info(message)

    def gan_training_finished(self):
        """Handle GAN training completion"""
        self.train_gan_btn.setEnabled(True)
        self.stop_gan_btn.setEnabled(False)
        self.progress_bar.setValue(0)
        # Generate and display samples
        self.generate_gan_samples()

    def generate_gan_samples(self):
        """Generate and display samples from trained GAN"""
        if not hasattr(self, 'generator'):
            return
        try:
            # Generate samples
            noise = tf.random.normal([16, self.latent_dim_spin.value()])
            samples = self.generator(noise, training=False).numpy()
            # Visualize samples
            self.gan_figure.clear()
            ax = self.gan_figure.add_subplot(111)
            if samples.shape[1] > 2:
                # Use PCA for visualization if data is high-dimensional
                pca = PCA(n_components=2)
                samples_2d = pca.fit_transform(samples)
                ax.scatter(samples_2d[:, 0], samples_2d[:, 1])
                ax.set_title("Generated Samples (PCA projection)")
            else:
                ax.scatter(samples[:, 0], samples[:, 1])
                ax.set_title("Generated Samples")
            self.gan_canvas.draw()
        except Exception as e:
            self.show_error(f"Error generating samples: {str(e)}")
            logger.error(f"Sample generation error: {str(e)}")


        try:
            # Initialize models
            input_dim = self.X_train.shape[1]
            latent_dim = self.latent_dim_spin.value()
            
            # Create generator and discriminator instances
            self.generator = Generator(latent_dim, input_dim)
            self.discriminator = Discriminator(input_dim)

            # Prepare data
            dataset = tf.data.Dataset.from_tensor_slices(self.X_train)

            # Start training thread
            self.gan_thread = MLCourseGUI.GANTrainingThread(
                self.generator,
                self.discriminator,
                dataset,
                latent_dim,
                self.gan_epochs_spin.value(),
                self.gan_batch_size_spin.value()
            )

            self.gan_thread.progress_updated.connect(self.update_gan_progress)
            self.gan_thread.training_log.connect(self.update_gan_log)
            self.gan_thread.training_finished.connect(self.gan_training_finished)

            self.train_gan_btn.setEnabled(False)
            self.stop_gan_btn.setEnabled(True)
            self.gan_thread.start()

        except Exception as e:
            self.show_error(f"Error starting GAN training: {str(e)}")
            logger.error(f"GAN training error: {str(e)}")

    def stop_gan_training(self):
        """Stop GAN training"""
        if hasattr(self, 'gan_thread'):
            self.gan_thread.stop()
            self.gan_thread.wait()
            self.gan_training_finished()

    def update_gan_progress(self, value):
        """Update progress bar during GAN training"""
        self.progress_bar.setValue(value)

    def update_gan_log(self, message):
        """Update training log during GAN training"""
        self.gan_log_text.append(message)
        logger.info(message)

    def gan_training_finished(self):
        """Handle GAN training completion"""
        self.train_gan_btn.setEnabled(True)
        self.stop_gan_btn.setEnabled(False)
        self.progress_bar.setValue(0)
        
        # Generate and display samples
        self.generate_gan_samples()

    def generate_gan_samples(self):
        """Generate and display samples from trained GAN"""
        if not hasattr(self, 'generator'):
            return

        try:
            # Generate samples
            noise = tf.random.normal([16, self.latent_dim_spin.value()])
            samples = self.generator(noise, training=False).numpy()

            # Visualize samples
            self.gan_figure.clear()
            ax = self.gan_figure.add_subplot(111)
            
            if samples.shape[1] > 2:
                # Use PCA for visualization if data is high-dimensional
                pca = PCA(n_components=2)
                samples_2d = pca.fit_transform(samples)
                ax.scatter(samples_2d[:, 0], samples_2d[:, 1])
                ax.set_title("Generated Samples (PCA projection)")
            else:
                ax.scatter(samples[:, 0], samples[:, 1])
                ax.set_title("Generated Samples")

            self.gan_canvas.draw()
            
        except Exception as e:
            self.show_error(f"Error generating samples: {str(e)}")
            logger.error(f"Sample generation error: {str(e)}")

class PlotDialog(QDialog):
    def __init__(self, fig, title="Plot", metrics=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        layout = QVBoxLayout(self)
        canvas = FigureCanvas(fig)
        layout.addWidget(canvas)
        if metrics:
            for m in metrics:
                layout.addWidget(QLabel(m))
        self.setLayout(layout)

def main():
    app = QApplication(sys.argv)
    window = MLCourseGUI()
    window.show()
    sys.exit(app.exec())

if __name__ == '__main__':
    main()
