import sys
import numpy as np
import pandas as pd
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                           QHBoxLayout, QTabWidget, QPushButton, QLabel, 
                           QComboBox, QFileDialog, QSpinBox, QDoubleSpinBox,
                           QGroupBox, QScrollArea, QTextEdit, QStatusBar,
                           QProgressBar, QCheckBox, QGridLayout, QMessageBox,
                           QDialog, QLineEdit)
from PyQt6.QtCore import Qt
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
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers

class MLCourseGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Machine Learning Course GUI")
        self.setGeometry(100, 100, 1400, 800)
        
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
        
        # Scaling options in a compact layout
        scaling_layout = QHBoxLayout()
        scaling_layout.setSpacing(5)
        scaling_label = QLabel("Scaling:")
        scaling_label.setFixedWidth(50)
        self.scaling_combo = QComboBox()
        self.scaling_combo.addItems(["No Scaling", "Standard Scaling", "Min-Max Scaling", "Robust Scaling"])
        scaling_layout.addWidget(scaling_label)
        scaling_layout.addWidget(self.scaling_combo)
        
        # Test split options in a compact layout
        split_layout = QHBoxLayout()
        split_layout.setSpacing(5)
        split_label = QLabel("Split:")
        split_label.setFixedWidth(50)
        self.split_spin = QSpinBox()
        self.split_spin.setRange(10, 40)
        self.split_spin.setValue(20)
        self.split_spin.setSuffix("%")
        split_layout.addWidget(split_label)
        split_layout.addWidget(self.split_spin)
        
        # Add to left section
        left_section.addLayout(dataset_layout)
        left_section.addWidget(load_btn)
        left_section.addLayout(scaling_layout)
        left_section.addLayout(split_layout)
        
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
            self.figure.tight_layout()
            
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
                model = GaussianNB(
                    var_smoothing=param_widgets['var_smoothing'].value()
                )
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

            # Train model
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
            
            # Split data
            test_size = self.split_spin.value() / 100
            self.X_train, self.X_test, self.y_train, self.y_test = \
                model_selection.train_test_split(X, y, test_size=test_size, random_state=42)
            
            # Apply scaling if selected
            self.apply_scaling()
            
            # Update axis options
            self.update_axis_options()
            
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
                    
                    # Split data
                    test_size = self.split_spin.value() / 100
                    self.X_train, self.X_test, self.y_train, self.y_test = \
                        model_selection.train_test_split(X, y, 
                                                      test_size=test_size, 
                                                      random_state=42)
                    
                    # Apply scaling if selected
                    self.apply_scaling()
                    
                    # Update axis options
                    self.update_axis_options()
                    
                    self.status_bar.showMessage(f"Loaded custom dataset: {file_name}")
                    self.plot_raw_data()
                    
        except Exception as e:
            self.show_error(f"Error loading custom dataset: {str(e)}")

    def train_neural_network(self):
        """Train neural network with current configuration"""
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
            ("Dimensionality Reduction", self.create_dim_reduction_tab),
            ("Reinforcement Learning", self.create_rl_tab)
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
            {"var_smoothing": "double"}
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
            elif isinstance(param_type, list):
                widget = QComboBox()
                widget.addItems(param_type)
            
            param_layout.addWidget(widget)
            param_widgets[param_name] = widget
            layout.addLayout(param_layout)
        
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
        
        # Placeholder for CNN-specific controls
        label = QLabel("CNN Controls (To be implemented)")
        layout.addWidget(label)
        
        group.setLayout(layout)
        return group
    
    def create_rnn_controls(self):
        """Create controls for Recurrent Neural Network"""
        group = QGroupBox("RNN Architecture")
        layout = QVBoxLayout()
        
        # Placeholder for RNN-specific controls
        label = QLabel("RNN Controls (To be implemented)")
        layout.addWidget(label)
        
        group.setLayout(layout)
        return group
    
    def create_neural_network(self):
        """Create neural network based on current configuration"""
        model = models.Sequential()
        
        # Add layers based on configuration
        for layer_config in self.layer_config:
            layer_type = layer_config["type"]
            params = layer_config["params"]
            
            if layer_type == "Dense":
                model.add(layers.Dense(**params))
            elif layer_type == "Conv2D":
                # Add input shape for the first layer
                if len(model.layers) == 0:
                    params['input_shape'] = self.X_train.shape[1:]
                model.add(layers.Conv2D(**params))
            elif layer_type == "MaxPooling2D":
                model.add(layers.MaxPooling2D())
            elif layer_type == "Flatten":
                model.add(layers.Flatten())
            elif layer_type == "Dropout":
                model.add(layers.Dropout(**params))
        
        # Add output layer based on number of classes
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
                self.progress_bar.setValue(progress)
                
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

def main():
    app = QApplication(sys.argv)
    window = MLCourseGUI()
    window.show()
    sys.exit(app.exec())

if __name__ == '__main__':
    main()
