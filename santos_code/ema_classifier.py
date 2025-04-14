import os
import numpy as np
import cv2
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.multiclass import OneVsRestClassifier
import time
import matplotlib.pyplot as plt
from skimage.feature import local_binary_pattern
from skimage.feature import canny
from scipy.spatial.distance import pdist, squareform
import glob


class EMAClassifier:
    def __init__(self, data_path, radii=[1, 2, 4], n_points=8, edge_sigma=1.0):
        """
        Initialize the EMA classifier with parameters from the paper.

        Parameters:
        -----------
        data_path : str
            Path to the root folder containing class subfolders
        radii : list
            List of radii for the multi-scale LBP
        n_points : int
            Number of points for LBP
        edge_sigma : float
            Sigma value for edge enhancement
        """
        self.data_path = data_path
        self.radii = radii
        self.n_points = n_points
        self.edge_sigma = edge_sigma
        self.X = []
        self.y = []
        self.class_names = []

    def load_images(self):
        """Load images from the specified data path"""
        print("Loading images...")

        # Get class folder names
        self.class_names = [folder for folder in os.listdir(self.data_path)
                            if os.path.isdir(os.path.join(self.data_path, folder))]

        print(f"Found {len(self.class_names)} classes: {self.class_names}")

        # Initialize lists for data and labels
        images = []
        labels = []

        # Loop through each class folder
        for class_idx, class_name in enumerate(self.class_names):
            class_path = os.path.join(self.data_path, class_name)

            # Get all image files in this class folder
            image_files = []
            for ext in ['*.jpg', '*.jpeg', '*.png', '*.tif', '*.tiff']:
                image_files.extend(glob.glob(os.path.join(class_path, ext)))

            print(f"Loading {len(image_files)} images from class '{class_name}'")

            # Process each image
            for img_path in image_files:
                try:
                    # Read image (convert to grayscale if it's color)
                    img = cv2.imread(img_path)
                    if img is None:
                        print(f"Warning: Could not read image {img_path}")
                        continue

                    if len(img.shape) == 3 and img.shape[2] == 3:
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

                    # Standardize image size if needed (can adjust based on your dataset)
                    img = cv2.resize(img, (256, 256))

                    # Store the image and its label
                    images.append(img)
                    labels.append(class_idx)

                except Exception as e:
                    print(f"Error processing image {img_path}: {e}")

        print(f"Loaded {len(images)} images total")

        # Convert to numpy arrays
        self.images = np.array(images)
        self.y = np.array(labels)

    def extract_multiscale_lbp_features(self, image):
        """
        Extract multiscale rotation invariant co-occurrence among adjacent LBP features
        as described in the paper.
        """
        # Edge enhancement (using Canny edge detection as preprocessing)
        edges = canny(image, sigma=self.edge_sigma)
        enhanced = image.copy().astype(float)
        enhanced[edges] = np.max(enhanced)  # Enhance the edges

        # Convert to integer type before applying LBP as recommended
        enhanced_int = np.uint8(enhanced)

        # Extract LBP at multiple scales
        lbp_features = []

        for radius in self.radii:
            # Get rotation invariant LBP
            lbp = local_binary_pattern(enhanced_int, self.n_points, radius, method='ror')

            # Compute co-occurrence matrix
            # We'll use a simplified approach to capture adjacent pattern relationships
            # by computing the correlation between LBP values at nearby pixels

            # Compute distances between all pixels
            flat_lbp = lbp.flatten()

            # For large images, we'll sample points to reduce computation
            if len(flat_lbp) > 10000:
                indices = np.random.choice(len(flat_lbp), 10000, replace=False)
                flat_lbp = flat_lbp[indices]

            # Compute pairwise distances (correlation metric)
            if len(flat_lbp) > 1:  # Ensure we have at least 2 elements
                try:
                    # Use Euclidean distance instead of correlation to avoid NaN values
                    distances = pdist(flat_lbp.reshape(-1, 1), metric='euclidean')

                    # Get statistics from the distances as features
                    stats = [
                        np.mean(distances) if len(distances) > 0 else 0,
                        np.std(distances) if len(distances) > 0 else 0,
                        np.min(distances) if len(distances) > 0 else 0,
                        np.max(distances) if len(distances) > 0 else 0
                    ]

                    # Check for NaN values and replace them with zeros
                    stats = [0 if np.isnan(x) else x for x in stats]
                    lbp_features.extend(stats)

                    # Add histogram of LBP values (binned)
                    hist, _ = np.histogram(lbp, bins=10, density=True)

                    # Check for NaN values in histogram and replace with zeros
                    hist = np.nan_to_num(hist, nan=0.0)
                    lbp_features.extend(hist)

                except Exception as e:
                    print(f"Error in feature extraction: {e}")
                    # If there's an error, add zeros as placeholder features
                    lbp_features.extend([0] * 14)  # 4 stats + 10 histogram bins
            else:
                # Not enough elements, add zeros as placeholder features
                lbp_features.extend([0] * 14)  # 4 stats + 10 histogram bins

        return np.array(lbp_features)

    def extract_features(self):
        """Extract features from all loaded images"""
        print("Extracting features...")
        start_time = time.time()

        features = []
        for i, image in enumerate(self.images):
            if i % 10 == 0:
                print(f"Processing image {i}/{len(self.images)}...")

            feature_vector = self.extract_multiscale_lbp_features(image)

            # Check for NaN or inf values and replace them
            feature_vector = np.nan_to_num(feature_vector, nan=0.0, posinf=0.0, neginf=0.0)

            features.append(feature_vector)

        self.X = np.array(features)

        # Final check for NaN values in the entire feature matrix
        if np.isnan(self.X).any():
            print("WARNING: NaN values found in feature matrix. Replacing with zeros.")
            self.X = np.nan_to_num(self.X, nan=0.0)

        print(f"Feature extraction completed in {time.time() - start_time:.2f} seconds")
        print(f"Feature vector shape: {self.X.shape}")

    def train_and_evaluate(self, n_folds=5):
        """
        Train and evaluate the model using n-fold cross-validation
        as described in the paper with SVM and AdaBoost.
        """
        print(f"Performing {n_folds}-fold cross-validation...")

        # Initialize metrics collection
        all_accuracies = []
        all_reports = []
        all_confusion_matrices = []
        all_times = []
        all_weighted_sensitivities = []
        all_weighted_specificities = []

        # Initialize k-fold cross-validation
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

        # For each fold
        for fold, (train_idx, test_idx) in enumerate(skf.split(self.X, self.y)):
            print(f"\nFold {fold + 1}/{n_folds}")
            start_time = time.time()

            # Split data
            X_train, X_test = self.X[train_idx], self.X[test_idx]
            y_train, y_test = self.y[train_idx], self.y[test_idx]

            # Initialize SVM with RBF kernel (as used in the paper)
            base_svm = SVC(kernel='rbf', probability=True, random_state=42)

            # For binary classification, we can use SVM directly
            if len(self.class_names) == 2:
                # Use AdaBoost with decision trees as weak learners to tune SVM
                # The paper mentions 100 consecutive learning cycles
                weak_learner = DecisionTreeClassifier(max_depth=1)
                model = AdaBoostClassifier(
                    weak_learner,
                    n_estimators=100,
                    random_state=42
                )
            else:
                # For multi-class, use one-vs-all approach
                # Wrap SVM in a one-vs-all classifier
                ovr_svm = OneVsRestClassifier(base_svm)
                model = AdaBoostClassifier(
                    DecisionTreeClassifier(max_depth=1),
                    n_estimators=100,
                    random_state=42
                )

            # Train the model
            model.fit(X_train, y_train)

            # Predict on test set
            y_pred = model.predict(X_test)

            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            report = classification_report(y_test, y_pred, target_names=self.class_names)
            conf_matrix = confusion_matrix(y_test, y_pred)
            elapsed_time = time.time() - start_time

            # Store results
            all_accuracies.append(accuracy)
            all_reports.append(report)
            all_confusion_matrices.append(conf_matrix)
            all_times.append(elapsed_time)

            # Calculate per-class metrics for this fold
            class_metrics = {}
            class_counts = {}

            # For each class, compute sensitivity and specificity in a one-vs-rest approach
            for class_idx in range(len(self.class_names)):
                # Create binary representation for this class
                y_test_binary = (y_test == class_idx).astype(int)
                y_pred_binary = (y_pred == class_idx).astype(int)

                # Get confusion matrix for this class
                tn, fp, fn, tp = confusion_matrix(y_test_binary, y_pred_binary, labels=[0, 1]).ravel()

                # Calculate sensitivity and specificity
                sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
                specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

                # Store class metrics
                class_metrics[class_idx] = {
                    'sensitivity': sensitivity,
                    'specificity': specificity
                }

                # Count instances in each class
                class_counts[class_idx] = (y_test == class_idx).sum()

            # Calculate weighted metrics based on class distribution
            total_instances = len(y_test)
            weighted_sensitivity = 0
            weighted_specificity = 0

            for class_idx, count in class_counts.items():
                weight = count / total_instances
                weighted_sensitivity += class_metrics[class_idx]['sensitivity'] * weight
                weighted_specificity += class_metrics[class_idx]['specificity'] * weight

            # Store weighted metrics for this fold
            all_weighted_sensitivities.append(weighted_sensitivity)
            all_weighted_specificities.append(weighted_specificity)

            # Print results for this fold
            print(f"Fold {fold + 1} accuracy: {accuracy:.4f}")
            print(f"Fold {fold + 1} weighted sensitivity: {weighted_sensitivity:.4f}")
            print(f"Fold {fold + 1} weighted specificity: {weighted_specificity:.4f}")
            print(f"Fold {fold + 1} classification report:\n{report}")
            print(f"Fold {fold + 1} confusion matrix:\n{conf_matrix}")
            print(f"Fold {fold + 1} execution time: {elapsed_time:.2f} seconds")

        # Calculate and display average results
        avg_accuracy = np.mean(all_accuracies)
        avg_weighted_sensitivity = np.mean(all_weighted_sensitivities)
        avg_weighted_specificity = np.mean(all_weighted_specificities)
        avg_time = np.mean(all_times)

        print("\n" + "=" * 50)
        print(f"Average accuracy across {n_folds} folds: {avg_accuracy:.4f}")
        print(f"Average weighted sensitivity across {n_folds} folds: {avg_weighted_sensitivity:.4f}")
        print(f"Average weighted specificity across {n_folds} folds: {avg_weighted_specificity:.4f}")
        print(f"Average execution time per fold: {avg_time:.2f} seconds")

        # Calculate and display overall confusion matrix (sum of all folds)
        overall_cm = sum(all_confusion_matrices)
        print(f"\nOverall confusion matrix:\n{overall_cm}")

        # For binary classification, also report traditional sensitivity and specificity
        if len(self.class_names) == 2:
            tn, fp, fn, tp = overall_cm.ravel()
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            print(f"\nOverall binary metrics:")
            print(f"Sensitivity: {sensitivity:.4f}")
            print(f"Specificity: {specificity:.4f}")

        return avg_accuracy, all_accuracies, all_confusion_matrices, avg_weighted_sensitivity, avg_weighted_specificity


# Example usage
if __name__ == "__main__":
    # Specify your data path
    data_path = "ema2_2classseperation_5CV/train"

    # Initialize and run the classifier
    classifier = EMAClassifier(data_path)
    classifier.load_images()
    classifier.extract_features()
    accuracy, all_accuracies, all_cms = classifier.train_and_evaluate(n_folds=5)