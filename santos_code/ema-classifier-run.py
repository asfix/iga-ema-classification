import os
import matplotlib.pyplot as plt
import numpy as np
from ema_classifier import EMAClassifier
from sklearn.metrics import confusion_matrix  # Add this import

# Define the data path - modify this to your dataset location
DATA_PATH = "../data/ema2_4classseperation_5CV/train"

# Check if the path exists
if not os.path.exists(DATA_PATH):
    print(f"Error: The specified path '{DATA_PATH}' does not exist.")
    print("Please update the DATA_PATH variable to the correct location of your dataset.")
    exit(1)

print(f"Starting EMA classification on dataset at: {DATA_PATH}")

# Initialize the classifier
classifier = EMAClassifier(data_path=DATA_PATH)

# Load the images
classifier.load_images()

# Extract features
classifier.extract_features()

# Train and evaluate using 5-fold cross-validation
accuracy, all_accuracies, all_cms, avg_sensitivity, avg_specificity = classifier.train_and_evaluate(n_folds=5)

# Visualize results
plt.figure(figsize=(10, 6))
plt.bar(range(1, len(all_accuracies) + 1), all_accuracies)
plt.xlabel('Fold')
plt.ylabel('Accuracy')
plt.title('Accuracy across 5-fold Cross-Validation')
plt.ylim(0, 1.0)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('ema_accuracy_results.png')
print(f"Results visualization saved to ema_accuracy_results.png")

# Plot per-class metrics for each class
if len(classifier.class_names) > 1:
    metrics_by_class = {}

    # Calculate metrics for each class from the overall confusion matrix
    for class_idx in range(len(classifier.class_names)):
        # Create a binary confusion matrix for this class
        class_name = classifier.class_names[class_idx]

        # Create a per-class metrics visualization
        plt.figure(figsize=(12, 6))

        # Set up subplots
        plt.subplot(1, 2, 1)
        plt.title(f'Sensitivity for Class: {class_name}')

        # Track metrics across folds for this class
        fold_sensitivities = []
        fold_specificities = []

        # For each fold's confusion matrix
        for cm in all_cms:
            # Convert multi-class confusion matrix to binary for this class
            binary_y_true = []
            binary_y_pred = []

            # Reconstruct original length predictions from confusion matrix
            for true_class in range(len(classifier.class_names)):
                for pred_class in range(len(classifier.class_names)):
                    count = cm[true_class, pred_class]
                    binary_y_true.extend([1 if true_class == class_idx else 0] * count)
                    binary_y_pred.extend([1 if pred_class == class_idx else 0] * count)

            # Calculate binary metrics
            binary_cm = confusion_matrix(binary_y_true, binary_y_pred, labels=[0, 1])
            if binary_cm.shape == (2, 2):  # Ensure proper 2x2 shape
                tn, fp, fn, tp = binary_cm.ravel()
                sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
                specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
                fold_sensitivities.append(sensitivity)
                fold_specificities.append(specificity)

        # Plot fold-wise sensitivities
        plt.bar(range(1, len(fold_sensitivities) + 1), fold_sensitivities)
        plt.ylim(0, 1.0)
        plt.xlabel('Fold')
        plt.ylabel('Sensitivity')
        plt.grid(axis='y', linestyle='--', alpha=0.7)

        # Plot fold-wise specificities
        plt.subplot(1, 2, 2)
        plt.title(f'Specificity for Class: {class_name}')
        plt.bar(range(1, len(fold_specificities) + 1), fold_specificities)
        plt.ylim(0, 1.0)
        plt.xlabel('Fold')
        plt.ylabel('Specificity')
        plt.grid(axis='y', linestyle='--', alpha=0.7)

        plt.tight_layout()
        plt.savefig(f'ema_metrics_class_{class_name}.png')

        # Store average metrics
        metrics_by_class[class_name] = {
            'avg_sensitivity': np.mean(fold_sensitivities),
            'avg_specificity': np.mean(fold_specificities)
        }

    # Print final per-class metrics
    print("\nPer-class average metrics:")
    for class_name, metrics in metrics_by_class.items():
        print(f"Class '{class_name}':")
        print(f"  Average Sensitivity: {metrics['avg_sensitivity']:.4f}")
        print(f"  Average Specificity: {metrics['avg_specificity']:.4f}")

# Calculate overall confusion matrix
if len(all_cms) > 0:
    overall_cm = sum(all_cms)

    # For binary classification
    if overall_cm.shape == (2, 2):
        tn, fp, fn, tp = overall_cm.ravel()
        sensitivity = tp / (tp + fn)
        specificity = tn / (tn + fp)

        print("\nFinal performance metrics:")
        print(f"Average accuracy: {accuracy:.4f}")
        print(f"Binary sensitivity: {sensitivity:.4f}")
        print(f"Binary specificity: {specificity:.4f}")
        print(f"Overall confusion matrix:\n{overall_cm}")
    else:
        print("\nFinal performance metrics:")
        print(f"Average accuracy: {accuracy:.4f}")
        print(f"Weighted sensitivity: {avg_sensitivity:.4f}")
        print(f"Weighted specificity: {avg_specificity:.4f}")
        print(f"Overall confusion matrix:\n{overall_cm}")

print("\nClassification completed successfully!")