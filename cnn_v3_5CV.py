import os
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, SubsetRandomSampler
from sklearn.model_selection import KFold
from tqdm import tqdm
import cv2
from PIL import Image
from datasets import load_dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_curve, auc
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import wandb
from efficientnet_pytorch import EfficientNet
import torchvision.models as models
import albumentations as A
from albumentations.pytorch import ToTensorV2
import datetime
from torchvision.models import EfficientNet_V2_S_Weights, EfficientNet_V2_M_Weights, EfficientNet_V2_L_Weights

import pandas as pd
from sklearn.model_selection import StratifiedKFold

# Configuration
applyCutmix = False
applyMixup = False
num_epochs = 30
n_folds = 5

# Model variant
# Options: 'efficientnet-b0' to 'efficientnet-b7', 'efficientnetv2_s', 'efficientnetv2_m', 'efficientnetv2_l'
model_variant = 'efficientnetv2_l'
wandb_project_name = "Mehmet-IGA-4Classes-5CV"
data_directory = "../data/ema2_4classseperation_5CV/train"

# Create base experiment directorys
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
base_experiment_name = f"{model_variant}_mixup{applyMixup}_cutmix{applyCutmix}_{timestamp}"
base_experiment_dir = os.path.join("experiments", base_experiment_name)
os.makedirs(base_experiment_dir, exist_ok=True)

# Load the complete dataset
dataset = load_dataset("imagefolder", data_dir=data_directory, split="train")



# Set up label mappings
id2label = {id: label for id, label in enumerate(dataset.features["label"].names)}
label2id = {label: id for id, label in id2label.items()}
print("Classes:", id2label)

# Determine image size based on model
if 'efficientnet-' in model_variant:
    size = EfficientNet.get_image_size(model_variant)
elif 'efficientnetv2' in model_variant:
    size = 640
else:
    size = 224
print("Image size:", size)



def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)
    cx = np.random.randint(W)
    cy = np.random.randint(H)
    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)
    return bbx1, bby1, bbx2, bby2


def cutmix_data(x, y, alpha=1.0):
    """
    Performs CutMix augmentation
    Args:
        x: Input images (batch)
        y: Labels
        alpha: Alpha value for beta distribution
    Returns:
        mixed images, label a, label b, lambda value
    """
    lam = np.random.beta(alpha, alpha)
    batch_size = x.size()[0]

    # Create index tensor on the same device as x
    index = torch.randperm(batch_size, device=x.device)

    bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)
    x[:, :, bbx1:bbx2, bby1:bby2] = x[index, :, bbx1:bbx2, bby1:bby2]

    # Ensure labels are handled correctly
    y_a, y_b = y, y[index]

    # Calculate the new lambda value
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size()[-1] * x.size()[-2]))

    return x, y_a, y_b, lam


def cutmix_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def mixup_data(x, y, alpha=0.2):
    """
    Performs Mixup augmentation
    Args:
        x: Input images (batch)
        y: Labels
        alpha: Alpha value for beta distribution
    Returns:
        mixed images, label a, label b, lambda value
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    batch_size = x.size()[0]

    # Create index tensor on the same device as x
    index = torch.randperm(batch_size, device=x.device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]

    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)



def get_train_transforms(size):
    return A.Compose([
        A.Resize(height=size, width=size),
        A.HorizontalFlip(p=0.4),
        A.VerticalFlip(p=0.4),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.15, rotate_limit=25, p=0.3),
        A.ElasticTransform(alpha=1, sigma=50, p=0.05),
        A.GridDistortion(num_steps=5, distort_limit=0.3, p=0.05),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])


def get_val_transforms(size):
    return A.Compose([
        A.Resize(height=size, width=size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])


def apply_transforms(examples, transforms):
    examples["pixel_values"] = [
        transforms(image=np.array(image.convert("RGB")))["image"]
        for image in examples["image"]
    ]
    return examples


# Set up the transforms
train_transforms = get_train_transforms(size)
val_transforms = get_val_transforms(size)


# [Previous model class implementation remains the same]

class CustomModel(nn.Module):
    def __init__(self, num_classes, variant):
        super(CustomModel, self).__init__()
        if 'efficientnet-' in variant:
            self.base_model = EfficientNet.from_pretrained(variant)
            self.classifier = nn.Linear(self.base_model._fc.in_features, num_classes)
        elif 'efficientnetv2' in variant:
            if variant == 'efficientnetv2_s':
                self.base_model = models.efficientnet_v2_s(weights=EfficientNet_V2_S_Weights.IMAGENET1K_V1)
            elif variant == 'efficientnetv2_m':
                self.base_model = models.efficientnet_v2_m(weights=EfficientNet_V2_M_Weights.IMAGENET1K_V1)
            elif variant == 'efficientnetv2_l':
                self.base_model = models.efficientnet_v2_l(weights=EfficientNet_V2_L_Weights.IMAGENET1K_V1)
            self.classifier = nn.Linear(self.base_model.classifier[1].in_features, num_classes)
            self.base_model.classifier = nn.Identity()
        elif 'densenet' in variant:
            if variant == 'densenet121':
                self.base_model = models.densenet121(pretrained=True)
            elif variant == 'densenet169':
                self.base_model = models.densenet169(pretrained=True)
            elif variant == 'densenet201':
                self.base_model = models.densenet201(pretrained=True)
            self.classifier = nn.Linear(self.base_model.classifier.in_features, num_classes)

    def forward(self, pixel_values):
        if 'efficientnet-' in model_variant:
            features = self.base_model.extract_features(pixel_values)
            features = self.base_model._avg_pooling(features)
            features = features.flatten(start_dim=1)
        elif 'efficientnetv2' in model_variant:
            features = self.base_model.features(pixel_values)
            features = self.base_model.avgpool(features)
            features = features.flatten(start_dim=1)
        else:  # DenseNet
            features = self.base_model.features(pixel_values)
            features = torch.nn.functional.adaptive_avg_pool2d(features, (1, 1))
            features = features.flatten(start_dim=1)
        output = self.classifier(features)
        return output


def collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    labels = torch.tensor([example["label"] for example in examples])
    return {"pixel_values": pixel_values, "labels": labels}

def plot_roc_curve(y_true, y_prob, save_path):
    plt.figure(figsize=(10, 8))

    # Calculate ROC curve and AUC for each class
    n_classes = y_prob.shape[1]
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for i in range(n_classes):
        y_true_binary = (y_true == i).astype(int)
        fpr[i], tpr[i], _ = roc_curve(y_true_binary, y_prob[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

        plt.plot(fpr[i], tpr[i], lw=2,
                 label=f'ROC curve (class {id2label[i]}) (AUC = {roc_auc[i]:.2f})')

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curves')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    return roc_auc


def calculate_metrics(train_loss, val_loss, train_labels, train_preds, train_probs,
                      val_labels, val_preds, val_probs, train_dataloader, val_dataloader,
                      fold, epoch, fold_experiment_dir):
    """Calculate all training and validation metrics"""
    metrics = {}

    # Calculate average losses
    metrics['train_loss'] = train_loss / len(train_dataloader)
    metrics['val_loss'] = val_loss / len(val_dataloader)

    # Calculate accuracy
    metrics['train_accuracy'] = accuracy_score(train_labels, train_preds)
    metrics['val_accuracy'] = accuracy_score(val_labels, val_preds)

    # Calculate precision, recall, F1, and specificity
    train_precision, train_recall, train_f1, _ = precision_recall_fscore_support(
        train_labels, train_preds, average='weighted'
    )
    val_precision, val_recall, val_f1, _ = precision_recall_fscore_support(
        val_labels, val_preds, average='weighted'
    )

    # Calculate specificity (true negative rate)
    def calculate_specificity(y_true, y_pred):
        cm = confusion_matrix(y_true, y_pred)
        specificity = []
        n_classes = len(np.unique(y_true))

        for i in range(n_classes):
            # Create binary confusion matrix for current class
            true_neg = sum(cm[j, k] for j in range(n_classes) for k in range(n_classes)
                           if j != i and k != i)
            false_pos = sum(cm[j, i] for j in range(n_classes) if j != i)

            spec = true_neg / (true_neg + false_pos) if (true_neg + false_pos) > 0 else 0
            specificity.append(spec)

        return np.mean(specificity)  # Return average specificity across all classes

    train_specificity = calculate_specificity(train_labels, train_preds)
    val_specificity = calculate_specificity(val_labels, val_preds)

    metrics['train_precision'] = train_precision
    metrics['train_recall'] = train_recall
    metrics['train_f1'] = train_f1
    metrics['train_specificity'] = train_specificity
    metrics['val_precision'] = val_precision
    metrics['val_recall'] = val_recall
    metrics['val_f1'] = val_f1
    metrics['val_specificity'] = val_specificity

    # Calculate ROC AUC
    train_roc_auc = plot_roc_curve(
        np.array(train_labels),
        np.array(train_probs),
        os.path.join(fold_experiment_dir, f"train_roc_epoch_{epoch + 1}.png")
    )
    val_roc_auc = plot_roc_curve(
        np.array(val_labels),
        np.array(val_probs),
        os.path.join(fold_experiment_dir, f"val_roc_epoch_{epoch + 1}.png")
    )

    metrics['train_auc'] = np.mean(list(train_roc_auc.values()))
    metrics['val_auc'] = np.mean(list(val_roc_auc.values()))

    return metrics

def generate_confusion_matrix(model, dataloader, device, save_path, id2label):
    """Generate and save confusion matrix"""
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            pixel_values = batch["pixel_values"].to(device)
            labels = batch["labels"].to(device)
            outputs = model(pixel_values)
            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(12, 8))
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=list(id2label.values())
    )
    disp.plot(xticks_rotation=45)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def log_final_results(fold_results, base_experiment_dir, config):
    """Calculate and log final cross-validation results"""
    # Initialize wandb run for overall results
    wandb.init(
        project=wandb_project_name,
        name=f"{base_experiment_name}_overall_results",
        config=config,
        reinit=True
    )

    metrics = ["accuracy", "precision", "recall", "specificity", "f1", "auc"]
    final_stats = {}

    for metric in metrics:
        values = [fold[metric] for fold in fold_results]
        mean_value = np.mean(values)
        std_value = np.std(values)
        final_stats[f"{metric}_mean"] = mean_value
        final_stats[f"{metric}_std"] = std_value

        # Log to wandb
        wandb.log({
            f"final/{metric}/mean": mean_value,
            f"final/{metric}/std": std_value
        })

    # Create results DataFrame
    results_df = pd.DataFrame({
        'Metric': metrics,
        'Mean': [final_stats[f"{m}_mean"] for m in metrics],
        'Std': [final_stats[f"{m}_std"] for m in metrics]
    })

    # Save results to CSV
    results_df.to_csv(os.path.join(base_experiment_dir, "final_results.csv"), index=False)

    # Create and save final results plot
    plt.figure(figsize=(14, 6))  # Increased width to accommodate additional metric
    x = np.arange(len(metrics))
    plt.bar(x, results_df['Mean'], yerr=results_df['Std'], capsize=5)
    plt.xticks(x, metrics, rotation=45)
    plt.title('Cross-Validation Results with Standard Deviation')
    plt.ylabel('Score')
    plt.tight_layout()

    final_plot_path = os.path.join(base_experiment_dir, "final_results_plot.png")
    plt.savefig(final_plot_path)
    plt.close()

    # Log plot to wandb
    wandb.log({
        "final_results_plot": wandb.Image(final_plot_path),
        "final_results_table": wandb.Table(dataframe=results_df)
    })

# Initialize K-Fold cross validator
#kfold = KFold(n_splits=n_folds, shuffle=True, random_state=42)


kfold = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
labels = dataset['label']

# Store results for each fold
fold_results = []


# Perform k-fold cross-validation
for fold, (train_idx, val_idx) in enumerate(kfold.split(range(len(dataset)),labels)):
    fold_suffix = f"{fold + 1}{'st' if fold == 0 else 'nd' if fold == 1 else 'rd' if fold == 2 else 'th'}_fold"
    print(f"\nTraining Fold {fold + 1}/{n_folds}")

    # Create fold-specific experiment directory
    fold_experiment_dir = os.path.join(base_experiment_dir, fold_suffix)
    os.makedirs(fold_experiment_dir, exist_ok=True)

    # Initialize wandb for this fold
    if fold > 0:  # Only need to finish previous run if it's not the first fold
        wandb.finish()
    wandb.init(
        project=wandb_project_name,
        name=f"{base_experiment_name}_{fold_suffix}",
        group=base_experiment_name,
        config={
            "model_variant": model_variant,
            "num_epochs": num_epochs,
            "n_folds": n_folds,
            "applyCutmix": applyCutmix,
            "applyMixup": applyMixup,
            "fold": fold + 1
        },
        reinit=True
    )

    # Create fold-specific datasets
    train_subsampler = SubsetRandomSampler(train_idx)
    val_subsampler = SubsetRandomSampler(val_idx)

    # Apply transforms

    # Create separate datasets for training and validation with appropriate transforms
    train_dataset = load_dataset("imagefolder", data_dir=data_directory, split="train")
    val_dataset = load_dataset("imagefolder", data_dir=data_directory, split="train")

    train_dataset.set_transform(lambda examples: apply_transforms(examples, train_transforms))
    val_dataset.set_transform(lambda examples: apply_transforms(examples, val_transforms))



    # Create DataLoaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=4,
        sampler=train_subsampler,
        collate_fn=collate_fn
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=4,
        sampler=val_subsampler,
        collate_fn=collate_fn
    )

    # Initialize model and training components
    num_classes = len(id2label)
    model = CustomModel(num_classes, model_variant)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=2e-5, weight_decay=0.03)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.2)

    # File paths for saving
    best_model_path = os.path.join(fold_experiment_dir, "best_model.pth")
    confusion_matrix_path = os.path.join(fold_experiment_dir, "confusion_matrix.png")
    roc_curve_path = os.path.join(fold_experiment_dir, "roc_curve.png")

    # Variables for saving the best model
    best_accuracy = 0.0
    best_metrics = None

    # Training loop for this fold
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        train_preds = []
        train_probs = []
        train_labels = []

        # Training progress bar
        for batch in tqdm(train_dataloader, desc=f"Fold {fold + 1}, Epoch {epoch + 1}/{num_epochs} - Training"):
            pixel_values = batch["pixel_values"].to(device)
            labels = batch["labels"].to(device)

            if applyCutmix:
                inputs, labels_a, labels_b, lam = cutmix_data(pixel_values, labels)
                outputs = model(inputs)
                loss = cutmix_criterion(criterion, outputs, labels_a, labels_b, lam)
            elif applyMixup:
                inputs, labels_a, labels_b, lam = mixup_data(pixel_values, labels)
                outputs = model(inputs)
                loss = mixup_criterion(criterion, outputs, labels_a, labels_b, lam)
            else:
                outputs = model(pixel_values)
                loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            probs = torch.softmax(outputs, dim=1)
            _, predicted = outputs.max(1)

            train_loss += loss.item()
            train_preds.extend(predicted.cpu().numpy())
            train_probs.extend(probs.detach().cpu().numpy())
            train_labels.extend(labels.cpu().numpy())

        # Validation phase
        model.eval()
        val_loss = 0
        val_preds = []
        val_probs = []
        val_labels = []

        with torch.no_grad():
            for batch in tqdm(val_dataloader, desc=f"Fold {fold + 1}, Epoch {epoch + 1}/{num_epochs} - Validation"):
                pixel_values = batch["pixel_values"].to(device)
                labels = batch["labels"].to(device)

                outputs = model(pixel_values)
                loss = criterion(outputs, labels)

                probs = torch.softmax(outputs, dim=1)
                _, predicted = outputs.max(1)

                val_loss += loss.item()
                val_preds.extend(predicted.cpu().numpy())
                val_probs.extend(probs.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())

        # Calculate metrics
        metrics = calculate_metrics(
            train_loss, val_loss,
            train_labels, train_preds, train_probs,
            val_labels, val_preds, val_probs,
            train_dataloader, val_dataloader,
            fold, epoch, fold_experiment_dir
        )

        # Log metrics to wandb
        wandb.log({
            "train/loss": metrics['train_loss'],
            "train/accuracy": metrics['train_accuracy'],
            "train/precision": metrics['train_precision'],
            "train/recall": metrics['train_recall'],
            "train/specificity": metrics['train_specificity'],
            "train/f1": metrics['train_f1'],
            "train/auc": metrics['train_auc'],
            "eval/loss": metrics['val_loss'],
            "eval/accuracy": metrics['val_accuracy'],
            "eval/precision": metrics['val_precision'],
            "eval/recall": metrics['val_recall'],
            "eval/specificity": metrics['val_specificity'],
            "eval/f1": metrics['val_f1'],
            "eval/auc": metrics['val_auc'],
            "epoch": epoch + 1,
            "fold": fold + 1
        })

        # Save best model
        if metrics['val_accuracy'] > best_accuracy:
            best_accuracy = metrics['val_accuracy']
            best_metrics = {
                "accuracy": metrics['val_accuracy'],
                "precision": metrics['val_precision'],
                "recall": metrics['val_recall'],
                "specificity": metrics['val_specificity'],
                "f1": metrics['val_f1'],
                "auc": metrics['val_auc']
            }
            torch.save(model.state_dict(), best_model_path)
            print(f"New best model saved with accuracy: {best_accuracy:.4f}")

    # Store fold results
    fold_results.append(best_metrics)

    # Load the best model checkpoint for final evaluation
    model.load_state_dict(torch.load(best_model_path))
    model.eval()

    # Generate and save confusion matrix using the best model
    generate_confusion_matrix(
        model, val_dataloader, device,
        confusion_matrix_path, id2label
    )

    # Log confusion matrix to wandb
    wandb.log({
        "confusion_matrix": wandb.Image(confusion_matrix_path)
    })

    print(f"\nCompleted fold {fold + 1}")
    print(f"Best metrics for fold {fold + 1}:")
    for metric, value in best_metrics.items():
        print(f"{metric}: {value:.4f}")

# Calculate and log final results
config = {
    "model_variant": model_variant,
    "num_epochs": num_epochs,
    "n_folds": n_folds,
    "applyCutmix": applyCutmix,
    "applyMixup": applyMixup
}
log_final_results(fold_results, base_experiment_dir, config)

wandb.finish()
print("\nExperiment completed! Results saved to:", base_experiment_dir)
print("Final results and visualizations have been logged to WandB")