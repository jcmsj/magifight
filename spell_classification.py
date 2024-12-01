import torch
from torch import nn
from torchvision import transforms, models
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import (
    confusion_matrix,
    precision_recall_fscore_support,
    accuracy_score,
)
import pandas as pd
import os
from datetime import datetime
import time

class SpellClassifier(nn.Module):
    def __init__(self, num_classes):
        super(SpellClassifier, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 192, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(192),
            # nn.Dropout(0.5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=0),
            nn.Conv2d(192, 128, kernel_size=3, stride=1, padding=1),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            # nn.Dropout(0.5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            # nn.Dropout(0.5),
            nn.Flatten(start_dim=1),
            nn.Linear(128 * 9, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes),
            # for confidence
            nn.Softmax(dim=1),
        )

    def forward(self, x):
        return self.model(x)


# Define transforms for both RGB and grayscale images
rgb_transform = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

# Update the grayscale transform definition
grayscale_transform = transforms.Compose([
    transforms.Resize((48, 48)),  # Resize all images to 48x48
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

class SpellImageDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, color_mode="rgb"):
        self.data_dir = data_dir
        self.images = []
        self.labels = []
        self.color_mode = color_mode
        self.transform = rgb_transform if color_mode == "rgb" else grayscale_transform
        self.load_data()

    def load_data(self):
        import os
        from PIL import Image

        for root, _, files in os.walk(self.data_dir):
            for file in files:
                if not file.endswith(".bmp"):
                    continue
                try:
                    path = os.path.join(root, file)
                    image = Image.open(path)
                    # Convert to grayscale if needed
                    if self.color_mode == "L":
                        image = image.convert("L")
                    # Apply transform
                    image = self.transform(image)
                    self.images.append(image)
                    self.labels.append(int(file.split(".")[-2]))
                except Exception as e:
                    print(f"Error loading {file}: {e}")
                    continue

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]


def calculate_metrics(y_true, y_pred, num_classes):
    """Calculate comprehensive metrics for the model evaluation."""
    conf_matrix = confusion_matrix(y_true, y_pred)

    # Calculate precision, recall, f1 for each class
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred)

    # Calculate accuracy
    acc = accuracy_score(y_true, y_pred)

    # Calculate TP, TN, FP, FN for each class
    tp = np.diag(conf_matrix)
    fp = np.sum(conf_matrix, axis=0) - tp
    fn = np.sum(conf_matrix, axis=1) - tp
    tn = np.sum(conf_matrix) - (fp + fn + tp)

    # Calculate mAP and mAR
    ap = np.mean(precision)
    ar = np.mean(recall)

    metrics = {
        "confusion_matrix": conf_matrix,
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "true_positive": tp,
        "true_negative": tn,
        "false_positive": fp,
        "false_negative": fn,
        "ap": ap,
        "ar": ar,
    }
    return metrics


def validate(model, device, loader, num_classes=5):
    """Run validation only and return metrics."""
    model.eval()
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        accuracy = 100 * correct / total
        print(f"Accuracy @ {total} test images: {accuracy} %")

        metrics = calculate_metrics(
            all_labels, all_predictions, num_classes=num_classes
        )

        # Save metrics with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_dir = "validation_results"
        os.makedirs(results_dir, exist_ok=True)

        # Save results
        conf_df = pd.DataFrame(
            metrics["confusion_matrix"],
            index=[i for i in range(num_classes)],
            columns=[i for i in range(num_classes)],
        )
        conf_df.to_csv(f"{results_dir}/confusion_matrix_{timestamp}.csv")

        metrics_dict = {
            "Class": list(range(num_classes)),
            "Precision": metrics["precision"],
            "Recall": metrics["recall"],
            "F1": metrics["f1"],
            "TP": metrics["true_positive"],
            "TN": metrics["true_negative"],
            "FP": metrics["false_positive"],
            "FN": metrics["false_negative"],
        }

        metrics_df = pd.DataFrame(metrics_dict)
        metrics_df.to_csv(f"{results_dir}/metrics_{timestamp}.csv", index=False)

        global_metrics_df = pd.DataFrame(
            {
                "Metric": ["AP", "AR", "Accuracy", "F1"],
                "Value": [metrics["ap"], metrics["ar"], metrics["accuracy"], np.mean(metrics["f1"])],
            }
        )
        global_metrics_df.to_csv(
            f"{results_dir}/global_metrics_{timestamp}.csv", index=False
        )

        print("\nMetrics saved to CSV files in validation_results directory")
        print("\nConfusion Matrix:")
        print(conf_df)
        print("\nDetailed Metrics:")
        print(metrics_df)
        print("\nGlobal Metrics:")
        print(global_metrics_df)

        return metrics, accuracy

def train(
    model,
    device,
    data_dir,
    model_file_template_name="model.ckpt",
    num_epochs=300,
    color_mode="rgb",
):
    from torch.utils.data import DataLoader

    dataset = SpellImageDataset(data_dir, color_mode=color_mode)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    generator = torch.Generator().manual_seed(42)
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size],
        generator=generator
    )
    train_loader = DataLoader(train_dataset, batch_size=48, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=48, shuffle=False)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)

    epoch_times = []  # New list to store epoch times
    model.to(device)
    model.train()
    for epoch in range(num_epochs):
        epoch_start = time.time()

        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        epoch_time = time.time() - epoch_start
        epoch_times.append(epoch_time)

        if (epoch + 1) % 10 == 0:
            print(
                f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}, Time: {epoch_time:.2f}s"
            )

    metrics, accuracy = validate(model, device, val_loader, num_classes=5)

    # Save epoch times
    epoch_times_df = pd.DataFrame(
        {"Epoch": range(1, num_epochs + 1), "Time_Seconds": epoch_times}
    )
    epoch_times_df.to_csv(
        f"training_results/epoch_times_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        index=False,
    )
    
    torch.save(model.state_dict(), model_file_template_name)


def predict(model, device, image, conf_threshold=0.5):
    model.eval()
    with torch.no_grad():
        image = image.to(device)
        outputs = model(image)
        print(outputs)
        confidence, predicted = torch.max(outputs, 1)
        if confidence.item() >= conf_threshold:
            return predicted.item(), confidence.item()
        return -1, confidence.item()


def get_unique_model_filename(base_template):
    """Generate a unique model filename by adding numeric suffix if needed."""
    name, ext = os.path.splitext(base_template)
    counter = 0
    while True:
        if counter == 0:
            filename = f"{name}{ext}"
        else:
            filename = f"{name}_{counter}{ext}"

        if not os.path.exists(filename):
            return filename
        counter += 1


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, help="path to model file")
    parser.add_argument("--save-interval", type=int, default=10, help="save interval")
    parser.add_argument("--input", type=str, help="input file")
    parser.add_argument(
        "--arch",
        type=str,
        choices=["harrynet", "alexnet", "resnet34"],
        default="harrynet",
        help="model architecture to use",
    )
    parser.add_argument(
        "--epochs", type=int, default=300, help="number of epochs for training"
    )
    parser.add_argument(
        "--num-classes", type=int, default=5, help="number of classes to classify"
    )
    parser.add_argument(
        "--conf", type=float, default=0.5, help="confidence threshold for prediction"
    )
    parser.add_argument("--val-only", action="store_true", help="run validation only")
    parser.add_argument("--val-data", type=str, help="validation data directory")
    args = parser.parse_args()

    # Initialize model based on selected architecture
    if args.arch == "harrynet":
        model = SpellClassifier(args.num_classes)
    elif args.arch == "alexnet":
        model = models.alexnet(num_classes=args.num_classes)
    else:  # resnet34
        model = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
        num_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Linear(num_features, args.num_classes), nn.Softmax(dim=1)
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    if args.model:
        model.load_state_dict(torch.load(args.model, weights_only=True))
        print(f"Loaded model from {args.model}")

    if args.val_only:
        if not args.model:
            print("Error: --model argument is required for validation")
            return
        if not args.val_data:
            print("Error: --val-data argument is required for validation")
            return

        from torch.utils.data import DataLoader

        val_dataset = SpellImageDataset(
            args.val_data, color_mode="L" if args.arch == "harrynet" else "rgb"
        )
        val_loader = DataLoader(val_dataset, batch_size=60, shuffle=False)
        validate(model, device, val_loader, num_classes=args.num_classes)
    elif args.input:
        from PIL import Image

        image = Image.open(args.input)
        # check the loaded model if it is the spell classifier
        if args.arch == "harrynet":
            image = image.convert("L")
        image = im2Tensor(image)
        image = image.unsqueeze(0)
        classification, confidence = predict(
            model, device, image.to(device), conf_threshold=args.conf
        )
        print(f"Predicted class: {classification}, Confidence: {confidence:.2f}")
    else:
        data_dir = "./datasets/spells"
        base_template = f"{args.arch}.ckpt"
        model_template = get_unique_model_filename(base_template)
        train(
            model,
            device,
            data_dir,
            model_file_template_name=model_template,
            num_epochs=args.epochs,
            color_mode="L" if args.arch == "harrynet" else "rgb",
        )


if __name__ == "__main__":
    main()
