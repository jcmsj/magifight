import torch
from torch import nn
from torchvision import transforms

class SpellClassifier(nn.Module):
    def __init__(self, num_classes):
        super(SpellClassifier, self).__init__()
        pool = lambda: nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.model = nn.Sequential(
            nn.Conv2d(1, 96, kernel_size=3, stride=1, padding=1),
            pool(),
            nn.Conv2d(96, 128, kernel_size=3, stride=1, padding=1),
            pool(),
            nn.Dropout(0.5),
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            pool(),
            # flatten except batch dimension
            nn.Flatten(start_dim=1),
            # create 3 fully connected layers, the first 2 layers have 128 neurons
            nn.Linear(64* 36, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.model(x)

im2Tensor = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
class SpellImageDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.images = []
        self.labels = []
        self.transform = im2Tensor
        self.load_data()

    def load_data(self):
        import os
        from PIL import Image
        for root, _, files in os.walk(self.data_dir):
            for file in files:
                if not file.endswith('.bmp'):
                    continue
                path = os.path.join(root, file)
                image = Image.open(path).convert('L')  # Convert to grayscale
                image = self.transform(image)
                self.images.append(image)
                self.labels.append(int(file.split('.')[-2]))
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]

def train(model, device, data_dir):
    from torch.utils.data import DataLoader
    
    dataset = SpellImageDataset(data_dir)
    train_size = int(0.7 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0008)
    
    model.to(device)
    model.train()
    num_epochs = 200
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # every 10 epochs, print the loss
        if (epoch+1) % 10 == 0:
            # print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item()}')
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}')

    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print(f'Accuracy @ {total} test images: {100 * correct / total} %')
    
    torch.save(model.state_dict(), 'model.ckpt')

def predict(model, device, image):
    model.eval()
    with torch.no_grad():
        image = image.to(device)
        outputs = model(image)
        _, predicted = torch.max(outputs.data, 1)
        return predicted
    


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, help='path to model file')
    parser.add_argument('--save-interval', type=int, default=10, help='save interval')
    parser.add_argument('--input', type=str, help='input file')
    args = parser.parse_args()

    model = SpellClassifier(3)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    if args.model:
        model.load_state_dict(torch.load(args.model, weights_only=True))
        print(f"Loaded model from {args.model}")

    if args.input:
        from PIL import Image
        image = Image.open(args.input).convert('L')
        image = im2Tensor(image)
        image = image.unsqueeze(0)
        print(predict(model, device, image.to(device)))
    else:
        data_dir = "./datasets/spells"
        train(model, device, data_dir)

if __name__ == '__main__':
    main()
