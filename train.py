# train.py – ускоренная версия: torchvision вместо albumentations, num_workers=2
import os
import json
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Dataset
from torchvision import datasets, transforms, models
from tqdm import tqdm

# =============== КОНФИГУРАЦИЯ ===============
SEED = 42
IMG_SIZE = 224
BATCH_SIZE = 64
EPOCHS = 20
LR = 1e-4
WEIGHT_DECAY = 1e-4
NUM_CLASSES = 38
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# =============== 1. ПУТЬ К ЛОКАЛЬНОМУ ДАТАСЕТУ ===============
dataset_path = r"C:\Users\jmtj5\PycharmProjects\PlantDoctor\plantvillage_data"
print("Dataset path:", dataset_path)

data_root = os.path.join(dataset_path, "PlantVillage")
if not os.path.isdir(data_root):
    data_root = dataset_path
print("Data root:", data_root)

# =============== 2. ПОДГОТОВКА ДАННЫХ ===============
# Трансформации на основе torchvision.transforms (быстрее, чем albumentations на CPU)
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Обёртка для применения transform к Subset (random_split возвращает Subset)
class TransformSubset(Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform
    def __len__(self):
        return len(self.subset)
    def __getitem__(self, idx):
        x, y = self.subset[idx]
        if self.transform:
            x = self.transform(x)
        return x, y

full_dataset = datasets.ImageFolder(data_root)

train_size = int(0.7 * len(full_dataset))
val_size = int(0.15 * len(full_dataset))
test_size = len(full_dataset) - train_size - val_size
train_ds, val_ds, test_ds = random_split(full_dataset, [train_size, val_size, test_size])

train_ds = TransformSubset(train_ds, train_transform)
val_ds = TransformSubset(val_ds, val_transform)
test_ds = TransformSubset(test_ds, val_transform)

class_names = full_dataset.classes
with open("class_names.json", "w", encoding='utf-8') as f:
    json.dump(class_names, f, ensure_ascii=False)
print(f"Classes ({len(class_names)}):", class_names)

# =============== 3. ПОСТРОЕНИЕ МОДЕЛЕЙ ===============
def get_model(name):
    if name == 'resnet50':
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        model.fc = nn.Linear(2048, NUM_CLASSES)
    elif name == 'efficientnet_b0':
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        model.classifier[1] = nn.Linear(1280, NUM_CLASSES)
    elif name == 'mobilenet_v3_large':
        model = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.IMAGENET1K_V1)
        model.classifier[3] = nn.Linear(1280, NUM_CLASSES)
    elif name == 'densenet121':
        model = models.densenet121(weights=models.Densenet121_Weights.IMAGENET1K_V1)
        model.classifier = nn.Linear(1024, NUM_CLASSES)
    elif name == 'vit_b_16':
        model = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)
        model.heads.head = nn.Linear(768, NUM_CLASSES)
    elif name == 'convnext_tiny':
        model = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
        model.classifier[2] = nn.Linear(768, NUM_CLASSES)
    else:
        raise ValueError(f"Unknown model: {name}")
    return model.to(DEVICE)

# =============== 4. ЦИКЛ ОБУЧЕНИЯ ===============
def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for images, labels in tqdm(loader, desc='Training', leave=False):
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    return total_loss / total, correct / total

def validate(model, loader, criterion):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    with torch.no_grad():
        for images, labels in tqdm(loader, desc='Validation', leave=False):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return total_loss / total, correct / total

def train_model(model, model_name, train_loader, val_loader):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    best_val_acc = 0.0
    patience = 5
    epochs_no_improve = 0
    for epoch in range(EPOCHS):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion)
        val_loss, val_acc = validate(model, val_loader, criterion)
        scheduler.step()
        print(f"Epoch {epoch+1}/{EPOCHS} | Train loss: {train_loss:.4f} acc: {train_acc:.4f} | Val loss: {val_loss:.4f} acc: {val_acc:.4f}")
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), f"{model_name}_best.pth")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve == patience:
                print("Early stopping")
                break
    model.load_state_dict(torch.load(f"{model_name}_best.pth"))
    return model

# =============== 5. ГЛАВНЫЙ БЛОК (защита для Windows multiprocessing) ===============
if __name__ == '__main__':
    # DataLoader с несколькими рабочими процессами
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    architectures = ['resnet50', 'efficientnet_b0', 'mobilenet_v3_large', 'densenet121', 'vit_b_16', 'convnext_tiny']
    results = {}

    for arch in architectures:
        print(f"\n====== Training {arch} ======")
        model = get_model(arch)
        model = train_model(model, arch, train_loader, val_loader)
        _, test_acc = validate(model, test_loader, nn.CrossEntropyLoss())
        results[arch] = test_acc
        print(f"{arch} test accuracy: {test_acc:.4f}")

    best_arch = max(results, key=results.get)
    print(f"\nBest model: {best_arch} with accuracy {results[best_arch]:.4f}")

    # =============== 6. ЭКСПОРТ ЛУЧШЕЙ МОДЕЛИ В ONNX ===============
    best_model = get_model(best_arch)
    best_model.load_state_dict(torch.load(f"{best_arch}_best.pth"))
    best_model.eval()

    dummy_input = torch.randn(1, 3, IMG_SIZE, IMG_SIZE, device=DEVICE)
    onnx_path = "best_model.onnx"
    torch.onnx.export(best_model, dummy_input, onnx_path,
                      input_names=["input"],
                      output_names=["output"],
                      dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}})
    print(f"Exported best model to {onnx_path}")

    print("\nAll done! Run 'streamlit run app.py' to start the application.")
