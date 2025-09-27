# In train.py

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import time

# This `if __name__ == '__main__':` block is crucial for Windows compatibility
# when using multiprocessing for data loading.
if __name__ == '__main__':
    print("PyTorch Version: ", torch.__version__)
    print("Starting model training script...")

    # --- 1. Define Paths and Transformations ---
    train_dir = 'chest_xray/train'
    test_dir = 'chest_xray/test'

    # CORRECTED: This block fixes the SyntaxError and adds data augmentation
    # to the training set to make the model more robust.
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]),
    }

    # --- 2. Load Data and Create Dataloaders ---
    print("\nLoading datasets...")
    image_datasets = {x: datasets.ImageFolder(os.path.join(globals()[f"{x}_dir"]), data_transforms[x])
                      for x in ['train', 'test']}

    # Set num_workers to 0 for best Windows compatibility.
    dataloaders = {x: DataLoader(image_datasets[x], batch_size=32, shuffle=True, num_workers=0)
                   for x in ['train', 'test']}

    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'test']}
    class_names = image_datasets['train'].classes

    print(f"Class names: {class_names}")
    print(f"Training dataset size: {dataset_sizes['train']}")
    print(f"Test dataset size: {dataset_sizes['test']}")

    # --- 3. Handle Class Imbalance with Class Weights ---
    print("\nCalculating class weights to handle data imbalance...")
    train_normal_count = len(os.listdir(os.path.join(train_dir, 'NORMAL')))
    train_pneumonia_count = len(os.listdir(os.path.join(train_dir, 'PNEUMONIA')))
    total_train_samples = train_normal_count + train_pneumonia_count

    # Formula: n_samples / (n_classes * n_samples_j)
    weight_for_normal = total_train_samples / (2.0 * train_normal_count)
    weight_for_pneumonia = total_train_samples / (2.0 * train_pneumonia_count)

    class_weights = torch.tensor([weight_for_normal, weight_for_pneumonia])
    print(f"Calculated weights -> Normal: {weight_for_normal:.2f}, Pneumonia: {weight_for_pneumonia:.2f}")

    # --- 4. Set up Model, Loss, and Optimizer ---
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = models.resnet18(weights='IMAGENET1K_V1')

    for param in model.parameters():
        param.requires_grad = False

    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, len(class_names))

    model = model.to(device)
    class_weights = class_weights.to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.fc.parameters(), lr=0.001)

    # --- 5. Training Loop ---
    def train_model(model, criterion, optimizer, num_epochs=10):
        since = time.time()
        print("\n--- Starting Training ---")

        for epoch in range(num_epochs):
            print(f'Epoch {epoch+1}/{num_epochs}')
            print('-' * 10)

            for phase in ['train', 'test']:
                if phase == 'train':
                    model.train()
                else:
                    model.eval()

                running_loss = 0.0
                running_corrects = 0

                # Iterate over data.
                for inputs, labels in dataloaders[phase]:
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    optimizer.zero_grad()

                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)

                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)

                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_acc = running_corrects.double() / dataset_sizes[phase]

                print(f'{phase.capitalize()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

        time_elapsed = time.time() - since
        print(f'\nTraining complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
        return model

    # Train the model
    model_ft = train_model(model, criterion, optimizer, num_epochs=10)

    # --- 6. Save the Trained Model ---
    print("\n--- Saving Model ---")
    torch.save(model_ft.state_dict(), 'pneumonia_classifier.pth')
    print("Model saved successfully to pneumonia_classifier.pth")