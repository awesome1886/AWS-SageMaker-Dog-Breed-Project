import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
import argparse
import os
import sys

# --- ADD THESE TWO LINES HERE ---
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
# --------------------------------

# Import smdebug for the Debugger/Profiler requirements
import smdebug.pytorch as smd

def test(model, test_loader, criterion, device, hook):
    print("Testing Model on Whole Testing Dataset")
    model.eval()
    
    # Set debugger hook mode to EVAL
    if hook:
        hook.set_mode(smd.modes.EVAL)
        
    running_loss = 0
    running_corrects = 0
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data).item()

    total_loss = running_loss / len(test_loader.dataset)
    total_acc = running_corrects / len(test_loader.dataset)
    print(f"Testing Accuracy: {100*total_acc:.2f}%, Testing Loss: {total_loss:.4f}")

def train(model, train_loader, validation_loader, criterion, optimizer, device, hook):
    epochs = 2
    best_loss = 1e6
    image_dataset = {'train':train_loader, 'valid':validation_loader}
    loss_counter = 0
    
    # Set debugger hook mode to TRAIN
    if hook:
        hook.set_mode(smd.modes.TRAIN)
    
    for epoch in range(epochs):
        print(f"Epoch: {epoch}")
        for phase in ['train', 'valid']:
            if phase == 'train':
                model.train()
                if hook: hook.set_mode(smd.modes.TRAIN)
            else:
                model.eval()
                if hook: hook.set_mode(smd.modes.EVAL)
            
            running_loss = 0.0
            running_corrects = 0
            
            for inputs, labels in image_dataset[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                if phase == 'train':
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                
                _, preds = torch.max(outputs, 1)
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data).item()

            epoch_loss = running_loss / len(image_dataset[phase].dataset)
            epoch_acc = running_corrects / len(image_dataset[phase].dataset)
            
            if phase == 'valid':
                if epoch_loss < best_loss:
                    best_loss = epoch_loss
                else:
                    loss_counter += 1

            print('{} loss: {:.4f}, acc: {:.4f}, best loss: {:.4f}'.format(phase, epoch_loss, epoch_acc, best_loss))
            
        if loss_counter == 1:
            break
    return model

def net():
    # Use ResNet50 for Transfer Learning
    model = models.resnet50(pretrained=True)

    # Freeze all convolutional layers
    for param in model.parameters():
        param.requires_grad = False   

    # Replace the Fully Connected layer
    # The dog dataset has 133 classes
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_features, 256),
        nn.ReLU(),
        nn.Linear(256, 133) 
    )
    return model

def create_data_loaders(data, batch_size):
    train_data_path = os.path.join(data, 'train')
    test_data_path = os.path.join(data, 'test')
    validation_data_path=os.path.join(data, 'valid')

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    train_data = torchvision.datasets.ImageFolder(root=train_data_path, transform=train_transform)
    train_data_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)

    test_data = torchvision.datasets.ImageFolder(root=test_data_path, transform=test_transform)
    test_data_loader  = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True)

    validation_data = torchvision.datasets.ImageFolder(root=validation_data_path, transform=test_transform)
    validation_data_loader  = torch.utils.data.DataLoader(validation_data, batch_size=batch_size, shuffle=True) 
    
    return train_data_loader, test_data_loader, validation_data_loader

def main(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Running on Device {device}")

    # Initialize Model
    model=net()
    model = model.to(device)
    
    # Register SMDebug Hook
    hook = smd.Hook.create_from_json_file()
    hook.register_hook(model)
    
    # Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=args.lr)
    
    # Data Loaders
    train_loader, test_loader, validation_loader = create_data_loaders(args.data_dir, args.batch_size)
    
    # Train
    print("Starting Training...")
    model=train(model, train_loader, validation_loader, criterion, optimizer, device, hook)
    
    # Test
    print("Starting Testing...")
    test(model, test_loader, criterion, device, hook)
    
    # Save the Model
    print("Saving Model...")
    torch.save(model.cpu().state_dict(), os.path.join(args.model_dir, "model.pth"))

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    
    # Hyperparameters sent by SageMaker
    parser.add_argument('--batch-size', type=int, default=64, metavar='N', help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N', help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=2, metavar='N', help='number of epochs to train (default: 1)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR', help='learning rate (default: 0.01)')
    
    # Container Environment Variables
    parser.add_argument('--data-dir', type=str, default=os.environ['SM_CHANNEL_TRAINING'])
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    
    args=parser.parse_args()
    
    main(args)