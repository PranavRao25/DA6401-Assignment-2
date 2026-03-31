import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb
import os
import json
import pandas as pd
from object_localization import ObjectLocalizer
import yaml
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from augments import resize_pipeline
from dataset import *

parent = Path(__file__).resolve().parent.parent

with open(parent / "train1.yaml", "r") as f:
    args = yaml.full_load(f)

with open(parent / "configs.yaml", "r") as f:
    configs = yaml.full_load(f)

complete_train_path = configs["data"]["complete"]["train"]
incomplete_train_path = configs["data"]["incomplete"]["train"]
complete_valid_path = configs["data"]["complete"]["valid"]
incomplete_valid_path = configs["data"]["incomplete"]["valid"]
complete_test_path = configs["data"]["complete"]["test"]
incomplete_test_path = configs["data"]["incomplete"]["test"]

class TrainObjectLocalization():
    def __init__(self) -> None:
        self.device = args["device"] or "cpu"
        self.batch_size = args["batch_size"] or 32
        self.epochs = args["epochs"] or 50
        self.lr = args["lr"] or 1e-4
        self.wgt_decay = args["wgt_decay"] or 1e-4
        self.dropout = args["drop_out"] or 0.5
        self.model_save_name = args["path"]
        
        # Load dataset
        self.dataset_path = configs["data"]["processed"]

        # Data related parameters
        self.IN_CHANNELS = 3 # e.g., RGB images
        self.NUM_CLASSES = 37 # e.g., Oxford-IIIT Pet dataset

        self.model = ObjectLocalizer(in_channels=self.IN_CHANNELS, num_classes=self.NUM_CLASSES,
                                     output_size=4, p = self.dropout)

        self._initialize_loader()

    def _initialize_loader(self):
            # 3. Data Transformations (TODO: perform in seperate file/class)
        # CRITICAL: Input must be exactly 224x224 for your current linear layer configuration
        # transform = transforms.Compose([
        #     transforms.Resize((224, 224)),
        #     transforms.ToTensor(),
        #     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        # ])
        
        # Example placeholder datasets (replace with your actual Dataset)
        # train_dataset = torchvision.datasets.ImageFolder(root='path/to/train', transform=transform)
        # val_dataset = torchvision.datasets.ImageFolder(root='path/to/val', transform=transform)
        
        # Using dummy data for demonstration
        # class DummyDataset(Dataset):
        #     def __init__(self, num_samples): self.num_samples = num_samples
        #     def __len__(self): return self.num_samples
        #     def __getitem__(self, idx): return torch.randn(3, 224, 224), torch.randint(0, NUM_CLASSES, (1,)).item()
        
        # train_loader = DataLoader(DummyDataset(1000), batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
        # val_loader = DataLoader(DummyDataset(200), batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
        pass

    def train_model(self):
        self.model = self.model.to(self.device)
        criterion = nn.CrossEntropyLoss()  # Classification
        optimizer = optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=3)
        
        for epoch in range(self.epochs):
            self.model.train()
            running_loss = 0.0
            correct_train = 0
            total_train = 0
            val_loss = 0.0
            correct_val = 0
            total_val = 0
            
            loop = tqdm(self.train_loader, desc=f"Epoch [{epoch+1}/{self.epochs}] Train")
            for images, labels in loop:
                images, labels = images.to(self.device), labels.to(self.device)
                
                optimizer.zero_grad(set_to_none=True)
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total_train += labels.size(0)
                correct_train += (predicted == labels).sum().item()
                
                loop.set_postfix(loss=loss.item(), acc=100. * correct_train / total_train)
                
            train_acc = 100. * correct_train / total_train
            train_loss = running_loss / len(self.train_loader)
            
            # Validation Loop
            self.model.eval()
            with torch.no_grad():
                for images, labels in self.val_loader:
                    images, labels = images.to(self.device), labels.to(self.device)
                    outputs = self.model(images)
                    loss = criterion(outputs, labels)
                    
                    val_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    total_val += labels.size(0)
                    correct_val += (predicted == labels).sum().item()
                    
            val_acc = 100. * correct_val / total_val
            avg_val_loss = val_loss / len(self.val_loader)
            
            print(f"Epoch [{epoch+1}/{self.epochs}] Summary: "
                f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | "
                f"Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            
            is_best = val_acc > best_val_acc
            if is_best:
                best_val_acc = val_acc
                
            # Create the state dictionary
            state = {
                'epoch': epoch + 1,
                'state_dict': self.model.state_dict(),
                'best_acc': best_val_acc,
                'optimizer': optimizer.state_dict(),
                # 'scaler': scaler.state_dict() # Save AMP scaler state
            }
            
            self.save_checkpoint(state, is_best)

            # Step the scheduler based on validation accuracy
            scheduler.step(val_acc)
        
        print("Training complete")
        return best_val_acc

    def save_checkpoint(self, state, is_best, filename='checkpoint.pth'):
        """
        Saves the model and training state.
        """
        checkpoint_dir = configs["model"]["checkpoint"]
        file_path = Path(checkpoint_dir) / filename
        
        # Save the current state
        torch.save(state, file_path)
        
        # If this is the best model seen so far, save a copy of it
        if is_best:
            best_filepath = os.path.join(checkpoint_dir, 'model_best.pth')
            torch.save(state, best_filepath)
            print(f"--> Saved new best model to {best_filepath}")

class TestObjectLocalization():
    def __init__(self) -> None:
        self.device = args["device"] or "cpu"
        self.batch_size = args["batch_size"] or 32
        self.epochs = args["epochs"] or 50
        self.lr = args["lr"] or 1e-4
        self.dropout = args["drop_out"] or 0.5
        self.wgt_decay = args["wgt_decay"] or 1e-4
        self.model_save_name = args["path"]

        # Data related parameters
        self.IN_CHANNELS = 3 # e.g., RGB images
        self.NUM_CLASSES = 37 # e.g., Oxford-IIIT Pet dataset

        self.model = ObjectLocalizer(in_channels=self.IN_CHANNELS, num_classes=self.NUM_CLASSES,
                                     output_size=4, p = self.dropout)

        self._initialize_loader()

    def _initialize_loader(self):
        # Assuming you have a test_loader defined similarly to train_loader
        # test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
        pass

    def load_checkpoint(self, checkpoint_path, model, optimizer=None, scaler=None):
        """
        Loads a checkpoint to resume training or perform inference.
        """

        file_path = Path(configs["model"]["checkpoint"]) / checkpoint_path
        checkpoint = torch.load(file_path)
        
        model.load_state_dict(checkpoint['state_dict'])
        
        if optimizer and 'optimizer' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
        if scaler and 'scaler' in checkpoint:
            scaler.load_state_dict(checkpoint['scaler'])
            
        print(f"=> Loaded checkpoint (epoch {checkpoint['epoch']})")
        return checkpoint['epoch'], checkpoint['best_acc']

    def evaluate_model(self, class_names=None):
        """
        Evaluates the model on a dedicated test set and prints detailed metrics.
        """

        model = self.model.to(self.device)
        model.eval()
        
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for images, labels in self.test_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
        print("\n--- Test Set Evaluation ---")
        report = classification_report(all_labels, all_preds, target_names=class_names, digits=4)
        print(report)
        cm = confusion_matrix(all_labels, all_preds)
        return cm, all_labels, all_preds

    def plot_confusion_matrix(self, cm, num_classes):
        """
        Visualizes the confusion matrix.
        """
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=False, cmap='Blues', fmt='g')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.title('Confusion Matrix')
        plt.show()

if __name__ == "__main__":
    trainer = TrainObjectLocalization()
    trainer.train_model()

    tester = TestObjectLocalization()
    tester.evaluate_model()
