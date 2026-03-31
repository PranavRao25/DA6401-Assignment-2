import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from src.dataset import CompleteImageDataset, IncompleteImageDataset
import albumentations as A
from tqdm import tqdm
import wandb
import os
import json
import pandas as pd
import logging
from src.vgg11 import VGG11 
import yaml
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from src.augments import complete_resize_pipeline, incomplete_resize_pipeline

parent = Path(__file__).resolve().parent.parent

logging.basicConfig(filename = parent / "prog_log.log",
                format = '%(levelname)s: %(message)s',
                filemode = 'a')
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

with open(parent / "train_1.yaml", "r") as f:
    args = yaml.full_load(f)

with open(parent / "config.yaml", "r") as f:
    configs = yaml.full_load(f)

# Load dataset
complete_train_path = configs["data"]["complete"]["train"]
incomplete_train_path = configs["data"]["incomplete"]["train"]
complete_valid_path = configs["data"]["complete"]["valid"]
incomplete_valid_path = configs["data"]["incomplete"]["valid"]
complete_test_path = configs["data"]["complete"]["test"]
incomplete_test_path = configs["data"]["incomplete"]["test"]

class TrainVGG11():
    def __init__(self, wandb_project) -> None:
        logger.info("Train init")
        
        wandb.init(
            project=wandb_project,
            config=args,
            name=args["name"]
        )

        print(args)
        self.device = args.get("device", "cpu")
        self.batch_size = args.get("batch_size", 128)
        self.epochs = args.get("epochs", 1)
        self.lr = args.get("lr", 0.01)
        self.wgt_decay = args.get("wgt_decay", 1e-4)
        self.model_save_name = args["path"]

        # Data related parameters
        self.IN_CHANNELS = 3 # e.g., RGB images
        self.NUM_CLASSES = 37 # e.g., Oxford-IIIT Pet dataset

        self.model = VGG11(in_channels=self.IN_CHANNELS, num_classes=self.NUM_CLASSES)

        self._initialize_loader()

    def _initialize_loader(self):
        self.complete_train_dataset = CompleteImageDataset(complete_train_path,
                                                           transform=complete_resize_pipeline)
        self.incomplete_train_dataset = IncompleteImageDataset(incomplete_train_path,
                                                               transform=incomplete_resize_pipeline)

        self.complete_valid_dataset = CompleteImageDataset(complete_valid_path,
                                                           transform=complete_resize_pipeline)
        self.incomplete_valid_dataset = IncompleteImageDataset(incomplete_valid_path,
                                                               transform=incomplete_resize_pipeline)

        self.complete_train_loader = DataLoader(self.complete_train_dataset,
                                                batch_size=self.batch_size, shuffle=True, num_workers=4,
                                                pin_memory=True)
        self.incomplete_train_loader = DataLoader(self.incomplete_train_dataset, 
                                                  batch_size=self.batch_size, shuffle=True, num_workers=4,
                                                  pin_memory=True)

        self.complete_valid_loader = DataLoader(self.complete_valid_dataset,
                                                batch_size=self.batch_size, shuffle=False, num_workers=4,
                                                pin_memory=True)
        self.incomplete_valid_loader = DataLoader(self.incomplete_valid_dataset,
                                                  batch_size=self.batch_size, shuffle=False, num_workers=4,
                                                  pin_memory=True)
        
        logger.info("data loader done")

    def train_model(self):
        logger.info("train start")

        self.model = self.model.to(self.device)

        wandb.watch(self.model, log="all", log_freq=100)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.wgt_decay)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=3)
        
        for epoch in range(self.epochs):
            logger.info(f"epoch - {epoch}")
            self.model.train()
            running_loss = 0.0
            correct_train = 0
            total_train = 0
            val_loss = 0.0
            correct_val = 0
            total_val = 0
            
            for train_loader, val_loader in zip(
                [self.complete_train_loader, self.incomplete_train_loader],
                [self.complete_valid_loader, self.incomplete_valid_loader]):
                
                loop = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{self.epochs}] Train")
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
                train_loss = running_loss / len(train_loader)
                
                self.model.eval()
                with torch.no_grad():
                    for images, labels in val_loader:
                        images, labels = images.to(self.device), labels.to(self.device)
                        outputs = self.model(images)
                        loss = criterion(outputs, labels)
                        
                        val_loss += loss.item()
                        _, predicted = torch.max(outputs.data, 1)
                        total_val += labels.size(0)
                        correct_val += (predicted == labels).sum().item()
                
                val_acc = 100. * correct_val / total_val
                avg_val_loss = val_loss / len(val_loader)
            
            print(f"Epoch [{epoch+1}/{self.epochs}] Summary: "
                f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | "
                f"Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            
            wandb.log({
                "epoch": epoch + 1,
                "train/loss": train_loss,
                "train/accuracy": train_acc,
                "val/loss": avg_val_loss,
                "val/accuracy": val_acc,
                "lr": optimizer.param_groups[0]["lr"]
            })
            
            is_best = val_acc > best_val_acc # type: ignore
            if is_best:
                best_val_acc = val_acc
                
            state = {
                'epoch': epoch + 1,
                'state_dict': self.model.state_dict(),
                'best_acc': best_val_acc,
                'optimizer': optimizer.state_dict(),
            }
            
            self.save_checkpoint(state, is_best)
            scheduler.step(val_acc)
        
        print("Training complete")
        wandb.finish()

        return best_val_acc

    def save_checkpoint(self, state, is_best):
        """
        Saves the model and training state.
        """
        checkpoint_dir = configs["model"]["checkpoint"]
        file_path = Path(checkpoint_dir) / self.model_save_name # filename
        
        torch.save(state, file_path)

        if is_best:
            best_filepath = os.path.join(checkpoint_dir, 'model_best.pth')
            torch.save(state, best_filepath)
            print(f"--> Saved new best model to {best_filepath}")

class TestVGG11():
    def __init__(self, wandb_project) -> None:
        wandb.init(
            project=wandb_project,
            config=args,
            name=args["name"]
        )

        self.device = args["device"] or "cpu"
        self.batch_size = args["batch_size"] or 32
        self.epochs = args["epochs"] or 50
        self.lr = args["lr"] or 1e-4
        self.wgt_decay = args["wgt_decay"] or 1e-4
        self.dropout = args["drop_out"] or 0.5
        self.model_save_name = args["path"]

        # Data related parameters
        self.IN_CHANNELS = 3 # e.g., RGB images
        self.NUM_CLASSES = 37 # e.g., Oxford-IIIT Pet dataset

        self.model = VGG11(in_channels=self.IN_CHANNELS,
                           num_classes=self.NUM_CLASSES, p = self.dropout)

        self._initialize_loader()

    def _initialize_loader(self):
        self.complete_test_dataset = CompleteImageDataset(complete_test_path,
                                                          transform=complete_resize_pipeline)
        self.incomplete_test_dataset = IncompleteImageDataset(incomplete_test_path,
                                                              transform=incomplete_resize_pipeline)

        self.complete_test_loader = DataLoader(self.complete_test_dataset,
                                                batch_size=self.batch_size, shuffle=True, num_workers=4,
                                                pin_memory=True)
        self.incomplete_test_loader = DataLoader(self.incomplete_test_dataset, 
                                                  batch_size=self.batch_size, shuffle=True, num_workers=4,
                                                  pin_memory=True)

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
        
        for test_loader in [self.complete_test_loader, self.incomplete_test_loader]:
            with torch.no_grad():
                for images, labels in test_loader:
                    images, labels = images.to(self.device), labels.to(self.device)
                    
                    outputs = model(images)
                    _, predicted = torch.max(outputs, 1)
                    
                    all_preds.extend(predicted.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())
                
        print("\n--- Test Set Evaluation ---")
        report = classification_report(all_labels, all_preds, target_names=class_names, digits=4)
        report_dict = classification_report(all_labels, all_preds, target_names=class_names, output_dict=True)
        print(report)
        
        wandb.log({
            "test/accuracy": report_dict["accuracy"], # type: ignore
            "test/macro_f1": report_dict["macro_avg"]["f1-score"], # type: ignore
            "test/weighted_f1": report_dict["weighted avg"]["f1-score"], # type: ignore
            "test/confusion_matrix": wandb.plot.confusion_matrix(
                probs=None,
                y_true=all_labels,
                preds=all_preds,
                class_names=class_names
            )
        })

        cm = confusion_matrix(all_labels, all_preds)
        wandb.finish()
        
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
    wandb_project = configs["wandb"]

    trainer = TrainVGG11(wandb_project)
    trainer.train_model()

    tester = TestVGG11(wandb_project)
    tester.evaluate_model()
