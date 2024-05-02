import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

class LeNet5(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2) 
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)
        self.sigmoid = torch.nn.Sigmoid()
    
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
    def training_step(self, batch):
        images, labels = batch
        out = self.forward(images)
        loss = torch.nn.BCEWithLogitsLoss()(out, labels.unsqueeze(1).float())
        return loss
    
    def validation_step(self, batch):
        images, labels = batch
        out = self.forward(images)
        loss = torch.nn.BCEWithLogitsLoss()(out, labels.unsqueeze(1).float())
        return loss
    
    def train_model(self, train_dataloader, val_dataloader, epochs=10, lr=0.001, device=None, wandb=None, freeze=False):
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
        train_losses = []
        val_losses = []
        print("Training on: " + str(device))

        aucs = []
        best_fpr = []
        best_tpr = []
        best_auc = 0

        for epoch in range(epochs):
            # Training Phase with gpu
            self.train()
            for i, batch in enumerate(train_dataloader):
                _, images, labels = batch
                optimizer.zero_grad()
                images, labels = images.to(device), labels.to(device)
                loss_train = self.training_step((images, labels))
                loss_train.backward()
                optimizer.step()
                train_losses.append(loss_train.item())
            # Validation phase with gpu
            lr_scheduler.step()
            self.eval()
            full_val = []

            for batch in val_dataloader:
                _, images, labels = batch
                full_val.append((images, labels))
                images, labels = images.to(device), labels.to(device)
                loss_val = self.validation_step((images, labels))
                val_losses.append(loss_val.item())
            
            # Metrics evaluation
            full_val = torch.cat([images for images, _ in full_val]), torch.cat([labels for _, labels in full_val])
            images, labels = full_val
            images, labels = images.to(device), labels.to(device)
            out = self.sigmoid(self.forward(images))
            fpr, tpr, _ = roc_curve(labels.cpu().detach(), out.cpu().detach())
            roc_auc = auc(fpr, tpr)

            if roc_auc > best_auc:
                best_auc = roc_auc
                best_fpr = fpr
                best_tpr = tpr
            aucs.append(roc_auc)
                
            print("Epoch: " + str(epoch) + " Loss train: " + str(np.average(train_losses)))
            print("Epoch: " + str(epoch) + " Loss val: " + str(np.average(val_losses)))
            if wandb:
                wandb.log({"loss train average": np.average(train_losses), "epoch": epoch,
                           "loss val average": np.average(val_losses)})
            if epoch + 1 % 5 == 0:
                torch.save(self.state_dict(), "./weights/lenet5_model.pth")
        torch.save(self.state_dict(), "./weights/lenet5_model.pth")

        
        
        print(f'Best AUC: {best_auc}')
        fig, ax = plt.subplots()
        ax.plot(best_fpr, best_tpr)
        ax.set_title("Best ROC")

        plt.show()
    
    def predict_model(self, images, device, path=None):
        self.eval()  # Set the model to evaluation mode
        outputs = []
        with torch.no_grad():  # No need to compute gradients during inference
            for batch in images:
                img_name, image, _ = batch
                image = image.to(device)
                out = self.forward(image)
                outputs.append((img_name,torch.nn.functional.softmax(out, dim=1)))
            # from Nx(32x1) to a list of tuples (img_name, prediction)
            final_outputs=[]
            for output in outputs:
                # tuple of tensors
                img_names, predictions = output
                for img_name, prediction in zip(img_names, predictions):
                    final_outputs.append((img_name, torch.round(self.sigmoid(prediction))))
        if path:
            torch.save(str(outputs), path)  # Optionally save the predictions
        return final_outputs