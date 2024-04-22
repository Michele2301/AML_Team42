import numpy as np
from torchvision.models import resnet34, ResNet34_Weights
import torch
import os

class CactusModel(torch.nn.Module):
    def __init__(self):
        super(CactusModel, self).__init__()
        if os.path.exists("./weights/resnet50_weights.pth"):
            self.resnet = resnet34(pretrained=False)
            self.resnet.load_state_dict(torch.load("./weights/resnet50_weights.pth"))
        else:
            self.resnet = resnet34(weights=ResNet34_Weights.DEFAULT)
        self.resnet.fc = torch.nn.Sequential(
            torch.nn.Linear(in_features=self.resnet.fc.in_features, out_features=256, bias=True),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.5),  # Add dropout with dropout rate of 0.5
            torch.nn.Linear(in_features=256, out_features=2, bias=True),
        )

    def forward(self, x):
        return self.resnet(x)

    def training_step(self, batch):
        images, labels = batch
        out = self.resnet(images)
        loss = torch.nn.functional.cross_entropy(input=out, target=labels)
        return loss

    def validation_step(self, batch):
        images, labels = batch
        out = self.resnet(images)
        loss = torch.nn.functional.cross_entropy(input=out, target=labels)
        return loss

    def train_model(self, train_dataloader, val_dataloader, epochs=10, lr=0.001, device=None, wandb=None, freeze=False):
        if freeze:
            for param in self.resnet.parameters():
                param.requires_grad = False
            for param in self.resnet.fc.parameters():
                param.requires_grad = True
        else:
            for param in self.resnet.parameters():
                param.requires_grad = True
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        train_losses=[]
        val_losses=[]
        print("Training on: " + str(device))
        for epoch in range(epochs):
            # Training Phase with gpu
            self.train()
            for i,batch in enumerate(train_dataloader):
                images, labels = batch
                optimizer.zero_grad()
                images, labels = images.to(device), labels.to(device)
                loss_train = self.training_step((images, labels))
                loss_train.backward()
                optimizer.step()
                train_losses.append(loss_train.item())
                print("Epoch: " + str(epoch) + " Batch: " + str(i) + " Loss train: " + str(loss_train.item()))
                if wandb:
                    wandb.log({"loss train": loss_train.item(), "epoch": epoch, "batch": i})

            # Validation phase with gpu
            self.eval()
            for batch in val_dataloader:
                images, labels = batch
                images, labels = images.to(device), labels.to(device)
                loss_val = self.validation_step((images, labels))
                val_losses.append(loss_val.item())
            print("Epoch: " + str(epoch) + " Loss train: " + str(np.average(train_losses)))
            print("Epoch: " + str(epoch) + " Loss val: " + str(np.average(val_losses)))
            if wandb:
                wandb.log({"loss train average": str(np.average(train_losses)), "epoch": epoch, "loss val average": str(np.average(val_losses))})
            if epoch+1 % 5 == 0:
                torch.save(self.state_dict(), "./weights/cactus_model.pth")
        torch.save(self.state_dict(), "./weights/cactus_model.pth")

    def predict_model(self, images, device, path=None):
        self.eval()  # Set the model to evaluation mode
        outputs = []
        with torch.no_grad():  # No need to compute gradients during inference
            for batch in images:
                image, _ = batch
                image = image.to(device)
                out = self.resnet(image)
                out = torch.nn.functional.softmax(out, dim=1)
                outputs.append(torch.max(out,dim=1))
        outputs = torch.cat(outputs, dim=0)
        if path:
            torch.save(str(outputs), path)  # Optionally save the predictions
        return outputs

