import numpy as np
from torchvision.models import resnet34, ResNet34_Weights
import torch
import os


class CactusModel(torch.nn.Module):
    def __init__(self):
        super(CactusModel, self).__init__()
        self.resnet = resnet34(weights=ResNet34_Weights.DEFAULT)
        self.resnet.fc = torch.nn.Sequential(
            torch.nn.Linear(in_features=self.resnet.fc.in_features, out_features=256, bias=True),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.5),  # Add dropout with dropout rate of 0.5
            torch.nn.Linear(in_features=256, out_features=1, bias=True),
        )
        if os.path.exists("./weights/cactus_model.pth"):
            self.load_state_dict(torch.load("./weights/cactus_model.pth"))

    def forward(self, x):
        return self.resnet(x)

    def training_step(self, batch):
        images, labels = batch
        out = self.resnet(images)
        loss = torch.nn.BCEWithLogitsLoss()(out, labels.unsqueeze(1).float())
        return loss

    def validation_step(self, batch):
        images, labels = batch
        out = self.resnet(images)
        loss = torch.nn.BCEWithLogitsLoss()(out, labels.unsqueeze(1).float())
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
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
        train_losses = []
        val_losses = []
        print("Training on: " + str(device))
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
            for batch in val_dataloader:
                _, images, labels = batch
                images, labels = images.to(device), labels.to(device)
                loss_val = self.validation_step((images, labels))
                val_losses.append(loss_val.item())
            print("Epoch: " + str(epoch) + " Loss train: " + str(np.average(train_losses)))
            print("Epoch: " + str(epoch) + " Loss val: " + str(np.average(val_losses)))
            if wandb:
                wandb.log({"loss train average": np.average(train_losses), "epoch": epoch,
                           "loss val average": np.average(val_losses)})
            if epoch + 1 % 5 == 0:
                torch.save(self.state_dict(), "./weights/cactus_model.pth")
        torch.save(self.state_dict(), "./weights/cactus_model.pth")

    def predict_model(self, images, device, path=None):
        self.eval()  # Set the model to evaluation mode
        outputs = []
        with torch.no_grad():  # No need to compute gradients during inference
            for batch in images:
                img_name, image, _ = batch
                image = image.to(device)
                out = self.resnet(image)
                out = torch.sigmoid(out)
                out = torch.round(out)
                outputs.append((img_name, out))
            # from Nx(32x1) to a list of tuples (img_name, prediction)
            final_outputs=[]
            for output in outputs:
                # tuple of tensors
                img_names, predictions = output
                for img_name, prediction in zip(img_names, predictions):
                    final_outputs.append((img_name, prediction.item()))
        if path:
            with open(path, "w") as f:
                for img_name, prediction in final_outputs:
                    f.write(img_name + "," + str(prediction) + "\n")
        return final_outputs