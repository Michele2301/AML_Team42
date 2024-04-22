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
        self.resnet.fc = torch.nn.Linear(in_features=self.resnet.fc.in_features, out_features=2, bias=True)
        # add a softmax layer
        self.resnet = torch.nn.Sequential(self.resnet, torch.nn.Softmax(dim=1))

    def forward(self, x):
        return self.resnet(x)

    def training_step(self, batch):
        images, labels = batch
        out = self.resnet(images)
        loss = torch.nn.functional.cross_entropy(out, labels)
        return loss

    def validation_step(self, batch):
        images, labels = batch
        out = self.resnet(images)
        loss = torch.nn.functional.cross_entropy(out, labels)
        return loss

    def train_model(self, train_dataloader, val_dataloader, epochs=10, lr=0.001, device=None, wandb=None, freeze=False):
        if freeze:
            for param in self.resnet[0].parameters():
                param.requires_grad = False
            for param in self.resnet[0].fc.parameters():
                param.requires_grad = True
        else:
            for param in self.resnet.parameters():
                param.requires_grad = True
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        print("Training on: " + str(device))
        for epoch in range(epochs):
            # Training Phase with gpu
            self.train()
            for batch in train_dataloader:
                images, labels = batch
                images, labels = images.to(device), labels.to(device)
                loss = self.training_step((images, labels))
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
            # Validation phase with gpu
            self.eval()
            for batch in val_dataloader:
                images, labels = batch
                images, labels = images.to(device), labels.to(device)
                loss = self.validation_step((images, labels))
            print("Epoch: " + str(epoch) + " Loss: " + str(loss.item()))
            if wandb:
                wandb.log({"loss": loss.item(), "epoch": epoch})
            if epoch % 5 == 0:
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
                outputs.append(out)
        outputs = torch.cat(outputs, dim=0)
        if path:
            torch.save(str(outputs), path)  # Optionally save the predictions
        return outputs

