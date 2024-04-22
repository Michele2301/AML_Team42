import torch
import os


class BasicBlock(torch.nn.Module):
    def __init__(self, size_in, size_out, kernel_size, stride, padding):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(in_channels=size_in, out_channels=size_out, kernel_size=kernel_size, stride=stride,
                                     padding=padding)
        self.bn1 = torch.nn.BatchNorm2d(size_out)
        self.relu = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv2d(in_channels=size_out, out_channels=size_out, kernel_size=kernel_size, stride=1,
                                     padding=padding)
        self.bn2 = torch.nn.BatchNorm2d(size_out)
        self.downsample = None
        self.size_in = size_in
        self.size_out = size_out
        self.stride = stride
        self.padding = padding
        self.kernel_size = kernel_size
        if size_in != size_out:
            self.downsample = torch.nn.Sequential(
                torch.nn.Conv2d(size_in, size_out, kernel_size=1, stride=stride, bias=False),
                torch.nn.BatchNorm2d(size_out),
            )

    def forward(self, x):
        identity = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        if self.downsample is not None:
            identity = self.downsample(identity)
        x += identity
        x = self.relu(x)
        return x


class CustomResNet34(torch.nn.Module):
    # input 224,224,3
    # output 2

    def __init__(self):
        super().__init__()
        # first part (no skip connections)
        self.conv1 = torch.nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3)
        self.bn1 = torch.nn.BatchNorm2d(64)
        self.relu = torch.nn.ReLU()
        self.maxpool = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # blocks with skip connections
        # 64 -> 64 3 times (input is 224,224,3)
        self.basic_block1 = BasicBlock(64, 64, 3, 1, 1)
        self.basic_block2 = BasicBlock(64, 64, 3, 1, 1)
        self.basic_block3 = BasicBlock(64, 64, 3, 1, 1)
        # 64 -> 128 4 times (input is 112,112,64)
        self.basic_block4 = BasicBlock(64, 128, 3, 2, 1)
        self.basic_block5 = BasicBlock(128, 128, 3, 1, 1)
        self.basic_block6 = BasicBlock(128, 128, 3, 1, 1)
        self.basic_block7 = BasicBlock(128, 128, 3, 1, 1)
        # 128 -> 256 6 times (input is 56,56,128)
        self.basic_block8 = BasicBlock(128, 256, 3, 2, 1)
        self.basic_block9 = BasicBlock(256, 256, 3, 1, 1)
        self.basic_block10 = BasicBlock(256, 256, 3, 1, 1)
        self.basic_block11 = BasicBlock(256, 256, 3, 1, 1)
        self.basic_block12 = BasicBlock(256, 256, 3, 1, 1)
        self.basic_block13 = BasicBlock(256, 256, 3, 1, 1)
        # 256 -> 512 3 times (input is 28,28,256)
        self.basic_block14 = BasicBlock(256, 512, 3, 2, 1)
        self.basic_block15 = BasicBlock(512, 512, 3, 1, 1)
        self.basic_block16 = BasicBlock(512, 512, 3, 1, 1)
        # final layers
        # goes from batch_size,512,28,28 to batch_size,512,1,1
        self.avgpool = torch.nn.AdaptiveAvgPool2d((1, 1))
        # fc layer
        self.fc = torch.nn.Linear(512, 2)
        # softmax
        self.softmax = torch.nn.Softmax(dim=1)
        if os.path.exists("./weights/custom_resnet.pth"):
            self.load_state_dict(torch.load("./weights/custom_resnet.pth"))

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.basic_block1(x)
        x = self.basic_block2(x)
        x = self.basic_block3(x)
        x = self.basic_block4(x)
        x = self.basic_block5(x)
        x = self.basic_block6(x)
        x = self.basic_block7(x)
        x = self.basic_block8(x)
        x = self.basic_block9(x)
        x = self.basic_block10(x)
        x = self.basic_block11(x)
        x = self.basic_block12(x)
        x = self.basic_block13(x)
        x = self.basic_block14(x)
        x = self.basic_block15(x)
        x = self.basic_block16(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        x = self.softmax(x)
        return x

    def training_step(self, batch):
        images, labels = batch
        out = self(images)
        loss = torch.nn.functional.cross_entropy(out, labels)
        return loss

    def validation_step(self, batch):
        images, labels = batch
        out = self(images)
        loss = torch.nn.functional.cross_entropy(out, labels)
        return loss

    def train_model(self, train_dataloader, val_dataloader, epochs=10, lr=0.001, device=None, wandb=None):
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        print("Training on: " + str(device))
        for epoch in range(epochs):
            # Training Phase with gpu
            self.train()
            torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
            for batch in train_dataloader:
                images, labels = batch
                optimizer.zero_grad()
                images, labels = images.to(device), labels.to(device)
                loss_train = self.training_step((images, labels))
                loss_train.backward()
                optimizer.step()
            # Validation phase with gpu
            self.eval()
            for batch in val_dataloader:
                images, labels = batch
                images, labels = images.to(device), labels.to(device)
                loss_val = self.validation_step((images, labels))
            print("Epoch: " + str(epoch) + " Loss train: " + str(loss_train.item()))
            print("Epoch: " + str(epoch) + " Loss val: " + str(loss_val.item()))
            if wandb:
                wandb.log({"loss train": loss_train.item(), "epoch": epoch, "loss val": loss_val.item()})
            if epoch + 1 % 5 == 0:
                torch.save(self.state_dict(), "./weights/custom_resnet.pth")
        torch.save(self.state_dict(), "./weights/custom_resnet.pth")
