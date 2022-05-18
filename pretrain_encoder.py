import torch
from torch import nn
from torch.nn import functional as F
from my_models import ConvLayer

from torch.utils import data
from torch.utils.data import DataLoader
from my_utils.folder2lmdb import ImageFolderLMDB
from torchvision import transforms

import math
import args

device = torch.device(
    f"cuda:{args.gpu_num}" if torch.cuda.is_available() else 'cpu')
torch.cuda.set_device(device)  # change allocation of current GPU
print(f'cuda device : {device}')


def data_sampler(dataset, shuffle, distributed):
    if distributed:
        return data.distributed.DistributedSampler(dataset, shuffle=shuffle)

    if shuffle:
        return data.RandomSampler(dataset)

    else:
        return data.SequentialSampler(dataset)


class ResidualBlock(nn.Module):
    def __init__(self, in_channel, out_channel, blur_kernel=[1, 3, 3, 1]):
        super().__init__()

        self.conv1 = ConvLayer(in_channel, in_channel, 3)
        self.conv2 = ConvLayer(in_channel, out_channel, 3, downsample=True)

        self.skip = ConvLayer(
            in_channel, out_channel, 1, downsample=True, activate=False, bias=False
        )

    def forward(self, input):
        out = self.conv1(input)
        out = self.conv2(out)

        skip = self.skip(input)
        out = (out + skip) / math.sqrt(2)

        return out


class ResNet(nn.Module):
    def __init__(self, channels=3, num_classes=2, return_features=False, blur_kernel=[1, 3, 3, 1]):
        super(ResNet, self).__init__()

        self.rgb = channels
        self.in_planes = 32
        # RGB여서 3, in_planes는 내맘대로 16
        self.conv1 = nn.Conv2d(self.rgb, 32,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.block1 = ResidualBlock(32, 32)
        self.block2 = ResidualBlock(32, 64)
        self.block3 = ResidualBlock(64, 64)
        self.block4 = ResidualBlock(64, 128)
        self.block5 = ResidualBlock(128, 256)
        self.linear = nn.Linear(2048, num_classes)

        self.return_features = return_features

    def _make_layer(self, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(ResidualBlock(self.in_planes, planes, stride))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        feat_list = []
        out1 = F.relu(self.bn1(self.conv1(x)))
        feat_list.append(out1)
        out2 = self.block1(out1)
        feat_list.append(out2)
        out3 = self.block2(out2)
        feat_list.append(out3)
        out4 = self.block3(out3)
        feat_list.append(out4)
        out5 = self.block4(out4)
        feat_list.append(out5)
        out6 = F.avg_pool2d(out5, 4)
        out7 = out5.view(out6.size(0), -1)
        output = self.linear(out7)

        if self.return_features:
            return output, feat_list
        else:
            return output


def train(model, train_loader, optimizer, log_interval):
    model.train()
    for batch_idx, (image, label) in enumerate(train_loader):
        image = image.to(device)
        label = label.to(device)
        optimizer.zero_grad()
        output, _ = model(image)
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()

        if batch_idx % log_interval == 0:
            print("Train Epoch: {} [{}/{}({:.0f}%)]\tTrain Loss: {:.6f}".format(
                Epoch, batch_idx*len(image), len(train_loader.dataset), 100.*batch_idx/len(train_loader), loss.item()))
    return output


def evaluate(model, test_loader):
    model.eval()
    test_loss = 0
    correct = 0

    with torch.no_grad():
        for image, label in test_loader:
            image = image.to(device)
            label = label.to(device)
            output, _ = model(image)
            test_loss += criterion(output, label).item()
            # output에서 제일 큰 놈의 index를 반환한다(이경우에 0 or 1)
            prediction = output.max(1, keepdim=True)[1]
            correct += prediction.eq(label.view_as(prediction)).sum().item()

    test_loss /= len(test_loader.dataset)
    test_accuracy = 100*correct / len(test_loader.dataset)
    return test_loss, test_accuracy


if __name__ == '__main__':
    device = torch.device(
        f"cuda:{args.gpu_num}" if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(device)  # change allocation of current GPU
    print(f'cuda device : {device}')

    my_transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.Resize(args.img_size),
            transforms.CenterCrop(args.center_crop),
            transforms.Resize(args.img_size),
            transforms.ToTensor(),
            transforms.Normalize(
                (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True),
        ]

    )
    lmdb_ImageFolder = ImageFolderLMDB
    train_dataset = lmdb_ImageFolder(args.train_path, transform=my_transform)
    test_dataset = lmdb_ImageFolder(args.test_path, transform=my_transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch,
        sampler=data_sampler(train_dataset, shuffle=True, distributed=False),
        drop_last=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch,
        sampler=data_sampler(test_dataset, shuffle=False, distributed=False),
        drop_last=True,
    )

    model = ResNet(return_features=True).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    best_accuracy = 0
    # for Epoch in range(1, 10+1):
    #     train(model, train_loader, optimizer, log_interval=200)
    #     test_loss, test_accuracy = evaluate(model, test_loader)
    #     if test_accuracy > best_accuracy:
    #         best_accuracy = test_accuracy
    #         torch.save(model, 'pretrained/classifier/ResNet_64_smiling.pt')
    #         torch.save(model.state_dict(), 'pretrained/classifier/ResNet_64_parameters_smiling.pt')
    #     print("[EPOCH: {}], \tTest Loss: {:.4f},\tTest Accuracy: {:.2f}%\n".format(
    #         Epoch, test_loss, test_accuracy))
