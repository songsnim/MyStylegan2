import numpy as np
import matplotlib.pyplot as plt
import os

import torch
from torch import nn
import torch.nn.functional as F

import torchvision
from torchvision.datasets import ImageFolder
import torch.utils.data as data
from torch.utils.data import Dataset, DataLoader
from my_utils.folder2lmdb import ImageFolderLMDB
from torchvision import transforms

from torch import nn, optim
import itertools

import args
import wandb

import time
import datetime


def num_to_label(num):
    return 'smile' if num == 1 else 'non-smile'


def accumulate(model1, model2, decay=0.999):  # g_ema(exponential moving average) 계산을 위한 함수
    par1 = dict(model1.named_parameters())
    par2 = dict(model2.named_parameters())

    for k in par1.keys():
        par1[k].data.mul_(decay).add_(par2[k].data, alpha=1 - decay)


def prepare_dataloader(args):
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
    for image, label in test_loader:
        print(f"image : {image.shape}")
        print(f"label : {label.shape} {label[:10]}")
        break

    return train_loader, test_loader, image, label


def data_sampler(dataset, shuffle, distributed):
    if distributed:
        return data.distributed.DistributedSampler(dataset, shuffle=shuffle)

    if shuffle:
        return data.RandomSampler(dataset)

    else:
        return data.SequentialSampler(dataset)


def to_np(tensor):
    return tensor.cpu().detach().numpy()


def denorm(tensor):
    return (tensor + 1)/2


def get_config_from_args(args):
    params = list(args.__dict__.keys())
    params = params[params.index('gpu_num'):]
    output = {}
    for param in params:
        output[param] = args.__dict__[param]
    return output


if __name__ == '__main__':

    today = datetime.date.today()
    device = torch.device(
        f"cuda:{args.gpu_num}" if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(device)  # change allocation of current GPU
    print(f'오늘 날짜 : {today}')
    print(f"Description : {args.description}")
    print(f"torch version : {torch.__version__}")
    print(f"cuda version : {torch.version.cuda}")
    print(f'cuda device : {device}')

    n_gpu = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    print(f"n_gpu : {n_gpu}")
    args.distributed = n_gpu >= 2

    print("Importing models...")
    from my_models import *
    from my_train import *
    from pretrain_encoder import ResidualBlock, ResNet
    print("Importing models done..!")

    train_loader, test_loader, image, label = prepare_dataloader(args)

    E = ResNet(return_features=True).to(device)
    E.load_state_dict(torch.load(
        "pretrained/classifier/ResNet_64_parameters_smiling.pt"))
    pred, feat_list = E(image.to(device))
    G = Generator(feat_list, size=args.img_size,
                  style_dim=args.latent).to(device)
    _, styles, spaces = G(feat_list)
    D = Discriminator(args.img_size, args.latent).to(device)
    g_ema = Generator(feat_list, size=args.img_size,
                      style_dim=args.latent).to(device)
    P = Predictor(styles, args.disc_latent_ratio).to(device)

    g_ema.eval()
    accumulate(g_ema, G, 0)

    e_reg_ratio = args.d_reg_every / (args.d_reg_every + 1)
    g_reg_ratio = args.g_reg_every / (args.g_reg_every + 1)
    d_reg_ratio = args.d_reg_every / (args.d_reg_every + 1)

    recon_optim = optim.Adam(G.parameters(
    ), lr=args.lr * g_reg_ratio, betas=(0 ** g_reg_ratio, 0.99 ** g_reg_ratio))
    g_optim = optim.Adam(G.parameters(), lr=args.lr * g_reg_ratio,
                         betas=(0 ** g_reg_ratio, 0.99 ** g_reg_ratio))
    d_optim = optim.Adam(D.parameters(), lr=args.lr * d_reg_ratio,
                         betas=(0 ** d_reg_ratio, 0.99 ** d_reg_ratio))
    p_optim = optim.Adam(P.parameters(), lr=args.lr * g_reg_ratio,
                         betas=(0 ** g_reg_ratio, 0.99 ** g_reg_ratio))

    if args.ckpt is not None:
        print("load model:", args.ckpt)

        ckpt = torch.load(args.ckpt, map_location=lambda storage, loc: storage)

        try:
            ckpt_name = os.path.basename(args.ckpt)
            args.start_iter = int(os.path.splitext(ckpt_name)[0])

        except ValueError:
            pass

        E.load_state_dict(ckpt["e"])
        G.load_state_dict(ckpt["g"])
        D.load_state_dict(ckpt["d"])
        g_ema.load_state_dict(ckpt["g_ema"])
        recon_optim.load_state_dict(ckpt["recon_optim"])
        g_optim.load_state_dict(ckpt["g_optim"])
        d_optim.load_state_dict(ckpt["d_optim"])

    if args.wandb:
        wandb.init(project="StyleGAN2", entity='songyj', name=args.description,
                   config=get_config_from_args(args))
    torch.autograd.set_detect_anomaly(False)
    train(args, train_loader, test_loader, E, G, D, P, recon_optim,
          p_optim, g_optim, d_optim, g_ema, today, device)
