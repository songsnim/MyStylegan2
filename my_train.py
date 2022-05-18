import math
import os

import torch
from torch import nn, autograd
from torch.nn import functional as F
from torch.utils import data

from op import conv2d_gradfix
from non_leaking import augment, AdaptiveAugment

import torch.distributed as distributed
from distributed import get_rank, synchronize, reduce_loss_dict, reduce_sum, get_world_size
from tqdm import tqdm
import wandb
from lpips_pytorch import LPIPS, lpips

import args


def sample_data(loader):
    while True:
        for batch in loader:
            yield batch


def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag


def accumulate(model1, model2, decay=0.999):  # g_ema(exponential moving average) 계산을 위한 함수
    par1 = dict(model1.named_parameters())
    par2 = dict(model2.named_parameters())

    for k in par1.keys():
        par1[k].data.mul_(decay).add_(par2[k].data, alpha=1 - decay)


def d_logistic_loss(real_pred, fake_pred):
    real_loss = F.softplus(-real_pred)
    fake_loss = F.softplus(fake_pred)

    return real_loss.mean() + fake_loss.mean()


def d_r1_loss(real_pred, real_img):
    with conv2d_gradfix.no_weight_gradients():
        grad_real, = autograd.grad(  # 입력에 대한 output의 grad의 sum을 반환한다.
            outputs=real_pred.sum(), inputs=real_img, create_graph=True
        )
    grad_penalty = grad_real.pow(2).reshape(
        grad_real.shape[0], -1).sum(1).mean()

    return grad_penalty


def img_recon_loss(real_img, fake_img):
    # print(f"fake_image.shape = {fake_img.shape}, real_image.shape = {real_img.shape}")
    assert fake_img.shape == real_img.shape
    recon_loss = F.l1_loss(fake_img, real_img, reduction='mean')
    return recon_loss


def style_recon_loss(real_styles, fake_styles):
    # print(f"fake_latent.shape = {fake_latent.shape}, real_latent.shape = {real_latent.shape}")
    final_loss = 0
    for real_style, fake_style in zip(real_styles, fake_styles):
        if real_style.shape != fake_style.shape:
            real_style = real_style.squeeze(0)
        assert fake_style[0].shape == real_style[0].shape
        recon_loss = F.l1_loss(fake_style, real_style, reduction='mean')
        final_loss += recon_loss
    return final_loss


def lpips_recon_loss(fake_img, real_img):
    return lpips(fake_img, real_img, net_type='vgg', version='0.1')


def g_nonsaturating_loss(fake_pred):
    loss = F.softplus(-fake_pred).mean()

    return loss


def g_path_regularize(fake_img, style, mean_path_length, decay=0.01):
    # y 연산 (random noise image)
    noise = torch.randn_like(fake_img) / \
        math.sqrt(fake_img.shape[2] * fake_img.shape[3])
    # fake_image * noise => g(w) * y,  latents => w
    # 델연산(g(w) * y) => 합의 편미분
    grad, = autograd.grad(outputs=(fake_img * noise).sum(),
                          inputs=style, create_graph=True)
    # path length인 ||Jy||_2는 grad의 l2 norm이므로, 제곱의 합의 루트 (실제 구현할 때는 mean 추가)
    path_lengths = torch.sqrt(grad.pow(2).sum().mean())

    # the long running exponential moving average of path length = a 구하기
    path_mean = mean_path_length + decay * \
        (path_lengths.mean() - mean_path_length)

    # 최종 페널티!! E[(path length - a)^2]
    path_penalty = (path_lengths - path_mean).pow(2).mean()

    return path_penalty, path_mean.detach(), path_lengths


def get_sample_for_log(test_image, encoder, generator, g_ema):
    with torch.no_grad():
        g_ema.eval()
        _, feat_list = encoder(test_image)
        test_sample_ema, styles_ema, spaces_ema = g_ema(feat_list)
        test_sample, styles, spaces = generator(feat_list)

        sample = torch.cat([test_sample[:int(args.batch/2)],
                           test_image[:int(args.batch/2)]], dim=0)

        sample_ema = torch.cat(
            [test_sample_ema[:int(args.batch/2)], test_image[:int(args.batch/2)]], dim=0)
    return sample, sample_ema


def get_config_from_args(args):
    params = list(args.__dict__.keys())
    params = params[params.index('gpu_num'):]
    output = {}
    for param in params:
        output[param] = args.__dict__[param]
    return output


def get_accuracy(test_loader, encoder, generator, predictor, device):
    correct = 0
    with torch.no_grad():
        for idx, (image, label) in enumerate(test_loader):
            image = image.to(device)
            label = label.to(device)
            _, feat_list = encoder(image)
            _, styles, _ = generator(feat_list)
            P_pred = predictor(styles)
            # output에서 제일 큰 놈의 index를 반환한다(이경우에 0 or 1)
            prediction = P_pred.max(1, keepdim=True)[1]
            correct += prediction.eq(label.view_as(prediction)).sum().item()
            if idx == 100:
                break
    test_accuracy = 100*correct / ((idx+1)*args.batch)

    return test_accuracy


def train(args, train_loader, test_loader, encoder, generator, discriminator, predictor,
          recon_optim, p_optim, g_optim, d_optim, g_ema, today, device):
    train_loader = sample_data(train_loader)
    test_loader = sample_data(test_loader)

    CEloss = nn.CrossEntropyLoss()
    KLDloss = torch.nn.KLDivLoss()

    pbar = tqdm(range(args.iter), initial=args.start_iter,
                dynamic_ncols=True, smoothing=0.01)

    mean_path_length = 0

    # loss value setup
    d_loss_val = 0
    r1_loss = torch.tensor(0.0, device=device)
    g_loss_val = 0
    path_loss = torch.tensor(0.0, device=device)
    path_lengths = torch.tensor(0.0, device=device)
    mean_path_length_avg = 0
    loss_dict = {}

    if args.py:
        e_module = encoder.module
        g_module = generator.module
        d_module = discriminator.module
        p_module = predictor.module

    elif args.ipynb:
        e_module = encoder
        g_module = generator
        d_module = discriminator
        p_module = predictor

    accum = 0.5 ** (32 / (10*1000))
    ada_aug_prob = args.augment_p if args.augment_p > 0 else 0.0
    r_t_stat = 0

    if args.augment and args.augment_p == 0:
        ada_augment = AdaptiveAugment(
            args.ada_target, args.ada_length, 8, device)

    test_image, test_label = next(test_loader)
    test_image = test_image.to(device)
    test_label = test_label.to(device)

    for idx in pbar:
        i = idx + args.start_iter
        if i > args.iter:
            print("Done!")
            break

        real_img, real_label = next(train_loader)
        real_img = real_img.to(device)
        real_label = real_label.to(device)

        """===============================Train Discriminator================================== """

        requires_grad(generator, False)
        requires_grad(encoder, False)
        requires_grad(discriminator, True)
        requires_grad(predictor, False)

        _, feat_list = encoder(real_img)
        fake_img, styles, spaces = generator(feat_list)

        if args.augment:
            real_img_aug, _ = augment(real_img, ada_aug_prob)
            fake_img, _ = augment(fake_img, ada_aug_prob)

        else:
            real_img_aug = real_img

        fake_pred = discriminator(fake_img)
        real_pred = discriminator(real_img_aug)
        d_loss = d_logistic_loss(real_pred, fake_pred)

        loss_dict["d"] = d_loss
        loss_dict["real_score"] = real_pred.mean()
        loss_dict["fake_score"] = fake_pred.mean()

        discriminator.zero_grad()
        d_loss.backward()
        d_optim.step()

        if args.augment and args.augment_p == 0:
            ada_aug_prob = ada_augment.tune(real_pred)
            r_t_stat = ada_augment.r_t_stat

        d_regularize = i % args.d_reg_every == 0

        if d_regularize:
            real_img.requires_grad = True

            if args.augment:
                real_img_aug, _ = augment(real_img, ada_aug_prob)
            else:
                real_img_aug = real_img

            real_pred = discriminator(real_img_aug)
            r1_loss = d_r1_loss(real_pred, real_img)

            discriminator.zero_grad()
            r1_loss_backward = (args.r1 / 2 * r1_loss *
                                args.d_reg_every + 0 * real_pred[0])
            r1_loss_backward.backward()
            d_optim.step()

        loss_dict["r1"] = r1_loss

        """ ==================================train Generator================================== """

        requires_grad(generator, True)
        requires_grad(encoder, False)
        requires_grad(discriminator, False)
        requires_grad(predictor, True)

        _, feat_list = encoder(real_img)
        fake_img, styles, spaces = generator(feat_list)

        _, fake_feat_list = encoder(fake_img)
        _, fake_styles, fake_spaces = generator(fake_feat_list)

        w1 = torch.tensor([1, 1, 1], device=device)
        recon_loss = w1[0]*img_recon_loss(fake_img, real_img) + \
            w1[1]*style_recon_loss(styles, fake_styles) + \
            w1[2]*lpips_recon_loss(fake_img, real_img)

        loss_dict["recon"] = recon_loss

        if args.augment:
            fake_img, _ = augment(fake_img, ada_aug_prob)

        fake_pred = discriminator(fake_img)
        g_loss = g_nonsaturating_loss(fake_pred)

        loss_dict["g"] = g_loss

        real_pred_label = predictor(styles)
        fake_pred_label = predictor(fake_styles)

        w2 = torch.tensor([1, 1], device=device)
        p_pred_loss = CEloss(real_pred_label, real_label)
        if args.kl_pred_loss:
            p_reg_loss = KLDloss(fake_pred_label, real_pred_label)
            p_loss = w2[0] * p_pred_loss + w2[1] * p_reg_loss
        else:
            p_loss = w2[0] * p_pred_loss

        loss_dict["p"] = p_loss

        generator.zero_grad()
        predictor.zero_grad()

        recon_loss.detach()
        recon_loss.backward(retain_graph=True)
        g_loss.detach()
        g_loss.backward(retain_graph=True)
        p_loss.backward(retain_graph=True)

        recon_optim.step()
        g_optim.step()
        p_optim.step()

        g_regularize = i % args.g_reg_every == 0
        if g_regularize:
            path_batch_size = max(1, args.batch // args.path_batch_shrink)
            _, feat_list = encoder(real_img)
            fake_img, styles, spaces = generator(feat_list)

            total_path_loss = 0
            mean_path_lengths = [0]*len(styles)
            for idx, (space, mean_path_length) in enumerate(zip(spaces, mean_path_lengths)):
                path_loss, mean_path_length_output, path_lengths = g_path_regularize(
                    fake_img, space, mean_path_length)
                total_path_loss += path_loss
                mean_path_lengths[idx] = mean_path_length_output

            generator.zero_grad()
            weighted_path_loss = args.path_regularize * args.g_reg_every * total_path_loss

            if args.path_batch_shrink:
                weighted_path_loss += 0 * fake_img[0, 0, 0, 0]

            weighted_path_loss.backward()
            g_optim.step()

            mean_path_length_avg = (
                reduce_sum(sum(mean_path_lengths)).item() / get_world_size())

        loss_dict["path_loss"] = total_path_loss
        loss_dict["path_length"] = path_lengths.mean()

        if args.py:
            accumulate(g_ema, g_module, accum)
        elif args.ipynb:
            accumulate(g_ema, generator, accum)

        """ ================================== logging =================================="""
        loss_reduced = reduce_loss_dict(loss_dict)

        p_loss_val = loss_reduced["p"].item()
        recon_loss_val = loss_reduced["recon"].mean().item()
        d_loss_val = loss_reduced["d"].mean().item()
        g_loss_val = loss_reduced["g"].mean().item()
        r1_val = loss_reduced["r1"].mean().item()
        path_loss_val = loss_reduced["path_loss"].mean().item()
        real_score_val = loss_reduced["real_score"].mean().item()
        fake_score_val = loss_reduced["fake_score"].mean().item()
        path_length_val = loss_reduced["path_length"].mean().item()

        if get_rank() == 0:
            pbar.set_description(
                (
                    f"d: {d_loss_val:.4f}; g: {g_loss_val:.4f}; p: {p_loss_val:.4f}; r1: {r1_val:.4f}; "
                    f"path: {path_loss_val:.4f}; mean path: {mean_path_length_avg:.4f}; "
                    f"p: {ada_aug_prob:.4f}; recon_loss: {recon_loss_val:.4f};"
                )
            )

            if wandb:
                wandb.log(
                    {
                        "Pred": p_loss_val,
                        "Recon": recon_loss_val,
                        "Generator": g_loss_val,
                        "Discriminator": d_loss_val,
                        "Augment": ada_aug_prob,
                        "Rt": r_t_stat,
                        "R1": r1_val,
                        "Path Length Regularization": path_loss_val,
                        "Mean Path Length": mean_path_length,
                        "Real Score": real_score_val,
                        "Fake Score": fake_score_val,
                        "Path Length": path_length_val,
                    }
                )

            if i % 100 == 0:
                sample, sample_ema = get_sample_for_log(
                    test_image, encoder, generator, g_ema)
                sample = wandb.Image(sample)
                sample_ema = wandb.Image(sample_ema)
                wandb.log({"G, test": sample,
                           "g_ema, test": sample_ema})
                p_accuracy = get_accuracy(
                    test_loader, encoder, generator, predictor, device)
                wandb.log({"Accuracy": p_accuracy})

            if i % 1000 == 0:
                os.makedirs(
                    f"checkpoint/{today}_{args.description}/", exist_ok=True)
                torch.save(
                    {
                        "e": e_module.state_dict(),
                        "g": g_module.state_dict(),
                        "d": d_module.state_dict(),
                        "p": p_module.state_dict(),
                        "g_ema": g_ema.state_dict(),
                        "recon_optim": recon_optim.state_dict(),
                        "g_optim": g_optim.state_dict(),
                        "d_optim": d_optim.state_dict(),
                        # "args": args,
                        "ada_aug_prob": ada_aug_prob,
                    },
                    f"checkpoint/{today}_{args.description}/{str(i).zfill(6)}.pt",
                )
