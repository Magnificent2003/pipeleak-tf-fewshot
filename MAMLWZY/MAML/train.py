import argparse
import os
import random
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

import matplotlib.pyplot as plt
import torchvision.utils as vutils
from sklearn.manifold import TSNE
import pandas as pd

import datasets
import models
import utils
import utils.optimizers as optimizers


def main(config):

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)

    ckpt_name = args.name
    if ckpt_name is None:
        ckpt_name = config['encoder']
        ckpt_name += '_' + config['dataset'].replace('meta-', '')
        ckpt_name += '_{}_way_{}_shot'.format(
            config['train']['n_way'], config['train']['n_shot'])

    if args.tag is not None:
        ckpt_name += '_' + args.tag

    ckpt_path = os.path.join('./save', ckpt_name)

    utils.ensure_path(ckpt_path)
    utils.set_log_path(ckpt_path)

    writer = SummaryWriter(os.path.join(ckpt_path, 'tensorboard'))

    yaml.dump(config, open(os.path.join(ckpt_path, 'config.yaml'), 'w'))

    vis_path = os.path.join(ckpt_path, "vis")
    curve_path = os.path.join(ckpt_path, "curves")

    os.makedirs(vis_path, exist_ok=True)
    os.makedirs(curve_path, exist_ok=True)

    ###################################################
    # Dataset
    ###################################################

    train_set = datasets.make(config['dataset'], **config['train'])

    utils.log('meta-train set: {} (x{}), {}'.format(
        train_set[0][0].shape, len(train_set), train_set.n_classes))

    train_loader = DataLoader(
        train_set,
        config['train']['n_episode'],
        collate_fn=datasets.collate_fn,
        num_workers=1,
        pin_memory=True
    )

    eval_val = False

    if config.get('val'):

        eval_val = True

        val_set = datasets.make(config['dataset'], **config['val'])

        utils.log('meta-val set: {} (x{}), {}'.format(
            val_set[0][0].shape, len(val_set), val_set.n_classes))

        val_loader = DataLoader(
            val_set,
            config['val']['n_episode'],
            collate_fn=datasets.collate_fn,
            num_workers=1,
            pin_memory=True
        )

    ###################################################
    # Model
    ###################################################

    inner_args = utils.config_inner_args(config.get('inner_args'))

    config['encoder_args'] = config.get('encoder_args') or dict()
    config['classifier_args'] = config.get('classifier_args') or dict()

    config['encoder_args']['bn_args']['n_episode'] = config['train']['n_episode']
    config['classifier_args']['n_way'] = config['train']['n_way']

    model = models.make(
        config['encoder'],
        config['encoder_args'],
        config['classifier'],
        config['classifier_args']
    )

    optimizer, lr_scheduler = optimizers.make(
        config['optimizer'],
        model.parameters(),
        **config['optimizer_args']
    )

    model = model.cuda()

    utils.log('num params: {}'.format(utils.compute_n_params(model)))

    ###################################################
    # Training
    ###################################################

    aves_keys = ['tl', 'ta', 'vl', 'va']

    trlog = {k: [] for k in aves_keys}

    max_va = 0.

    for epoch in range(1, config['epoch'] + 1):

        aves = {k: utils.AverageMeter() for k in aves_keys}

        model.train()

        writer.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch)

        ###################################################
        # META TRAIN
        ###################################################

        for data in tqdm(train_loader, desc='meta-train', leave=False):

            x_shot, x_query, y_shot, y_query = data

            x_shot = x_shot.cuda()
            y_shot = y_shot.cuda()

            x_query = x_query.cuda()
            y_query = y_query.cuda()

            if epoch == 1:

                grid = vutils.make_grid(x_shot[0], nrow=5, normalize=True)

                vutils.save_image(
                    grid,
                    os.path.join(vis_path, "support_epoch1.png")
                )

                grid = vutils.make_grid(x_query[0], nrow=5, normalize=True)

                vutils.save_image(
                    grid,
                    os.path.join(vis_path, "query_epoch1.png")
                )

            logits = model(
                x_shot,
                x_query,
                y_shot,
                inner_args,
                meta_train=True
            )

            logits = logits.flatten(0, 1)

            labels = y_query.flatten()

            pred = torch.argmax(logits, dim=-1)

            acc = utils.compute_acc(pred, labels)

            loss = F.cross_entropy(logits, labels)

            aves['tl'].update(loss.item(), 1)

            aves['ta'].update(acc, 1)

            if epoch % 5 == 0:

                img = x_query[0][0].detach().cpu().permute(1,2,0)

                plt.imshow(img)

                plt.title(
                    f"GT:{labels[0].item()} Pred:{pred[0].item()}"
                )

                plt.axis("off")

                plt.savefig(
                    os.path.join(vis_path, f"pred_epoch{epoch}.png")
                )

                plt.close()

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

        ###################################################
        # META VAL
        ###################################################

        if eval_val:

            model.eval()

            with torch.no_grad():

                for data in tqdm(val_loader, desc='meta-val', leave=False):

                    x_shot, x_query, y_shot, y_query = data

                    x_shot = x_shot.cuda()
                    y_shot = y_shot.cuda()

                    x_query = x_query.cuda()
                    y_query = y_query.cuda()

                    logits = model(
                        x_shot,
                        x_query,
                        y_shot,
                        inner_args,
                        meta_train=False
                    )

                    logits = logits.flatten(0,1)

                    labels = y_query.flatten()

                    pred = torch.argmax(logits, dim=-1)

                    acc = utils.compute_acc(pred, labels)

                    loss = F.cross_entropy(logits, labels)

                    aves['vl'].update(loss.item(), 1)

                    aves['va'].update(acc, 1)

        ###################################################
        # t-SNE
        ###################################################

        if epoch % 10 == 0:

            model.eval()

            with torch.no_grad():

                x = x_query.flatten(0,1)

                feat = model.encoder(x)

                feat = feat.detach().cpu().numpy()

                feat = feat.reshape(feat.shape[0], -1)
                n = feat.shape[0]
                perp = max(2, min(5, n-1))

                tsne = TSNE(n_components=2,perplexity=perp,init="random",learning_rate="auto",random_state=0)

                feat2d = tsne.fit_transform(feat)

                plt.figure()

                plt.scatter(
                    feat2d[:,0],
                    feat2d[:,1],
                    c=labels.cpu().numpy(),
                    cmap='tab10'
                )

                plt.title(f"t-SNE epoch {epoch}")

                plt.savefig(
                    os.path.join(vis_path,f"tsne_epoch{epoch}.png")
                )

                plt.close()

        ###################################################
        # Tensorboard
        ###################################################

        writer.add_scalar('loss/train', aves['tl'].item(), epoch)
        writer.add_scalar('acc/train', aves['ta'].item(), epoch)

        if eval_val:

            writer.add_scalar('loss/val', aves['vl'].item(), epoch)
            writer.add_scalar('acc/val', aves['va'].item(), epoch)

        ###################################################
        # Save model
        ###################################################

        torch.save(
            model.state_dict(),
            os.path.join(ckpt_path, 'epoch-last.pth')
        )

        va = aves['va'].item()
        if va > max_va:

            max_va = va

            torch.save(
                model.state_dict(),
                os.path.join(ckpt_path, 'max-va.pth')
            )

        for k in aves_keys:
            trlog[k].append(aves[k].item())

    ###################################################
    # Save curves
    ###################################################

    plt.figure()

    plt.plot(trlog['tl'], label="train_loss")
    plt.plot(trlog['vl'], label="val_loss")

    plt.legend()

    plt.savefig(os.path.join(curve_path,"loss_curve.png"))

    plt.close()

    plt.figure()

    plt.plot(trlog['ta'], label="train_acc")
    plt.plot(trlog['va'], label="val_acc")

    plt.legend()

    plt.savefig(os.path.join(curve_path,"acc_curve.png"))

    plt.close()

    df = pd.DataFrame(trlog)

    df.to_csv(os.path.join(ckpt_path,"training_log.csv"))


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--config')

    parser.add_argument('--name', default=None)

    parser.add_argument('--tag', default=None)

    parser.add_argument('--gpu', default='0')

    args = parser.parse_args()

    config = yaml.load(open(args.config, 'r'), Loader=yaml.FullLoader)

    utils.set_gpu(args.gpu)

    main(config)