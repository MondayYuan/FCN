import numpy as np
import torch
from PIL import Image

from torch import nn
from torch.optim import Adam, RMSprop, SGD
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, CenterCrop, Normalize
from torchvision.transforms import ToTensor, ToPILImage, RandomCrop

from utils.dataset import VOC12
from utils.network import FCN_VGG
from utils.transform import Relabel, ToLabel, Colorize
from utils.visualize import Dashboard

import pickle
import time
import os

from argparse import ArgumentParser

NUM_CHANNELS = 3
NUM_CLASSES = 21

def train(args, model):
    model.train()
    
    train_set = VOC12(args.datadir, args.train_list)
    val_set = VOC12(args.datadir, args.val_list)

    train_loader = DataLoader(train_set,
        num_workers=args.num_workers, batch_size=args.batch_size, shuffle=True)

    criterion = nn.CrossEntropyLoss()
    if args.cuda:
        criterion = criterion.cuda()

    if args.model == 'fcn-vgg19-deconv':
        optimizer = SGD(model.parameters(), 5e-2, 2e-5)
    elif args.model in ['fcn-vgg16-interpolate', 'fcn-vgg19-interpolate']:
        optimizer = SGD(model.parameters(), 1e-3)
    
    epochs_losses = []
    macc_list = []
    miu_list = []

    for epoch in range(args.num_epochs):
        model.train()
        epoch_loss = 0
        for step, (images, lables) in enumerate(train_loader):
            inputs = images
            targets = lables[:, 0]
            if args.cuda:
                inputs = inputs.cuda()
                targets = targets.cuda()
            outputs = model(inputs)
            # print(inputs.shape, outputs.shape)
            optimizer.zero_grad()
            # print(outputs.shape, targets.shape)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            epoch_loss += loss

        epochs_losses.append(epoch_loss / (args.batch_size * len(train_loader)))
        if epoch % args.epochs_eval == 0:
            macc, MIU = evaluate(args, model, val_set, epoch)
            macc_list.append(macc)
            miu_list.append(MIU)
            time_stamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            print('epoch ={}, loss = {}, macc = {}, mean_iu = {} ---{}'.format(epoch, epoch_loss, macc, MIU, time_stamp))
        if epoch % args.epochs_save == 0:
            filename = os.path.join(args.log_dir, f'{args.model}.pth') 
            torch.save(model.state_dict(), filename)
            print(f'save: {filename} (epoch: {epoch})')

            with open(os.path.join(args.log_dir,'loss.pkl'), 'wb') as f:
                pickle.dump(epochs_losses, f)
            with open(os.path.join(args.log_dir,'macc.pkl'), 'wb') as f:
                pickle.dump(macc_list, f)
            with open(os.path.join(args.log_dir,'miu.pkl'), 'wb') as f:
                pickle.dump(miu_list, f)

def evaluate(args, model, val_set, epoch):
    model.eval()
    val_loader = DataLoader(val_set,
        num_workers=1, batch_size=1, shuffle=False)
    labels_truth = []
    labels_predict = []
    for step, (images, labels) in enumerate(val_loader):
        inputs = images
        targets = labels[:, 0]
        if args.cuda:
            inputs = inputs.cuda()
            targets = targets.cuda()
        outputs = model(inputs).max(1)[1]

        labels_truth.append(targets.cpu().numpy())
        labels_predict.append(outputs.cpu().numpy())
        
        if args.port > 0 and step == 0:
                image = inputs[0]
                image[0] = image[0] * .229 + .485
                image[1] = image[1] * .224 + .456
                image[2] = image[2] * .225 + .406

                board = Dashboard(args.port)

                board.image(image,
                    f'input (epoch: {epoch})')
                board.image(Colorize()(outputs[0].cpu()),
                    f'output (epoch: {epoch})')
                board.image(Colorize()(targets[0].cpu()),
                    f'target (epoch: {epoch})')
    
    def label_accuracy_score(label_trues, label_preds, n_class):
        """Returns accuracy score evaluation result.
        - overall accuracy
        - mean accuracy
        - mean IU
        - fwavacc
        """
        hist = np.zeros((n_class, n_class))

        def _fast_hist(label_true, label_pred, n_class):
            mask = (label_true >= 0) & (label_true < n_class)
            hist = np.bincount(
                n_class * label_true[mask].astype(int) +
                label_pred[mask], minlength=n_class ** 2).reshape(n_class, n_class)
            return hist

        for lt, lp in zip(label_trues, label_preds):
            hist += _fast_hist(lt.flatten(), lp.flatten(), n_class)
        acc = np.diag(hist).sum() / hist.sum()
        with np.errstate(divide='ignore', invalid='ignore'):
            acc_cls = np.diag(hist) / hist.sum(axis=1)
        acc_cls = np.nanmean(acc_cls)
        with np.errstate(divide='ignore', invalid='ignore'):
            iu = np.diag(hist) / (
                hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist)
            )
        mean_iu = np.nanmean(iu)
        freq = hist.sum(axis=1) / hist.sum()
        fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
        return acc, acc_cls, mean_iu, fwavacc

    acc, acc_cls, mean_iu, fwavacc = label_accuracy_score(labels_truth, labels_predict, NUM_CLASSES)

    return acc_cls, mean_iu
    

def test(args, model):
    model.eval()

    input_image = Image.open(args.image).resize((256, 256))

    input_transform = Compose([
        ToTensor(),
        Normalize([.485, .456, .406], [.229, .224, .225]),
    ])

    image = torch.unsqueeze(input_transform(input_image), 0)
    if args.cuda:
        image = image.cuda()
    
    label = model(image)
    color_transform = Colorize()
    label = color_transform(label[0].data.max(0)[1])
    label = ToPILImage()(label)

    # label.show()
    # input_image.show()

    if args.resized_image:
        input_image.save(args.resized_image)
    label.save(args.label)


def main(args):
    model = None
    if args.model == 'fcn-vgg16-interpolate':
        model = FCN_VGG(NUM_CLASSES, 'vgg16', 'interpolate')
    if args.model == 'fcn-vgg19-interpolate':
        model = FCN_VGG(NUM_CLASSES, 'vgg19', 'interpolate')
    if args.model == 'fcn-vgg19-deconv':
        model = FCN_VGG(NUM_CLASSES, 'vgg19', 'deconv')

    assert model is not None, f'model {args.model} not available'

    if args.cuda:
        model = model.cuda()
    
    if args.double_cudas:
        model = nn.DataParallel(model, [0, 1])
    
    if args.state:
        model.load_state_dict(torch.load(args.state))

    if args.mode == 'test':
        test(args, model)
    if args.mode == 'train':
        train(args, model)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--cuda', action='store_true', default=False)
    parser.add_argument('--double-cudas', action='store_true', default=False)
    parser.add_argument('--model', required=True)
    parser.add_argument('--state')

    subparsers = parser.add_subparsers(dest='mode')
    subparsers.required = True

    parser_eval = subparsers.add_parser('test')
    parser_eval.add_argument('image')
    parser_eval.add_argument('label')
    parser_eval.add_argument('--resized_image')

    parser_train = subparsers.add_parser('train')
    parser_train.add_argument('--port', type=int, default=-1, help='visdom\'s port')
    parser_train.add_argument('--datadir', default='./data', help='dir to store images and labels')
    parser_train.add_argument('--train_list', default='./data/my_train.txt', help='the images list of the training set')
    parser_train.add_argument('--val_list', default='./data/my_val.txt', help='the images list of the validation set')
    parser_train.add_argument('--num-epochs', type=int, default=100)
    parser_train.add_argument('--num-workers', type=int, default=4, help='number of threads used to load the training set')
    parser_train.add_argument('--batch-size', type=int, default=1)
    parser_train.add_argument('--epochs-eval', type=int, default=1, help='number of epochs between evaluation')
    parser_train.add_argument('--epochs-save', type=int, default=1, help='number of epochs between saving')
    parser_train.add_argument('--log_dir', required=True)

    main(parser.parse_args())
