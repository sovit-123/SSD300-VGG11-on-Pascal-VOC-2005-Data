import time
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import argparse

from model import SSD300, MultiBoxLoss
from datasets import PascalVOCDataset
from utils import label_map, adjust_learning_rate, save_checkpoint
from utils import AverageMeter, clip_gradient
from datetime import datetime

# construct the argument parser and parse the command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input', help='path to input data folder')
parser.add_argument('-t', '--continue-training', dest='continue_training', 
                        required=True, choices=['yes', 'no'],
                        help='whether to continue training or not')
args = vars(parser.parse_args())

# data parameters
data_folder = args['input']  # folder with data files

# model parameters
n_classes = len(label_map)  # number of different types of objects
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# learning parameters
if args['continue_training'] == 'yes': # continue training or not
    checkpoint = '../model_checkpoints/checkpoint_ssd300_vgg11.pth.tar'
else:
    print('Training from beginning')
    checkpoint = None
batch_size = 16  # batch size
iterations = 18000  # number of iterations to train
workers = 4  # number of workers for loading data in the DataLoader
print_freq = 20  # print training status every __ batches
lr = 1e-3  # learning rate
decay_lr_at = [13000, 14800, 16000]  # decay learning rate after these many iterations
decay_lr_to = 0.1  # decay learning rate to this fraction of the existing learning rate
momentum = 0.9  # momentum
weight_decay = 5e-4  # weight decay
grad_clip = None  # clip if gradients are exploding, which may happen at larger batch sizes (sometimes at 32) - you will recognize it by a sorting error in the MuliBox loss calculation

cudnn.benchmark = True

def main():
    """
    Training.
    """
    global start_epoch, label_map, epoch, checkpoint, decay_lr_at

    # initialize model or load checkpoint
    if checkpoint is None:
        start_epoch = 0
        model = SSD300(n_classes=n_classes)
        # initialize the optimizer, with twice the default learning rate...
        # ...for biases, as in the original Caffe repo
        biases = list()
        not_biases = list()
        for param_name, param in model.named_parameters():
            if param.requires_grad:
                if param_name.endswith('.bias'):
                    biases.append(param)
                else:
                    not_biases.append(param)
        optimizer = torch.optim.SGD(params=[{'params': biases, 'lr': 2 * lr}, {'params': not_biases}],
                                    lr=lr, momentum=momentum, weight_decay=weight_decay)

    else:
        checkpoint = torch.load(checkpoint)
        start_epoch = checkpoint['epoch'] + 1
        print('\nLoaded checkpoint from epoch %d.\n' % start_epoch)
        model = checkpoint['model']
        optimizer = checkpoint['optimizer']

    # move to default device
    model = model.to(device)
    criterion = MultiBoxLoss(priors_cxcy=model.priors_cxcy).to(device)

    # custom dataloaders
    train_dataset = PascalVOCDataset(data_folder,
                                     split='train')
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                               collate_fn=train_dataset.collate_fn, num_workers=workers,
                                               pin_memory=True)  # using `collate_fn()` here

    # calculate total number of epochs to train and the epochs to decay...
    # ...learning rate at (i.e. convert iterations to epochs)
    # to convert iterations to epochs,...
    # ...divide iterations by the number of iterations per epoch
    epochs = iterations // (len(train_dataset) // batch_size)
    decay_lr_at = [it // (len(train_dataset) // batch_size) for it in decay_lr_at]
    print(f"Training for {iterations} iterations...")
    print(f"Training for {epochs} epochs...")
    print(f"Batch size is {batch_size}")
    print(f"Logging every {print_freq} batches...")
    with open(file='../logs/train_logs.txt', mode='a+') as f:
        f.writelines(f"Training for {iterations} iterations...\n")
        f.writelines(f"Training for {epochs} epochs...\n")
        f.writelines(f"Batch size is {batch_size}\n")
        f.writelines(f"Logging every {print_freq} batches...\n")

    # epochs
    for epoch in range(start_epoch, epochs):

        # decay learning rate at particular epochs
        if epoch in decay_lr_at:
            adjust_learning_rate(optimizer, decay_lr_to)
            with open(file='../logs/train_logs.txt', mode='a+') as f:
                f.writelines(f"DECAYING learning rate.\n The new LR is {(optimizer.param_groups[1]['lr'],)}\n")

        # one epoch's training
        train(train_loader=train_loader,
              model=model,
              criterion=criterion,
              optimizer=optimizer,
              epoch=epoch)

        # save checkpoint after each epoch
        save_checkpoint(epoch, model, optimizer)


def train(train_loader, model, criterion, optimizer, epoch):
    """
    One epoch's training.

    :param train_loader: DataLoader for training data
    :param model: model
    :param criterion: MultiBox loss
    :param optimizer: optimizer
    :param epoch: epoch number
    """
    model.train()  # training mode enables dropout

    batch_time = AverageMeter()  # forward prop. + back prop. time
    data_time = AverageMeter()  # data loading time
    losses = AverageMeter()  # loss

    start = time.time()

    # batches
    for i, (images, boxes, labels) in enumerate(train_loader):
        data_time.update(time.time() - start)

        # move to default device
        images = images.to(device)  # (batch_size (N), 3, 300, 300)
        boxes = [b.to(device) for b in boxes]
        labels = [l.to(device) for l in labels]

        # forward prop.
        predicted_locs, predicted_scores = model(images)  # (N, 8732, 4), (N, 8732, n_classes)

        # loss
        loss = criterion(predicted_locs, predicted_scores, boxes, labels)  # scalar

        # backward prop.
        optimizer.zero_grad()
        loss.backward()

        # clip gradients, if necessary
        if grad_clip is not None:
            clip_gradient(optimizer, grad_clip)

        # update model
        optimizer.step()

        losses.update(loss.item(), images.size(0))
        batch_time.update(time.time() - start)

        start = time.time()

        # print status
        if i % print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data Time {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(epoch, i, len(train_loader),
                                                                  batch_time=batch_time,
                                                                  data_time=data_time, loss=losses))
            with open(file='../logs/train_logs.txt', mode='a+') as f:
                f.writelines('\nEpoch: [{0}][{1}/{2}]\t'
                  'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data Time {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(epoch, i, len(train_loader),
                                                                  batch_time=batch_time,
                                                                  data_time=data_time, loss=losses))
    del predicted_locs, predicted_scores, images, boxes, labels  # free some memory since their histories may be stored


if __name__ == '__main__':
    with open(file='../logs/train_logs.txt', mode='a+') as f:
        f.writelines(f"##### ----- ##### ----- ##### \n\n")
        f.writelines(f"NEW RUN: ({datetime.now()}), \n")
    print(f"Training on labels: {label_map}\n")
    main()