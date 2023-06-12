import sys
import time
import random
import argparse
import numpy as np
from seg_utils.utils import AverageMeter
from seg_utils.dataset_utils import dataset, split_dataset

import torch
from torchvision import transforms
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp
import torchvision.transforms.functional as TF

def train(model, device, train_loader, optimizer, criterion, epoch):
    acc = AverageMeter()
    iou = AverageMeter()
    losses = AverageMeter()

    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        target = target.unsqueeze(1)

        output = model(data)
        loss = criterion(output, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        target = target.long()
        tp, fp, fn, tn = smp.metrics.get_stats(output, target, 'binary', threshold=0.5)
        iou_value = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")
        accuracy = smp.metrics.accuracy(tp, fp, fn, tn, reduction="micro")

        losses.update(loss)
        acc.update(accuracy)
        iou.update(iou_value)
        
    sys.stdout.write('\r')
    sys.stdout.write('Epoch [%3d/%3d] loss: %.4f Accuracy: %.4f, IoU: %.4f\n' % (epoch, args.epoch, loss, acc.avg, iou.avg))
    sys.stdout.flush()

    return loss.item(), acc.avg, iou.avg


def test(model, device, test_loader):
    acc = AverageMeter()
    iou = AverageMeter()
    latency = AverageMeter()

    model.eval()
    with torch.no_grad():
        torch.cuda.synchronize()
        torch.cuda.synchronize()
        t_start = time.time()

        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            target = target.unsqueeze(1)

            output = model(data)

            target = target.long()
            tp, fp, fn, tn = smp.metrics.get_stats(output, target, 'binary', threshold=0.5)
            iou_value = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")
            accuracy = smp.metrics.accuracy(tp, fp, fn, tn, reduction="micro")

            ori_image = data * 255.0
            output_image = output * 255.0

            ori_image[:, 0] = output_image

            # 두 배열을 RGB 이미지로 결합
            ori_image = TF.to_pil_image(ori_image.squeeze().byte(), mode='RGB')
            ori_image.save("overlap/output_image" + str(batch_idx) + ".jpg")

            output_image = TF.to_pil_image(output_image.squeeze().byte(), mode='L')
            output_image.save("results/output_image" + str(batch_idx) + ".jpg")

            acc.update(accuracy)
            iou.update(iou_value)

        torch.cuda.synchronize()
        torch.cuda.synchronize()
        latency_time = time.time() - t_start    # μs ()
        torch.cuda.empty_cache()

        latency.update(latency_time, data.size(0))
    
    sys.stdout.write('\r')
    sys.stdout.write("Test | Accuracy: %.4f, IoU: %.4f, Latency: %.2f\n" % (acc.avg, iou.avg, latency.avg))
    sys.stdout.flush()

    return acc.avg, iou.avg, latency.avg


parser = argparse.ArgumentParser()
# General Settings
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--device', type=str, default='0')
# Dataset Settings
parser.add_argument('--root', type=str, default='train_data')
parser.add_argument('--batch_size', type=int, default=512)
parser.add_argument('--test_size', type=int, default=256)
parser.add_argument('--num_workers', type=int, default=8)
parser.add_argument('--split', type=int, default=0.8)
# Model Settings
parser.add_argument('--epoch', type=int, default=10)
parser.add_argument('--lr', '--learning_rate', type=float, default=0.001)

args = parser.parse_args()

def main():
    device = 'cuda:' + args.device
    args.device = torch.device(device)
    torch.cuda.set_device(args.device)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    cudnn.deterministic = True
    cudnn.benchmark = False

    ENCODER = 'resnet101'
    ENCODER_WEIGHTS = 'imagenet'
    ACTIVATION = 'sigmoid'

    model = smp.DeepLabV3Plus(
        encoder_name=ENCODER,
        encoder_weights=ENCODER_WEIGHTS,
        activation=ACTIVATION,
    )
    model.to(device)

    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
    ])

    datasets = dataset(args.root, transform=transform)
    train_dataset, test_dataset = split_dataset(datasets, args.split)

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=args.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=args.num_workers)

    loss = smp.losses.DiceLoss('binary')
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=1, T_mult=2, eta_min=5e-5,)

    for epoch in range(1, args.epoch + 1):
        train_loss, train_acc, train_iou = train(model, args.device, train_loader, optimizer, loss, epoch)

    test_acc, test_iou, latency = test(model, args.device, test_loader)
    print("FPS:{:.2f}".format(1000./latency))
    print("Latency:{:.2f}ms / {:.4f}s".format(latency, (latency/1000.)))


if __name__ == '__main__':
    main()