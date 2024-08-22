import os
import sys
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from torch import nn
from Hrank_resnet import resnet_110
from args import args
import datetime
from data.Data import CIFAR100
from trainer.trainer import validate, train
from utils.utils import set_random_seed, set_gpu, Logger, get_logger, get_lr


def main():
    print(args)
    sys.stdout = Logger('print process.log', sys.stdout)

    if args.random_seed is not None:
        set_random_seed(args.random_seed)

    main_worker(args)


def main_worker(args):
    now = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    if not os.path.isdir('pretrained_model/' + args.arch + '/' + args.set):
        os.makedirs('pretrained_model/' + args.arch + '/' + args.set, exist_ok=True)
    logger = get_logger('pretrained_model/' + args.arch + '/' + args.set + '/logger' + now + '.log')
    logger.info(args.arch)
    logger.info(args.set)
    logger.info(args.batch_size)
    logger.info(args.weight_decay)
    logger.info(args.lr)
    logger.info(args.epochs)
    logger.info(args.lr_decay_step)
    logger.info(args.num_classes)

    # model = my_VGG16(num_classes=10)
    # model = cvgg7_bn(num_classes=10)
    # model = googlenet([0.35]+[0.6]*2+[0.75]*5+[0.75]*2)  # we [0.32]+[0.65]*2+[0.72]*5+[0.8]*2,  other:[0.35]+[0.6]*2+[0.75]*5+[0.75]*2, [0.5]+[0.6]*2+[0.74]*5+[0.77]*2
    model = resnet_110(compress_rate=[0.]+[0.22]*2+[0.3]*18+[0.32]*36)  # ori [0.]+[0.2]*2+[0.3]*18+[0.35]*36   ori_resnet110_scores5: [0.]+[0.2]*2+[0.32]*18+[0.3]*36  ori_resnet110_scores6: [0.]+[0.22]*2+[0.3]*18+[0.32]*36
    model = set_gpu(args, model)
    logger.info(model)
    criterion = nn.CrossEntropyLoss().cuda()
    data = CIFAR100()

    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    # multi lr
    lr_decay_step = list(map(int, args.lr_decay_step.split(',')))
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=lr_decay_step, gamma=0.1)

    best_acc1 = 0.0
    best_acc5 = 0.0
    best_train_acc1 = 0.0
    best_train_acc5 = 0.0
    acc_list = []
    trainacc_list = []

    # create recorder
    args.start_epoch = args.start_epoch or 0

    # Start training
    for epoch in range(args.start_epoch, args.epochs):
        train_acc1, train_acc5 = train(data.train_loader, model, criterion, optimizer, epoch, args)
        acc1, acc5 = validate(data.val_loader, model, criterion, args)
        acc_list.append(acc1)
        trainacc_list.append(train_acc1)

        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)
        best_acc5 = max(acc5, best_acc5)
        best_train_acc1 = max(train_acc1, best_train_acc1)
        best_train_acc5 = max(train_acc5, best_train_acc5)
        save = ((epoch % args.save_every) == 0) and args.save_every > 0
        if is_best or save or epoch == args.epochs - 1:
            if is_best:
                torch.save(model.state_dict(), 'pretrained_model/' + args.arch + '/' + args.set + "/ori_scores_{}.pt".format(args.arch))
                logger.info(best_acc1)

        scheduler.step()

if __name__ == "__main__":
    # setup: python train.py --gpu 1 --arch cvgg16 --set cifar10 --lr 0.01 --batch_size 256 --weight_decay 0.005 --epochs 150 --lr_decay_step 50,100  --num_classes 10
    main()

