import copy
import datetime
import os
from itertools import combinations
from args import args
from model.VGG_cifar import cvgg16_bn
from model.samll_resnet import resnet56
from trainer.amp_trainer_dali import validate_ImageNet
from trainer.trainer import validate
from utils.get_dataset import get_dataset
from utils.get_model import get_model
from utils.get_params import get_layer_params
from utils.utils import set_random_seed, set_gpu, get_logger
import torch
from zero_nas import ZeroNas
from torchvision.models import *

## python Reassembly.py --pretrained --set cifar10 --num_classes 10 --batch_size 64  --arch resnet56 --gpu 0  --zero_proxy grad_norm --evaluate
## python Reassembly.py --pretrained --set cifar100 --num_classes 100 --batch_size 64  --arch resnet56 --gpu 0  --zero_proxy grad_norm --evaluate
## python Reassembly.py --pretrained --set cifar10 --num_classes 10 --batch_size 64  --arch cvgg16_bn --gpu 0  --zero_proxy grad_norm --evaluate
## python Reassembly.py --pretrained --set imagenet_dali --num_classes 1000 --batch_size 64  --arch ResNet50 --gpu 0  --zero_proxy grad_norm --evaluate

resnet56_c10_k3_partition = [1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3]
resnet56_c10_k4_partition = [1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 4, 4]
resnet56_c10_k5_partition = [1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 5, 5] #
resnet56_c10_k6_partition = [1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 5, 5, 6, 6]
resnet56_c10_k7_partition = [1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 6, 6, 7, 7]
resnet56_c100_k7_partition = [1, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 6, 6, 7, 7, 7, 7] #
resnet50_c10_k4_partition = [1, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4]
# cvgg16_k3_partition = [1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3]  # vgg16 cifar10
cvgg16_k4_partition = [1, 1, 2, 2, 3, 3, 3, 3, 3, 3, 4, 4, 4]  # vgg16 cifar10
cvgg16_layer_name = ['features.0', 'features.3', 'features.7', 'features.10', 'features.14', 'features.17', 'features.20',
               'features.24', 'features.27', 'features.30', 'features.34', 'features.37', 'features.40']
layer_num = {'resnet56': 28, 'resnet44': 22, 'resnet32': 18, 'resnet20': 10, 'cvgg16_bn': 13, 'resnet110': 55}

layer_list = {
    'ResNet50': [3, 4, 6, 3],
    'ResNet101': [3, 4, 23, 3]
}

def get_layer(partition):
    max_part = max(partition)
    print("Max partition: {}".format(max_part))

    block = []
    for i in range(max_part):
        block.append([])

    for i in range(len(partition)):
        for j in range(max_part):
            if partition[i] == j+1:
                block[j].append(i)

    return block, max_part


def main():
    if args.random_seed is not None:
        set_random_seed(args.random_seed)

    assert args.pretrained, 'this program needs pretrained model'
    now = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    if not os.path.isdir('experiment/' + args.arch + '/' + 'Modularity_%s' % args.set):
        os.makedirs('experiment/' + args.arch + '/' + 'Modularity_%s' % args.set, exist_ok=True)
    logger = get_logger('experiment/' + args.arch + '/' + 'Modularity_%s' % args.set + '/logger' + now + '.log')
    logger.info(args)
    model = get_model(args)

    model = set_gpu(args, model)
    criterion = torch.nn.CrossEntropyLoss().cuda()
    data = get_dataset(args)
    model.eval()

    # if args.evaluate:
    #     if args.set in ['cifar10', 'cifar100']:
    #         acc1, acc5 = validate(data.val_loader, model, criterion, args)
    #     else:
    #         acc1, acc5 = validate_ImageNet(data.val_loader, model, criterion, args)
    #
    # logger.info(acc1)
    logger.info(model)
    if args.arch == 'resnet56':
        if args.set == 'cifar10':
            # remain_layer = [3, 3, 4, 2, 2]  # 需要调整
            remain_layer = [3, 4, 5, 2, 2]  # 需要调整
        if args.set == 'cifar100':
            remain_layer = [1, 2, 4, 4, 2, 2, 4]  # 需要调整

    if args.arch == 'ResNet50':
        if args.set == 'imagenet_dali':
            remain_layer = [1, 4, 6, 2]  # 需要调整

    elif args.arch == 'cvgg16_bn':
        # remain_layer = [2, 3, 3]  # 需要调整 k3
        remain_layer = [2, 2, 3, 1]  # 需要调整
    value_list = replace_layer_initialization(data, args, remain_layer)
    print(value_list)
    search_best(value_list)
    # torch.save(value_list, "save/value_{}_{}_{}_{}-{}-{}-{}-{}.pth".format(args.arch, args.set, args.zero_proxy, remain_layer[0], remain_layer[1], remain_layer[2], remain_layer[3], remain_layer[4]))


def replace_layer_initialization(data, args, remain_layer):
    """ Run the methods on the data and then saves it to out_path. """
    if args.arch == 'resnet56':
        model1 = resnet56(num_classes=args.num_classes)
        model1 = set_gpu(args, model1)
        model2 = resnet56(num_classes=args.num_classes)
        model2 = set_gpu(args, model2)
        if args.set == 'cifar10':
            save = torch.load('/public/ly/Dynamic_Graph_Construction/pretrained_model/resnet56.th', map_location='cuda:%d' % args.gpu)
            ckpt = {k.replace('module.', ''): v for k, v in save['state_dict'].items()}
        elif args.set == 'cifar100':
            ckpt = torch.load('/public/ly/Dynamic_Graph_Construction/pretrained_model/resnet56/cifar100/scores.pt', map_location='cuda:%d' % args.gpu)

        model1.load_state_dict(ckpt)
        model1.eval()
        model2.eval()
        # acc1, acc5 = validate(data.val_loader, model1, criterion, args)  # pretrained
        # acc1, acc5 = validate(data.val_loader, model2, criterion, args)  # random
        random_state_list = get_layer_params(model1)
        if args.set == 'cifar10':
            k5_partition = resnet56_c10_k5_partition

        if args.set == 'cifar100':
            k5_partition = resnet56_c100_k7_partition

        block, max_part = get_layer(k5_partition)
        value_list = [[] for i in range(max_part)]
        for iii in range(max_part):
            for p in combinations(block[iii], remain_layer[iii]):
                x = (layer_num[args.arch] - 1) // 3
                for i in p:
                    if i == 0:
                        model2.conv1.load_state_dict(random_state_list[0])
                        model2.bn1.load_state_dict(random_state_list[1])
                    elif 0 < i <= x:
                        model2.layer1[i - 1].load_state_dict(random_state_list[i + 1])
                        # print(model1.layer1[i - 1])
                    elif x < i <= 2 * x:
                        model2.layer2[i - 1 - x].load_state_dict(random_state_list[i + 1])
                        # print(model1.layer2[i - 1 - x])
                    elif 2 * x < i <= 3 * x:
                        model2.layer3[i - 1 - (2 * x)].load_state_dict(random_state_list[i + 1])
                        # print(model1.layer3[i - 1 - (2 * x)])

                indicator = ZeroNas(dataloader=data.train_loader, indicator=args.zero_proxy, num_batch=args.num_batch)
                value = indicator.get_score(model2)[args.zero_proxy]
                value_list[iii].append((p, value))

    elif args.arch == 'cvgg16_bn':
        model1 = cvgg16_bn(num_classes=args.num_classes, batch_norm=True)
        model1 = set_gpu(args, model1)
        model2 = cvgg16_bn(num_classes=args.num_classes, batch_norm=True)
        model2 = set_gpu(args, model2)
        if args.set == 'cifar10':
            ckpt = torch.load('/public/ly/Dynamic_Graph_Construction/pretrained_model/cvgg16_bn/cifar10/scores.pt', map_location='cuda:%d' % args.gpu)
        elif args.set == 'cifar100':
            ckpt = torch.load('/public/ly/Dynamic_Graph_Construction/pretrained_model/cvgg16_bn/cifar100/scores.pt',map_location='cuda:%d' % args.gpu)

        model1.load_state_dict(ckpt)
        model1.eval()
        model2.eval()
        # acc1, acc5 = validate(data.val_loader, model1, criterion, args)  # pretrained
        # acc1, acc5 = validate(data.val_loader, model2, criterion, args)  # random
        block, max_part = get_layer(cvgg16_k4_partition)
        value_list = [[] for i in range(max_part)]
        for iii in range(max_part):
            for p in combinations(block[iii], remain_layer[iii]):
                for uuu in p:
                    model2.state_dict()['{}.weight'.format(cvgg16_layer_name[uuu])] = model1.state_dict()['{}.weight'.format(cvgg16_layer_name[uuu])]
                    model2.state_dict()['{}.bias'.format(cvgg16_layer_name[uuu])] = model1.state_dict()['{}.bias'.format(cvgg16_layer_name[uuu])]

                model2.load_state_dict(model2.state_dict())
                indicator = ZeroNas(dataloader=data.train_loader, indicator=args.zero_proxy, num_batch=args.num_batch)
                value = indicator.get_score(model2)[args.zero_proxy]
                value_list[iii].append((p, value))

    elif args.arch == 'ResNet50':
        model1 = resnet50(pretrained=True)
        model1 = set_gpu(args, model1)
        model2 = resnet50(pretrained=False)
        model2 = set_gpu(args, model2)
        model1.eval()
        model2.eval()

        random_state_list = get_layer_params(model1)
        block, max_part = get_layer(resnet50_c10_k4_partition)
        value_list = [[] for i in range(max_part)]
        for iii in range(max_part):
            for p in combinations(block[iii], remain_layer[iii]):
                for i in p:
                    if i == 0:
                        model2.conv1.load_state_dict(random_state_list[0])
                        model2.bn1.load_state_dict(random_state_list[1])
                    elif 0 < i <= 3: # 1,2,3
                        model2.layer1[i - 1].load_state_dict(random_state_list[i + 1])
                    elif 3 < i <= 7: # 4,5,6,7 -- 0,1,2,3
                        model2.layer2[i - 4].load_state_dict(random_state_list[i + 1])
                    elif 7 < i <= 13: # 8,9,10,11,12,13 -- 0,1,2,3,4,5
                        model2.layer3[i - 8].load_state_dict(random_state_list[i + 1])
                    elif 13 < i <= 16: # 14,15,16 -- 0,1,2
                        model2.layer4[i - 14].load_state_dict(random_state_list[i + 1])

                indicator = ZeroNas(dataloader=data.train_loader, indicator=args.zero_proxy, num_batch=args.num_batch)
                value = indicator.get_score(model2)[args.zero_proxy]
                value_list[iii].append((p, value))

    else:
        assert "the model has not prepared"

    return value_list


def search_best(value_list):
    num = len(value_list)
    for i in range(num):
        best = -100
        best_layer = None
        for j in value_list[i]:
            if best < j[1]:
                best = j[1]
                best_layer = j[0]

        print("Remaining layers: {}".format(best_layer))

    return

if __name__ == "__main__":
    main()
