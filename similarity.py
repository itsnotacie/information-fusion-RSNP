import datetime
import os
import torch
import tqdm
from args import args
from cka import linear_CKA
from trainer.amp_trainer_dali import validate_ImageNet
from trainer.trainer import validate
from utils.get_model import get_model
from utils.get_hook import get_inner_feature_for_resnet, get_inner_feature_for_vgg, get_inner_feature_for_smallresnet
from utils.get_dataset import get_dataset
from utils.utils import set_gpu, get_logger

'''
# setup up:
python similarity.py --gpu 0 --arch resnet56 --set cifar10 --num_classes 10 --batch_size 500 --pretrained  --evaluate
'''

def main():
    assert args.pretrained, 'this program needs pretrained model'
    now = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    if not os.path.isdir('experiment/' + args.arch + '/' + 'Modularity_%s' % args.set):
        os.makedirs('experiment/' + args.arch + '/' + 'Modularity_%s' % args.set, exist_ok=True)
    logger = get_logger('experiment/' + args.arch + '/' + 'Modularity_%s' % args.set + '/logger' + now + '.log')
    logger.info(args)
    model = get_model(args)
    # print(model)

    model = set_gpu(args, model)
    criterion = torch.nn.CrossEntropyLoss().cuda()
    data = get_dataset(args)
    model.eval()
    batch_count = 0
    # if args.evaluate:
    #     if args.set in ['cifar10', 'cifar100']:
    #         acc1, acc5 = validate(data.val_loader, model, criterion, args)
    #     else:
    #         acc1, acc5 = validate_ImageNet(data.val_loader, model, criterion, args)
    #
    # logger.info(acc1)

    inter_feature = []
    CKA_matrix_list = []
    def hook(module, input, output):
        inter_feature.append(output.clone().detach())

    with torch.no_grad():
        for i, data in tqdm.tqdm(
                enumerate(data.val_loader), ascii=True, total=len(data.val_loader)
        ):
            batch_count += 1
            if args.set == 'imagenet_dali':
                images = data[0]["data"].cuda(non_blocking=True)
                target = data[0]["label"].squeeze().long().cuda(non_blocking=True)
            else:
                images, target = data[0].cuda(args.gpu, non_blocking=True), data[1].cuda(args.gpu, non_blocking=True)

            if args.arch in ['resnet20', 'resnet32', 'resnet44', 'resnet56', 'resnet110']:
                handle_list = get_inner_feature_for_smallresnet(model, hook, args.arch)
            elif args.arch in ['ResNet18', 'ResNet34', 'ResNet50', 'ResNet101', 'ResNet152']:
                handle_list = get_inner_feature_for_resnet(model, hook, args.arch)
            else:
                handle_list = get_inner_feature_for_vgg(model, hook, args.arch)

            output = model(images)
            for m in range(len(inter_feature)):
                print('-'*50)
                print(m)
                if len(inter_feature[m].shape) != 2:
                    inter_feature[m] = inter_feature[m].reshape(args.batch_size, -1)

            CKA_matrix_for_visualization = CKA_heatmap(inter_feature)
            print(CKA_matrix_for_visualization)
            CKA_matrix_list.append(CKA_matrix_for_visualization)

            inter_feature = []
            for i in range(len(handle_list)):
                handle_list[i].remove()

            if batch_count == 5:
                break


    torch.save(CKA_matrix_list, 'save/CKA_matrix_for_visualization_{}_{}.pth'.format(args.arch, args.set))


def CKA_heatmap(inter_feature):
    layer_num = len(inter_feature)
    CKA_matrix = torch.zeros((layer_num, layer_num))
    for ll in range(layer_num):
        for jj in range(layer_num):
            if ll < jj:
                CKA_matrix[ll, jj] = CKA_matrix[jj, ll] = linear_CKA(inter_feature[ll], inter_feature[jj])
                # CKA_matrix[ll, jj] = CKA_matrix[jj, ll] = unbias_CKA(inter_feature[ll], inter_feature[jj])

    CKA_matrix_for_visualization = CKA_matrix + torch.eye(layer_num)
    return CKA_matrix_for_visualization


if __name__ == "__main__":
    main()
