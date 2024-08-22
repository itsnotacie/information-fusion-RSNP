"""
    Script for evaluating trained model on PyTorch / ImageNet-1K (demo mode).
"""

from args import args
import torch
from pytorchcv.model_provider import get_model as ptcv_get_model
from data.Data import CIFAR10
from trainer.trainer import validate


# def parse_args():
#     """
#     Create python script parameters.
#
#     Returns:
#     -------
#     ArgumentParser
#         Resulted args.
#     """
#     parser = argparse.ArgumentParser(
#         description="Evaluate an ImageNet-1K model on PyTorch (demo mode)",
#         formatter_class=argparse.ArgumentDefaultsHelpFormatter)
#     parser.add_argument(
#         "--input-size",
#         type=int,
#         default=224,
#         help="size of the input for model")
#     parser.add_argument(
#         "--mean-rgb",
#         nargs=3,
#         type=float,
#         default=(0.485, 0.456, 0.406),
#         help="Mean of RGB channels in the dataset")
#     parser.add_argument(
#         "--std-rgb",
#         nargs=3,
#         type=float,
#         default=(0.229, 0.224, 0.225),
#         help="STD of RGB channels in the dataset")
#
#     args = parser.parse_args()
#     return args


def main():
    """
    Main body of script.
    """
    # args = parse_args()
    models = {
        'alexn': 'alexnet',
        'v11': 'vgg11',
        'v13': 'vgg13',
        'v16': 'vgg16',
        'v19': 'vgg19',

    }

        # Create model with loading pretrained weights:
    name = ['resnet20_cifar10', 'resnet56_cifar10', 'resnet110_cifar10', 'preresnet20_cifar10', 'preresnet56_cifar10', 'preresnet110_cifar10', 'seresnet20_cifar10', 'seresnet56_cifar10', 'seresnet110_cifar10',
            'sepreresnet20_cifar10', 'sepreresnet56_cifar10', 'sepreresnet110_cifar10', 'pyramidnet110_a48_cifar10', 'pyramidnet110_a84_cifar10', 'pyramidnet110_a270_cifar10', 'wrn16_10_cifar10', 'wrn28_10_cifar10', 'wrn40_8_cifar10']
    net = ptcv_get_model('resnet20_cifar10', pretrained=True).cuda()
    dataset = CIFAR10()
    print(net)
    net.eval()
    criterion = torch.nn.CrossEntropyLoss()
    acc1, acc5 = validate(dataset.val_loader, net, criterion, args)
    print(acc1)





if __name__ == "__main__":
    main()