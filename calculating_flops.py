import torch
import argparse
from thop import profile
from trainer.amp_trainer_dali import validate_ImageNet
from trainer.trainer import validate
from utils.claculate_latency import compute_latency_ms_pytorch

from utils.get_dataset import get_dataset
from utils.get_model import get_model

parser = argparse.ArgumentParser(description='Calculating flops and params')

parser.add_argument(
    "--input_image_size",
    type=int,
    default=32,
    help='The input_image_size')
parser.add_argument("--gpu", default=None, type=int, help="Which GPU to use for training")
parser.add_argument("--arch", default=None, type=str, help="arch")
parser.add_argument("--pretrained", action="store_true", help="use pre-trained model")
parser.add_argument("--num_classes", default=10, type=int, help="number of class")
parser.add_argument("--finetune", action="store_true", help="finetune pre-trained model")
parser.add_argument("--set", help="name of dataset", type=str, default='cifar10')
parser.add_argument("--evaluate", dest="evaluate", action="store_true", help="evaluate model on validation set")
parser.add_argument("--print-freq", default=100, type=int, metavar="N", help="print frequency (default: 10)")
args = parser.parse_args()
torch.cuda.set_device(args.gpu)

model = get_model(args).cuda()
print(args)

# python calculating_flops.py --gpu 3 --arch cvgg11_bn_small --pretrained --evaluate

model.eval()
criterion = torch.nn.CrossEntropyLoss().cuda()
data = get_dataset(args)

if args.evaluate:
    if args.set in ['cifar10', 'cifar100']:
        acc1, acc5 = validate(data.val_loader, model, criterion, args)
    else:
        acc1, acc5 = validate_ImageNet(data.val_loader, model, criterion, args)

    print('Acc is {}'.format(acc1))


# calculate model size
input_image_size = args.input_image_size
print('image size is {}'.format(input_image_size))
input_image = torch.randn(1, 3, input_image_size, input_image_size).cuda()
flops, params = profile(model, inputs=(input_image,))
latency = compute_latency_ms_pytorch(model, input_image, iterations=None)

print('Params: %.2f' % (params))
print('Flops: %.2f' % (flops))
print('Latency: %.2f' % (latency))