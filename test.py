'''
Some parts of the code are modified from:
CAS : https://github.com/bymavis/CAS_ICLR2021
CIFS : https://github.com/HanshuYAN/CIFS
'''


import os, argparse
os.environ["CUDA_VISIBLE_DEVICES"] = '6'
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import torchvision
import torchvision.transforms as transforms

from models.BaseModel import BaseModelDNN
from datasets import TestTinyImageNetDataset

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'


parser = argparse.ArgumentParser(description='Configuration')
parser.add_argument('--load_name', type=str,default='robust_1_sgd_False_100', help='specify checkpoint load name')
parser.add_argument('--model', default='resnet18')
parser.add_argument('--dataset', default='tiny_imagenet')
parser.add_argument('--tau', default=0.1, type=float)
parser.add_argument('--bs', default=100, type=int, help='batch size')
parser.add_argument('--device', default=0, type=int)

args = parser.parse_args()

device = 'cuda:{}'.format(args.device) if torch.cuda.is_available() else 'cpu'


if args.model == 'resnet18':
    from models.resnet_trobust import ResNet18_trobust
    net = ResNet18_trobust
elif args.model == 'vgg16':
    from models.vgg_trobust import vgg16_trobust
    net = vgg16_trobust

elif args.model == 'wideresnet34':
    from models.wideresnet34_trobust import WideResNet34_trobust
    net = WideResNet34_trobust

if args.dataset == 'cifar10':
    image_size = (32, 32)
    num_classes = 10
    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.bs, shuffle=False)

elif args.dataset == 'cifar100':
    num_classes = 100
    image_size = (32, 32)
    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])

    testset = torchvision.datasets.CIFAR100(
        root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=args.bs, shuffle=False,num_workers=4)

elif args.dataset == 'svhn':
    image_size = (32, 32)
    num_classes = 10
    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])
    testset = torchvision.datasets.SVHN(root='./data', split='test', download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.bs, shuffle=False)
elif args.dataset == 'tiny_imagenet':
    num_classes = 200
    image_size = (224, 224)
    transform_test = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
    ])
    testset = TestTinyImageNetDataset(
        root='/data/hdd3/tangbowen/Datasets/', target_dir='/tiny-imagenet-200/val',transform=transform_test)
    # testset = torchvision.datasets.ImageFolder(
    #     root=os.path.join('/data/hdd4/tangbowen/Datasets/tiny-imagenet-200','val'), transform=transform_test)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=args.bs, shuffle=False,num_workers=4)
    
elif args.dataset=='imagenette':
    num_classes = 10
    image_size = (224,224)
    args.bs=50
    test_transform = transforms.Compose([
        transforms.Resize([224,224]),
        transforms.ToTensor(),
        # transforms.Normalize(mean, std),
    ])
    testset = torchvision.datasets.ImageFolder('/data/hdd4/tangbowen/Datasets/imagenette/val/',test_transform) 
    testloader = torch.utils.data.DataLoader(
        dataset=testset,
        batch_size=args.bs,
        shuffle=False,
        pin_memory=True,
        num_workers=4,
    )

def get_pred(out, labels):
    pred = out.sort(dim=-1, descending=True)[1][:, 0]
    second_pred = out.sort(dim=-1, descending=True)[1][:, 1]
    adv_label = torch.where(pred == labels, second_pred, pred)

    return adv_label


class CE_loss(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, logits_final, target):
        loss = F.cross_entropy(logits_final, target)

        return loss


class CW_loss(nn.Module):
    def __init__(self, num_classes=10) -> None:
        super().__init__()
        self.num_classes = num_classes

    def forward(self, logits_final, target):
        loss = self._cw_loss(logits_final, target, num_classes=self.num_classes)

        return loss

    def _cw_loss(self, output, target, confidence=50, num_classes=10):
        target = target.data
        target_onehot = torch.zeros(target.size() + (num_classes,))
        target_onehot = target_onehot.to(device)
        target_onehot.scatter_(1, target.unsqueeze(1), 1.)
        target_var = Variable(target_onehot, requires_grad=False)
        real = (target_var * output).sum(1)
        other = ((1. - target_var) * output - target_var * 10000.).max(1)[0]
        loss = -torch.clamp(real - other + confidence, min=0.)  # equiv to max(..., 0.)
        loss = torch.sum(loss)
        return loss


class Classifier(BaseModelDNN):
    def __init__(self) -> None:
        super(BaseModelDNN).__init__()
        mean = torch.tensor([0.48024578664982126, 0.44807218089384643, 0.3975477478649648]).cuda()
        std = torch.tensor([0.2769864069088257, 0.26906448510256, 0.282081906210584]).cuda()
        self.net = net(tau=args.tau, num_classes=num_classes, image_size=image_size,args=args).to(device)
        # self.net = net(depth=18, dataset=args.dataset, mean=mean, std=std,args=args).to(device)

        self.set_requires_grad([self.net], False)

    def predict(self, x,targets,epoch=100, is_eval=True):
        return self.net(x ,targets,epoch,is_eval=is_eval)


def main():
    if args.model == 'resnet18':
        args.dims = [[64],[128],[256],[512]]
    elif args.model == 'vgg16':
        args.dims = [[64],[128],[256],[512]]
    elif args.model == 'wideresnet34':
        args.dims = [[160],[320],[640]]
    elif args.model == 'ViT':
        args.dims = [[192],[192],[192]]
    elif args.model == 'CVT':
        args.dims = [[64], [192], [384]]

    if args.dataset == 'cifar10':
        args.dims.append([10])
    elif args.dataset == 'svhn':
        args.dims.append([10])
    elif args.dataset == 'tiny_imagenet':
        args.dims.append([200])
    elif args.dataset == 'cifar100':
        args.dims.append([100])
    elif args.dataset=='imagenette':
        args.dims.append([10])
    args.epoch = 100
    model = Classifier()
    checkpoint = torch.load('weights/tiny_imagenet/resnet18/robust_1_sgd_False_400.pth', map_location=device)
    model.net.load_state_dict(checkpoint)
    model.net.eval()

    from advertorch_fsr.attacks import FGSM, LinfPGDAttack

    lst_attack = [
        (FGSM, dict(
            loss_fn=CE_loss(),
            eps=8 / 255,
            clip_min=0.0, clip_max=1.0, targeted=False), 'FGSM'),
        (LinfPGDAttack, dict(
            loss_fn=CE_loss(),
            eps=8 / 255, nb_iter=20, eps_iter=0.1 * (8 / 255), rand_init=False,
            clip_min=0.0, clip_max=1.0, targeted=False), 'PGD-20'),
        (LinfPGDAttack, dict(
            loss_fn=CE_loss(),
            eps=8 / 255, nb_iter=100, eps_iter=0.1 * (8 / 255), rand_init=False,
            clip_min=0.0, clip_max=1.0, targeted=False), 'PGD-100'),
        (LinfPGDAttack, dict(
            loss_fn=CW_loss(num_classes=num_classes),
            eps=8 / 255, nb_iter=30, eps_iter=0.1 * (8 / 255), rand_init=False,
            clip_min=0.0, clip_max=1.0, targeted=False), 'C&W'),
    ]
    attack_results = []
    for attack_class, attack_kwargs, name in lst_attack:
        from metric.classification import defense_success_rate

        message, defense_success, natural_success = defense_success_rate(model.predict,
                                                                         testloader, attack_class,
                                                                         attack_kwargs, device=device)

        message = name + ': ' + message
        print(args.model+' '+args.dataset+' '+args.load_name+' '+message)
        attack_results.append(defense_success)
    attack_results.append(natural_success)
    attack_results = torch.cat(attack_results, 1)
    attack_results = attack_results.sum(1)
    attack_results[attack_results < len(lst_attack) + 1] = 0.
    if args.dataset == 'cifar10' or args.dataset == 'cifar100' or args.dataset == 'tiny_imagenet':
        print('Ensemble : {:.2f}%'.format(100. * attack_results.count_nonzero() / 10000.))
    elif args.dataset == 'svhn':
        print('Ensemble : {:.2f}%'.format(100. * attack_results.count_nonzero() / 26032.))
    elif args.dataset == 'imagenette':
        print('Ensemble : {:.2f}%'.format(100. * attack_results.count_nonzero() / 500.))


if __name__ == '__main__':
    main()
