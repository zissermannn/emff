import os
os.environ["CUDA_VISIBLE_DEVICES"] = '5'
import torch

import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.functional as F

import torchvision
import torchvision.transforms as transforms
from models.resnet_trobust import ResNet18_trobust
from models.vgg_trobust import vgg16_trobust
from models.wideresnet34_trobust import WideResNet34_trobust
from attacks.pgd import PGD
from tqdm.auto import tqdm
import numpy as np
import argparse
from os.path import join


def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'

best_acc=0.0
parser = argparse.ArgumentParser(description='FSR Training')
parser.add_argument('--save_name', type=str, default='robust_1',help='specify checkpoint save name')
parser.add_argument('--lam_sep', type=float, default=1.0, help='weight for separation loss')
parser.add_argument('--lam_rec', type=float, default=1.0, help='weight for recalibration loss')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate for classifier')
parser.add_argument('--bs', default=128, type=int, help='batch size')
parser.add_argument('--epoch', default=400, type=int, help='number of epochs')
parser.add_argument('--dataset', type=str, default='tiny_imagenet', help='target dataset')
parser.add_argument('--model', type=str, default='resnet18', help='model name')
parser.add_argument('--eps', type=float, default=8., help='perturbation constraint epsilon')
parser.add_argument('--alpha', type=float, default=0.25, help='step size alpha')
parser.add_argument('--tau', type=float, default=0.1, help='tau for Gumbel softmax')
parser.add_argument('--device', type=int,default=0, help='device id')
parser.add_argument('--load_name', type=str,default='robust_reserved', help='specify checkpoint load name')

args = parser.parse_args()

print(args)
# args.image_size=32
if args.model == 'resnet18':
    args.dims = [[64],[128],[256],[512]]
elif args.model == 'vgg16':
    args.dims = [[64],[128],[256],[512]]
elif args.model == 'wideresnet34':
    args.dims = [[160],[320],[640]]

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


device = 'cuda:{}'.format(args.device) if torch.cuda.is_available() else 'cpu'
start_epoch = 1
args.optim = 'sgd'
args.use_lr_schedule = False
if args.dataset == 'cifar10':
    num_classes = 10
    image_size = (32, 32)
    transform_train = transforms.Compose([
        # transforms.ToTensor(),
        # transforms.Lambda(lambda x: F.pad(x.unsqueeze(0),
        #                                   (4, 4, 4, 4), mode='constant', value=0).squeeze()),
        # transforms.ToPILImage(),
        # transforms.RandomCrop(32),
        # transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(), 
        transforms.ToTensor(),
    ])
    transform_test = transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor(),
    ])

    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=args.bs, shuffle=True,num_workers=8)

    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=args.bs, shuffle=False,num_workers=4)
    
elif args.dataset == 'cifar100':
    num_classes = 100
    image_size = (32, 32)
    transform_train = transforms.Compose([
        # transforms.ToTensor(),
        # transforms.Lambda(lambda x: F.pad(x.unsqueeze(0),
        #                                   (4, 4, 4, 4), mode='constant', value=0).squeeze()),
        # transforms.ToPILImage(),
        # transforms.RandomCrop(32),
        # transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(), 
        transforms.ToTensor(),
    ])
    transform_test = transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor(),
    ])

    trainset = torchvision.datasets.CIFAR100(
        root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=args.bs, shuffle=True,num_workers=8)

    testset = torchvision.datasets.CIFAR100(
        root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=args.bs, shuffle=False,num_workers=4)

elif args.dataset == 'svhn':
    num_classes = 10
    image_size = (32, 32)
    args.alpha=0.125
    args.lr = 0.01
    transform_train = transforms.Compose([
        transforms.ToTensor(),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])

    trainset = torchvision.datasets.SVHN(
        root='./data', split='train', download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=args.bs, shuffle=True,num_workers=8)

    testset = torchvision.datasets.SVHN(
        root='./data', split='test', download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=args.bs, shuffle=False,num_workers=4)
    
elif args.dataset == 'tiny_imagenet':
    num_classes = 200
    image_size = (224,224)
    transform_train = transforms.Compose([
        transforms.Resize(224),
        transforms.RandomCrop(224, padding=4),
        transforms.RandomHorizontalFlip(), 
        transforms.ToTensor(),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])

    trainset = torchvision.datasets.ImageFolder(
        root=os.path.join('/data/hdd4/tangbowen/Datasets/tiny-imagenet-200','train'), transform=transform_train)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=args.bs, shuffle=True,num_workers=16)
    
    testset = torchvision.datasets.ImageFolder(
        root=os.path.join('/data/hdd4/tangbowen/Datasets/tiny-imagenet-200','val'), transform=transform_test)
    # testset = TestTinyImageNetDataset(
    #     root='/data/hdd3/tangbowen/Datasets/', target_dir='/tiny-imagenet-200/val',transform=transform_test)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=args.bs, shuffle=False,num_workers=8)

elif args.dataset=='imagenette':
    num_classes = 10
    image_size = (224,224)
    args.bs=64
    train_list = [
        transforms.Resize(224),
        transforms.RandomCrop(224, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ]
    train_transform = transforms.Compose(train_list)
    test_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        # transforms.Normalize(mean, std),
    ])
    trainset = torchvision.datasets.ImageFolder('/data/hdd4/tangbowen/Datasets/imagenette/train/',train_transform)
    testset = torchvision.datasets.ImageFolder('/data/hdd4/tangbowen/Datasets/imagenette/val/',test_transform) 
    trainloader = torch.utils.data.DataLoader(
        dataset=trainset,
        batch_size=args.bs,
        shuffle=True,
        pin_memory=True,
        num_workers=8,
    )
    testloader = torch.utils.data.DataLoader(
        dataset=testset,
        batch_size=args.bs,
        shuffle=False,
        pin_memory=True,
        num_workers=4,
    )

models = {
    'resnet18': ResNet18_trobust(tau=args.tau, num_classes=num_classes, image_size=image_size,args=args),
    'vgg16': vgg16_trobust(tau=args.tau, num_classes=num_classes, image_size=image_size,args=args),
    'wideresnet34': WideResNet34_trobust(tau=args.tau, num_classes=num_classes, image_size=image_size,args=args),
}

model_name = args.model
net = models[model_name]
net = net.to(device)
cudnn.benchmark = True


criterion = nn.CrossEntropyLoss(reduction='mean')
if args.optim == 'adam':
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr,weight_decay=5e-4)
elif args.optim == 'sgd':
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)

# if args.use_lr_schedule:
#     lr_scheduler=optim.lr_scheduler.StepLR(optimizer,step_size=75,gamma=0.5)
    # lr_scheduler=optim.lr_scheduler.StepLR(optimizer,step_size=75,gamma=0.5)

def get_pred(out, labels):
    pred = out.sort(dim=-1, descending=True)[1][:, 0]
    second_pred = out.sort(dim=-1, descending=True)[1][:, 1]
    adv_label = torch.where(pred == labels, second_pred, pred)

    return adv_label


attack = PGD(net, args.eps/255.0, args.alpha * (args.eps/255.0), min_val=0, max_val=1, max_iters=10, _type='linf')


def adjust_learning_rate(optimizer, epoch):
    """decrease the learning rate"""
    lr = args.lr
    if epoch >= 75:
        lr = args.lr * 0.1
    if epoch >= 90:
        lr = args.lr * 0.01
    if epoch >= 175:
        lr = args.lr * 0.01
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    adv_cls_losses = 0
    sep_losses = 0
    rec_losses = 0
    adv_correct = 0
    total = 0

    adjust_learning_rate(optimizer, epoch)
    image_num = 1
    with tqdm(total=(len(trainset) - len(trainset) % args.bs)) as _tqdm:
        _tqdm.set_description('{} {} {} (Train) Epoch: {}/{}'.format(args.dataset, args.model, args.save_name, epoch, args.epoch))
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.to(device), targets.to(device)

            net.eval()
            adv_inputs = attack.perturb(inputs, targets, epoch, True)
            net.train()

            adv_outputs, adv_r_outputs, adv_nr_outputs, adv_rec_outputs,evidences, tmc_loss= net(adv_inputs,targets,epoch)
            adv_labels = get_pred(adv_outputs, targets)
            adv_cls_loss = criterion(adv_outputs, targets)

            loss = tmc_loss+adv_cls_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            adv_cls_losses += adv_cls_loss.item()
            # sep_losses += sep_loss.item()
            # rec_losses += rec_loss.item()
            _, adv_predicted = evidences.max(1)
            total += targets.size(0)
            adv_correct += adv_predicted.eq(targets).sum().item()

            _tqdm.set_postfix(
                Adv_Loss='{:.3f}'.format(adv_cls_losses / (batch_idx + 1)),
                # Sep_Loss='{:.3f}'.format(sep_losses / (batch_idx + 1)),
                # Rec_Loss='{:.3f}'.format(rec_losses / (batch_idx + 1)),
                Loss='{:.3f}'.format(loss / (batch_idx + 1)),
                Adv_Acc='{:.3f}%'.format(100. * adv_correct / total),
            )
            _tqdm.update(inputs.shape[0])

    # if not os.path.exists('./weights/{}/{}/'.format(args.dataset, args.model)):
    #     os.makedirs('./weights/{}/{}/'.format(args.dataset, args.model))
    # torch.save(net.state_dict(), './weights/{}/{}/{}.pth'.format(args.dataset, args.model, args.save_name+'_'+args.optim+'_'+str(args.use_lr_schedule)+'_'+str(args.epoch)))



def test(epoch):
    global best_acc
    net.eval()
    ori_test_loss = 0
    adv_test_loss = 0
    ori_correct = 0
    adv_correct = 0
    total = 0
    image_num=1
    # checkpoint = torch.load('./weights/{}/{}/{}.pth'.format(args.dataset, args.model, args.load_name, map_location=device))
    # net.load_state_dict(checkpoint)

    with tqdm(total=(len(testset) - len(testset) % args.bs), dynamic_ncols=True) as _tqdm:
        _tqdm.set_description('{} (Test) Epoch: {}/{}'.format(args.save_name, epoch, args.epoch))
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            adv_inputs = attack.perturb(inputs, targets, epoch, True)
            net.eval()

            ori_outputs, ori_r_outputs, ori_nr_outputs, ori_rec_outputs,evidence_a,tmc_loss = net(inputs,targets,epoch, is_eval=True)
            adv_outputs, adv_r_outputs, adv_nr_outputs, adv_rec_outputs,evidence_b, tmc_loss= net(adv_inputs,targets,epoch, is_eval=True)
            visual_im = torch.mean(net.out_f,dim=1).detach().cpu()
            # for idx in range(len(visual_im)):
            #     np.save(join('./visulization/robustness/',str(image_num).rjust(6,'0')+'.npy'), visual_im[idx].numpy(), allow_pickle=True, fix_imports=True)
            #     image_num+=1
            # ori_loss = criterion(ori_outputs, targets)
            # ori_test_loss += ori_loss.item()
            # _, ori_predicted = ori_outputs.max(1)
            _, predicted = evidence_a.max(1)
            ori_correct += predicted.eq(targets).sum().item()
            # pred+=predicted
            adv_loss = criterion(adv_outputs, targets)
            adv_test_loss += adv_loss.item()
            _, adv_predicted = evidence_b.max(1)
            # adv_pred+=adv_predicted
            adv_correct += adv_predicted.eq(targets).sum().item()
        
            total += targets.size(0)

            _tqdm.set_postfix(
                Ori_Loss='{:.3f}'.format(ori_test_loss/(batch_idx+1)),
                Ori_Acc='{:.3f}%'.format(100.*ori_correct/total),
                Adv_Loss='{:.3f}'.format(adv_test_loss/(batch_idx+1)),
                Adv_Acc='{:.3f}%'.format(100.*adv_correct/total),
            )
            _tqdm.update(inputs.shape[0])
    acc = (ori_correct + adv_correct)*100/(2*total)
    if acc > best_acc:
        if not os.path.exists('./weights/{}/{}/'.format(args.dataset, args.model)):
                os.makedirs('./weights/{}/{}/'.format(args.dataset, args.model))
        torch.save(net.state_dict(), './weights/{}/{}/{}.pth'.format(args.dataset, args.model, args.save_name+'_'+args.optim+'_'+str(args.use_lr_schedule)+'_'+str(args.epoch)))

        # torch.save(state, path)
        print('Saving~ ./weights/{}/{}/{}.pth'.format(args.dataset, args.model, args.save_name+'_'+args.optim+'_'+str(args.use_lr_schedule)+'_'+str(args.epoch)))

        # update best acc
        best_acc = acc
    # if not os.path.exists('./weights/{}/{}/'.format(args.dataset, args.model)):
    #     os.makedirs('./weights/{}/{}/'.format(args.dataset, args.model))
    # torch.save(net.state_dict(), './weights/{}/{}/{}.pth'.format(args.dataset, args.model, args.save_name+'_'+args.optim+'_'+str(args.use_lr_schedule)+'_'+str(args.epoch)))


for epoch in range(start_epoch, args.epoch + 1):
    if args.use_lr_schedule:
        if epoch==75 or epoch ==150 or epoch ==200:
            for param_group in optimizer.param_groups:
                param_group['lr'] = param_group['lr']*0.01
    train(epoch)
    # if epoch%10==0:
    test(epoch)
    
