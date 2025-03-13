import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils import data
from .models import Perceptron, LeNet, VGG

def create_loaders(args, dct):
    # Create training set and validation set
    dct['train_size'] = dct['tvsize'] - args.dataset.valid_size
    dct['valid_size'] = args.dataset.valid_size
    dct['trainset'], dct['validset'] = tuple(data.random_split(dct['tvset'], [dct['train_size'], dct['valid_size']]))

    # Create loaders
    dct['train_loader'] = data.DataLoader(dct['trainset'], args.dataset.batch_size, shuffle = True)
    dct['valid_loader'] = data.DataLoader(dct['validset'], args.dataset.batch_size)
    dct['test_loader'] = data.DataLoader(dct['testset'], args.dataset.batch_size)

    return dct

def build_MNIST(args, dct):
    data_augm = args.dataset.data_augm

    transform_train = [transforms.ToTensor()]
    transform_test = [transforms.ToTensor()]
    if not args.dataset.autoencoder:
        transform_train.append(transforms.Normalize((0.1307,), (0.3081,)))
        transform_test.append(transforms.Normalize((0.1307,), (0.3081,)))
    if args.model.name not in ('LeNet', 'VGG'):
        transform_train.append(transforms.Lambda(lambda x: x.view(-1)))
        transform_test.append(transforms.Lambda(lambda x: x.view(-1)))
    if data_augm:
        transform_train = [transforms.RandomCrop(28, padding = 3)] + transform_train
    transform_train = transforms.Compose(transform_train)
    transform_test = transforms.Compose(transform_test)
    
    dct['tvset'] = torchvision.datasets.MNIST(root = args.dataset.path, train = True,
            download = False, transform = transform_train)

    dct['testset'] = torchvision.datasets.MNIST(root = args.dataset.path, train = False,
            download = False, transform = transform_test)

    dct['test_size'] = 10000
    dct['tvsize'] = 60000

    dct['n_classes'] = 10
    dct['n_channels'] = 1
    dct['image_size'] = 28
    dct['channel_size'] = 28**2
    dct['input_size'] = 28**2

    if args.dataset.autoencoder:
        dct['classification'] = False
        dct['loss_fn'] = nn.BCELoss()
    else:
        dct['classification'] = True
        dct['loss_fn'] = nn.NLLLoss()
        dct['topk_acc'] = (1,)

    return create_loaders(args, dct)

def build_CIFAR10(args, dct):
    data_augm = args.dataset.data_augm

    transform_train = [transforms.ToTensor()]
    transform_test = [transforms.ToTensor()]
    if not args.dataset.autoencoder:
        transform_train.append(transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)))
        transform_test.append(transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)))
    if args.model.name not in ('LeNet', 'VGG'):
        transform_train.append(transforms.Lambda(lambda x: x.view(-1)))
        transform_test.append(transforms.Lambda(lambda x: x.view(-1)))
    if data_augm:
        transform_train = [transforms.RandomCrop(32, padding = 4),
                           transforms.RandomHorizontalFlip()] \
                          + transform_train
    transform_train = transforms.Compose(transform_train)
    transform_test = transforms.Compose(transform_test)

    dct['tvset'] = torchvision.datasets.CIFAR10(root = args.dataset.path, train = True,
            download = False, transform = transform_train)

    dct['testset'] = torchvision.datasets.CIFAR10(root = args.dataset.path, train = False,
            download = False, transform = transform_test)

    dct['test_size'] = 10000
    dct['tvsize'] = 50000

    dct['n_classes'] = 10
    dct['n_channels'] = 3
    dct['image_size'] = 32
    dct['channel_size'] = 32**2
    dct['input_size'] = 3 * 32**2

    if args.dataset.autoencoder:
        dct['classification'] = False
        dct['loss_fn'] = nn.BCELoss()
    else:
        dct['classification'] = True
        dct['loss_fn'] = nn.NLLLoss()
        dct['topk_acc'] = (1,)

    return create_loaders(args, dct)

def build_ImageNet(args, dct):
    data_augm = args.dataset.data_augm

    transform_train = [transforms.ToTensor()]
    transform_test = [transforms.ToTensor()]
    if not args.dataset.autoencoder:
        transform_train.append(transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
        transform_test.append(transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
        transform_train = [transforms.Resize(256), transforms.CenterCrop(224)] + transform_train
        transform_test = [transforms.Resize(256), transforms.CenterCrop(224)] + transform_test
    if args.model.name not in ('LeNet', 'VGG', "ResNet"):
        transform_train.append(transforms.Lambda(lambda x: x.view(-1)))
        transform_test.append(transforms.Lambda(lambda x: x.view(-1)))
    if data_augm:
        raise NotImplementedError("Data augmentation not implemented.")
        transform_train = [transforms.RandomResizedCrop(224),
                           transforms.RandomHorizontalFlip()] \
                          + transform_train
    transform_train = transforms.Compose(transform_train)
    transform_test = transforms.Compose(transform_test)

    dct['tvset'] = torchvision.datasets.ImageNet(root = args.dataset.path, split = "train",
            transform = transform_train)

    dct['testset'] = torchvision.datasets.ImageNet(root = args.dataset.path, split = "val",
            transform = transform_test)

    dct['test_size'] = 50000
    dct['tvsize'] = 1281167

    dct['n_classes'] = 100
    dct['n_channels'] = 3
    dct['image_size'] = 250
    dct['channel_size'] = 250**2
    dct['input_size'] = 3 * 250**2

    if args.dataset.autoencoder:
        dct['classification'] = False
        dct['loss_fn'] = nn.BCELoss()
    else:
        dct['classification'] = True
        dct['loss_fn'] = nn.NLLLoss()
        dct['topk_acc'] = (1,)

    return create_loaders(args, dct)

def build_toy_regression(args, dct):
    if args.model.name != 'Perceptron':
        raise ValueError('Error: with dataset "ToyRegression", the model must be "Perceptron", got {}.'\
                .format(args.model.name))

    args_teacher = args.dataset.teacher
    model_args = args_teacher.args
    act_function = args_teacher.act_function
    sigma_w = args_teacher.sigma_w
    sigma_b = args_teacher.sigma_b
    if '*' in model_args:
        n_layers = int(model_args[:model_args.find('*')])
        n_neurons = int(model_args[model_args.find('*') + 1:])
        model_args = '-'.join([str(n_neurons) for i in range(n_layers)])
    layers = [int(s) for s in model_args.split('-')]
    in_size = layers[0]
    out_size = layers[-1]

    teacher = Perceptron(layers, act_function, scaling = False, sigma_w = sigma_w, sigma_b = sigma_b,
        classification = False)
    with torch.no_grad():
        tv_in = torch.randn(args.dataset.train_size + args.dataset.valid_size, in_size, 
                dtype = dct['dtype'], device = dct['device'])
        tv_out = torch.randn(args.dataset.train_size + args.dataset.valid_size, out_size, 
                dtype = dct['dtype'], device = dct['device'])

        test_in = torch.randn(args.dataset.test_size, in_size, 
                dtype = dct['dtype'], device = dct['device'])
        test_out = torch.randn(args.dataset.test_size, out_size, 
                dtype = dct['dtype'], device = dct['device'])

    dct['tvsize'] = args.dataset.train_size + args.dataset.valid_size
    dct['test_size'] = args.dataset.test_size

    dct['tvset'] = data.TensorDataset(tv_in, tv_out)

    dct['testset'] = data.TensorDataset(test_in, test_out)

    dct['classification'] = False
    dct['input_size'] = in_size
    dct['loss_fn'] = nn.MSELoss()

    return create_loaders(args, dct)

def build_None(args, dct):
    args_teacher = args.dataset.teacher
    model_args = args_teacher.args
    act_function = args_teacher.act_function
    sigma_w = args_teacher.sigma_w
    sigma_b = args_teacher.sigma_b
    if '*' in model_args:
        n_layers = int(model_args[:model_args.find('*')])
        n_neurons = int(model_args[model_args.find('*') + 1:])
        model_args = '-'.join([str(n_neurons) for i in range(n_layers)])
    layers = [int(s) for s in model_args.split('-')]
    in_size = layers[0]
    out_size = layers[-1]

    teacher = Perceptron(layers, act_function, scaling = False, sigma_w = sigma_w, sigma_b = sigma_b,
        classification = False)
    with torch.no_grad():
        tv_in = torch.zeros(2, 1, dtype = dct['dtype'], device = dct['device'])
        tv_out = torch.zeros(2, 1, dtype = dct['dtype'], device = dct['device'])

        test_in = torch.zeros(1, 1, dtype = dct['dtype'], device = dct['device'])
        test_out = torch.zeros(1, 1, dtype = dct['dtype'], device = dct['device'])

    dct['tvsize'] = 2
    dct['test_size'] = 1

    dct['tvset'] = data.TensorDataset(tv_in, tv_out)

    dct['testset'] = data.TensorDataset(test_in, test_out)

    dct['classification'] = False
    dct['input_size'] = 1
    dct['loss_fn'] = lambda x, y: x

    return create_loaders(args, dct)

