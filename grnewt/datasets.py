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
    transform = [transforms.ToTensor()]
    if not args.dataset.autoencoder:
        transform.append(transforms.Normalize((0.1307,), (0.3081,)))
    if not args.model.name in ('LeNet', 'VGG'):
        transform.append(transforms.Lambda(lambda x: x.view(-1)))
    transform = transforms.Compose(transform)

    dct['tvset'] = torchvision.datasets.MNIST(root = args.dataset.path, train = True,
            download = False, transform = transform)

    dct['testset'] = torchvision.datasets.MNIST(root = args.dataset.path, train = False,
            download = False, transform = transform)

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
    transform = [transforms.ToTensor()]
    if not args.dataset.autoencoder:
        transform.append(transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)))
    if not args.model.name in ('LeNet', 'VGG'):
        transform.append(transforms.Lambda(lambda x: x.view(-1)))
    transform = transforms.Compose(transform)

    dct['tvset'] = torchvision.datasets.CIFAR10(root = args.dataset.path, train = True,
            download = False, transform = transform)

    dct['testset'] = torchvision.datasets.CIFAR10(root = args.dataset.path, train = False,
            download = False, transform = transform)

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

