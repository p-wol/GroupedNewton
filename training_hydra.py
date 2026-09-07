import os
import random
import time

import numpy as np
import torch
import torch.optim as optim
import torchvision

# from torchvision.models import resnet18, resnet34, resnet50, resnet101, resnet152
from torch.utils import data

# from kfac.optimizers import KFACOptimizer
from grnewt import (
    NewtonStochasticHv,
    NewtonSummaryFB,
    NewtonSummaryUniformAvg,
    NewtonSummaryStaticAvg,
    NewtonSummaryMovexpAvg,
    ParamStructure,
    ReduceDampingOnPlateau,
    compute_Hg_fullbatch,
    diff_n_fullbatch,
    fullbatch_gradient,
    loader_pre_hooks,
    optimizers,
)
from grnewt import partition as build_partition
from grnewt.config import Partition, UpdaterName, from_dictconfig, migrate
from grnewt.datasets import (
    build_CIFAR10,
    build_ImageNet,
    build_MNIST,
    build_None,
    build_toy_regression,
)
from grnewt.models import VGG, AutoencoderMLP, LeNet, Perceptron, Rosenbrock, RosenbrockT
from grnewt.nesterov import nesterov_lrs


def set_seeds(seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)  # noqa: NPY002 -- seeds the legacy global RNG on purpose:
    # grnewt.datasets and third-party code still call np.random.* directly.
    random.seed(seed)


def assign_device(device):
    device = int(device)
    if device > -1:
        if torch.cuda.is_available():
            device = "cuda:" + str(device)
        else:
            device = "cpu"
    elif device == -1:
        device = "cuda"
    elif device == -2:
        device = "cpu"
    else:
        ValueError(f"Unknown device: {device}")

    return device


def get_dtype(dtype):
    if dtype == 64:
        return torch.double
    elif dtype == 32:
        return torch.float
    else:
        raise ValueError(f"Unknown dtype: {dtype}")


class Trainer:
    def __init__(self, config, hydra_path):
        self.args = config

        set_seeds(self.args.seed)

        self.device = assign_device(self.args.system.device)
        self.dtype = get_dtype(self.args.system.dtype)

        self.path_metrics = f"{hydra_path}/metrics"
        if not os.path.isdir(self.path_metrics):
            os.makedirs(self.path_metrics)
        open(f"{self.path_metrics}/metrics.json", "w").close()

        self.path_artifacts = f"{hydra_path}/artifacts"
        if not os.path.isdir(self.path_artifacts):
            os.makedirs(self.path_artifacts)

        print(self.args)

    def build_datasets(self):
        """
        Sets the following attributes:
            'train_size'
            'valid_size'
            'test_size'
            'tvsize': train_size + valid_size

            'trainset'
            'validset'
            'testset'
            'tvset': union of 'trainset' and 'validset'

            'train_loader'
            'valid_loader'
            'test_loader'

            'classification': True if classification task
            'n_classes'
            'n_channels'
            'image_size': height or width of an input image
            'channel_size': image_size**2
            'input_size': size of an input sample
            'topk_acc'
            'loss_fn'
        """

        args = self.args
        dct = {"dtype": self.dtype, "device": self.device}

        if args.dataset.name == "MNIST":
            self.loader_pre_hook = lambda x, y: (
                x.to(device=self.device, dtype=self.dtype),
                y.to(self.device),
            )
            dct = build_MNIST(args, dct)
        elif args.dataset.name == "CIFAR10":
            self.loader_pre_hook = lambda x, y: (
                x.to(device=self.device, dtype=self.dtype),
                y.to(self.device),
            )
            dct = build_CIFAR10(args, dct)
        elif args.dataset.name == "ImageNet":
            self.loader_pre_hook = lambda x, y: (
                x.to(device=self.device, dtype=self.dtype),
                y.to(self.device),
            )
            dct = build_ImageNet(args, dct)
        elif args.dataset.name == "ToyRegression":
            dct = build_toy_regression(args, dct)
        elif args.dataset.name == "None":
            dct = build_None(args, dct)
        else:
            raise NotImplementedError(f"Unknown dataset: {args.dataset.name}.")

        dct.pop("dtype")
        dct.pop("device")

        for k, v in dct.items():
            setattr(self, k, v)

    def build_model(self):
        args = self.args

        # Activation function
        dct_act_functions = {
            "identity": lambda x: x,
            "tanh": torch.tanh,
            "relu": torch.relu,
            "sigmoid": torch.sigmoid,
            "elu": torch.nn.functional.elu,
        }
        act_function = dct_act_functions[args.model.act_function]

        # Create model
        model_args = str(args.model.args)
        if "*" in model_args:
            n_layers = int(args.model.args[: args.model.args.find("*")])
            n_neurons = int(args.model.args[args.model.args.find("*") + 1 :])
            model_args = "-".join([str(n_neurons) for i in range(n_layers)]) + f"-{self.n_classes}"

        sigma_w = args.model.init.sigma_w
        sigma_b = args.model.init.sigma_b
        if args.model.name == "Perceptron":
            layers = [int(s) for s in model_args.split("-")]
            layers = [self.input_size] + layers
            classification = True
            if args.dataset.name == "ToyRegression":
                classification = False

            model = Perceptron(
                layers,
                act_function,
                scaling=args.model.scaling,
                sigma_w=sigma_w,
                sigma_b=sigma_b,
                classification=classification,
            )
        elif args.model.name == "LeNet":
            layers = [int(s) for s in model_args.split("-")]
            layers = [self.n_channels] + layers

            model = LeNet(
                layers, act_function, scaling=args.model.scaling, sigma_w=sigma_w, sigma_b=sigma_b
            )
        elif args.model.name == "VGG":
            vgg_setup = [s for s in model_args.split("-")]
            vgg_type = vgg_setup[0][0]
            with_batch_norm = True if "bn" in vgg_setup[0] else False
            fc_sizes = [int(s) for s in vgg_setup[1:]]

            model = VGG(
                vgg_type,
                fc_sizes=fc_sizes,
                image_size=self.image_size,
                num_classes=self.n_classes,
                scaling=args.model.scaling,
                sigma_w=sigma_w,
                name_act_function=args.model.act_function,
                batch_norm=with_batch_norm,
            )
        elif args.model.name == "ResNet":
            model_name = "resnet" + model_args
            model = torchvision.models.__dict__[model_name]()
        elif args.model.name == "AutoencoderMLP":
            layers = [int(s) for s in model_args.split("-")]
            layers = [self.input_size] + layers
            classification = False

            model = AutoencoderMLP(
                layers, act_function, scaling=args.model.scaling, sigma_w=sigma_w, sigma_b=sigma_b
            )
        elif args.model.name == "Rosenbrock":
            lst_params = model_args.split("-")
            d = int(lst_params[0])
            a = float(lst_params[1])
            b = float(lst_params[2])

            model = Rosenbrock(d, a, b)
        elif args.model.name == "RosenbrockT":
            lst_params = model_args.split("-")
            d = int(lst_params[0])
            a = float(lst_params[1])
            b = float(lst_params[2])

            model = RosenbrockT(d, a, b)
        else:
            raise NotImplementedError(f"Unknown model: {args.model.name}.")

        return model.to(device=self.device, dtype=self.dtype)

    def build_optimizer(self, model):
        args = self.args
        # Single validation boundary. After this line nothing reads a DictConfig:
        # `hg` is a plain, typed, validated HgCfg. Any field the selected optimizer
        # does not read raises here rather than being silently dropped.
        hg = from_dictconfig(migrate(self.args.optimizer.hg), optimizer_name=args.optimizer.name)
        self.hg = hg

        # Define useful variables
        def full_loss(x, y):
            return self.loss_fn(self.model(x), y)

        # Build data loader for Hg
        hg_batch_size = args.dataset.batch_size if hg.batch_size == -1 else hg.batch_size
        self.hg_loader = data.DataLoader(self.trainset, hg_batch_size, shuffle=True, drop_last=True)

        # Build partition
        if hg.partition is Partition.canonical:
            param_groups, name_groups = build_partition.canonical(model)
        elif hg.partition is Partition.wb:
            param_groups, name_groups = build_partition.wb(model)
        elif hg.partition is Partition.trivial:
            param_groups, name_groups = build_partition.trivial(model)
        elif hg.partition is Partition.blocks:
            param_groups, name_groups = build_partition.blocks(model, hg.partition_arg)
        elif hg.partition is Partition.alternate:
            alternate = hg.partition_arg
            if args.model.name == "Perceptron":
                nlayers = len(model.layers)
            elif args.model.name == "VGG":
                nlayers = len(model.features)
            else:
                raise NotImplementedError(
                    f"partition=alternate is not defined for model {args.model.name}."
                )
            lst_names_w = [
                [f"{i}.weight" for i in range(nlayers) if i % alternate == r]
                for r in range(alternate)
            ]
            lst_names_b = [
                [f"{i}.bias" for i in range(nlayers) if i % alternate == r]
                for r in range(alternate)
            ]
            param_groups, name_groups = build_partition.names_by_lst(
                model, lst_names_w + lst_names_b
            )
        elif hg.partition in (Partition.vgg, Partition.perceptron):
            param_groups, name_groups = model.partition(hg.partition_str)
        else:
            raise NotImplementedError(f"Unknown partition: {hg.partition}.")

        # param_groups = ParamStructure(pgroups)

        print(name_groups)

        # Build optimizer
        if args.optimizer.name == "SGD":
            optimizer = optim.SGD(
                param_groups,
                lr=args.optimizer.lr,
                momentum=args.optimizer.momentum,
                weight_decay=args.optimizer.weight_decay,
            )
        elif args.optimizer.name == "Adam":
            optimizer = optim.Adam(param_groups, lr=args.optimizer.lr)
        elif args.optimizer.name.find("NewtonSummary") == 0:
            if hg.updater.name is UpdaterName.SGD:
                updater = optimizers.SGDUpdate(
                    model.parameters(),
                    lr=1,
                    momentum=hg.updater.momentum,
                    dampening=hg.updater.momentum_damp,
                )
            else:
                updater = optimizers.AdamUpdate(model.parameters(), lr=1)

            if args.optimizer.name == "NewtonSummaryFB":
                optimizer = NewtonSummaryFB(
                    param_groups,
                    full_loss,
                    self.model,
                    self.loss_fn,
                    self.hg_loader,
                    self.train_size,
                    loader_pre_hook=self.loader_pre_hook,
                    cfg=hg,
                )
            elif args.optimizer.name == "NewtonSummaryUniformAvg":
                optimizer = NewtonSummaryUniformAvg(
                    param_groups,
                    full_loss,
                    self.hg_loader,
                    updater,
                    loader_pre_hook=self.loader_pre_hook,
                    cfg=hg,
                )
            elif args.optimizer.name == "NewtonSummaryStaticAvg":
                optimizer = NewtonSummaryStaticAvg(
                    param_groups,
                    full_loss,
                    self.hg_loader,
                    updater,
                    loader_pre_hook=self.loader_pre_hook,
                    cfg=hg,
                )
            elif args.optimizer.name == "NewtonSummaryMovexpAvg":
                optimizer = NewtonSummaryStaticAvg(
                    param_groups,
                    full_loss,
                    self.hg_loader,
                    updater,
                    loader_pre_hook=self.loader_pre_hook,
                    cfg=hg,
                )
            else:
                raise NotImplementedError(f"Unknown NewtonSummary variant: {args.optimizer.name}.")
        elif args.optimizer.name == "NewtonStochasticHv":
            args_nsto = args.optimizer.newtonsto
            optimizer = NewtonStochasticHv(
                param_groups,
                self.model,
                self.loss_fn,
                self.hg_loader,
                loader_pre_hook=self.loader_pre_hook,
                lr_param=args_nsto.lr_param,
                lr_direction=args_nsto.lr_direction,
                dct_nesterov=None,
                ridge=args_nsto.ridge,
            )
        elif args.optimizer.name == "KFAC":
            # FIX (2026-08-21): `from kfac.optimizers import KFACOptimizer` is
            # commented out at the top of this file, so this branch was a
            # NameError, not a NotImplementedError.
            raise NotImplementedError(
                "optimizer.name=KFAC requires the `kfac` package and the import "
                "at the top of training_hydra.py, both currently absent."
            )
        elif args.optimizer.name == "LBFGS":
            if args.optimizer.lbfgs.line_search_fn == "none":
                line_search_fn = None
            else:
                line_search_fn = args.optimizer.lbfgs.line_search_fn

            optimizer = optim.LBFGS(
                param_groups,
                lr=args.optimizer.lr,
                max_iter=args.optimizer.lbfgs.max_iter,
                history_size=args.optimizer.lbfgs.history_size,
                line_search_fn=line_search_fn,
            )
        else:
            raise NotImplementedError(f"Unknown optimizer: {args.optimizer.name}.")

        # Store grouping data
        self.name_groups = name_groups
        self.param_groups = param_groups
        self.param_struct = ParamStructure(self.param_groups)
        self.tup_params = self.param_struct.tup_params

        self.full_loss = full_loss

        return optimizer

    def test_model(self, loader, dsname):
        with torch.no_grad():
            cum_nll = torch.zeros((), dtype=self.dtype, device=self.device)
            cum_pen = torch.zeros((), dtype=self.dtype, device=self.device)
            cum_loss = torch.zeros((), dtype=self.dtype, device=self.device)
            total = 0
            if self.classification:
                correct = [
                    torch.tensor([0], dtype=self.dtype, device=self.device)
                    for i in range(len(self.topk_acc))
                ]
            for i, (images, labels) in enumerate(loader):  # noqa: B007
                # Convert torch tensor to Variable
                images, labels = self.loader_pre_hook(images, labels)

                # Forward
                outputs = self.model(images)
                nll = self.loss_fn(outputs, labels)
                pen = torch.zeros((), dtype=self.dtype, device=self.device)
                loss = nll + pen

                cum_nll += nll.detach() * labels.size(0)
                cum_pen += pen.detach() * labels.size(0)
                cum_loss += loss.detach() * labels.size(0)

                total += labels.size(0)

                if self.classification:
                    maxk = max(self.topk_acc)
                    _, pred = outputs.topk(maxk, 1, True, True)
                    pred = pred.t()
                    tmp_correct = pred.eq(labels.view(1, -1).expand_as(pred))

                    for idk, k in enumerate(self.topk_acc):
                        correct[idk] += tmp_correct[:k].reshape(-1).float().sum(0, keepdim=True)

            # Compute performance
            mean_pen = cum_pen.item() / total
            mean_nll = cum_nll.item() / total
            mean_loss = cum_loss.item() / total

            metrics = {"nll": mean_nll, "pen": mean_pen, "loss": mean_loss}

            if self.classification:
                if len(self.topk_acc) == 1:
                    mean_acc = correct[0].item() / total
                else:
                    mean_acc = [corr.item() / total for corr in correct]
                metrics["acc"] = mean_acc

            metrics = {(dsname + "_" + k): v for k, v in metrics.items()}

            return metrics

    def step_train(self):
        self.model.train()

        cum_nll = torch.zeros((), dtype=self.dtype, device=self.device)
        cum_pen = torch.zeros((), dtype=self.dtype, device=self.device)
        cum_loss = torch.zeros((), dtype=self.dtype, device=self.device)
        total = 0
        if self.classification:
            correct = [
                torch.tensor([0], dtype=self.dtype, device=self.device)
                for i in range(len(self.topk_acc))
            ]
        self.idx_substep = 0
        self.logs_nlls = []
        # same as above: `i` is the batch count read after the loop.
        for i, (images, labels) in enumerate(self.train_loader):  # noqa: B007
            # Convert torch tensor to Variable
            images, labels = self.loader_pre_hook(images, labels)

            # Forward + Backward + Optimize
            self.optimizer.zero_grad()  # zero the gradient buffer
            outputs = self.model(images)
            nll = self.loss_fn(outputs, labels)
            pen = torch.zeros((), dtype=self.dtype, device=self.device)
            loss = nll + pen

            # TODO: detailed sequence of NLLs
            self.logs_nlls.append(nll.detach())

            cum_nll += nll.detach()
            cum_pen += pen.detach()
            cum_loss += loss.detach()

            total += labels.size(0)

            if self.classification:
                maxk = max(self.topk_acc)
                _, pred = outputs.topk(maxk, 1, True, True)
                pred = pred.t()
                tmp_correct = pred.eq(labels.view(1, -1).expand_as(pred))

                for idk, k in enumerate(self.topk_acc):
                    correct[idk] += tmp_correct[:k].reshape(-1).float().sum(0, keepdim=True)

            # KFAC specific
            if (
                self.args.optimizer.name == "KFAC"
                and self.optimizer.steps % self.optimizer.TCov == 0
            ):
                # compute true fisher
                self.optimizer.acc_stats = True
                with torch.no_grad():
                    sampled_y = torch.multinomial(
                        torch.nn.functional.softmax(outputs, dim=1), 1
                    ).squeeze()
                loss_sample = self.loss_fn(outputs, sampled_y)
                loss_sample.backward(retain_graph=True)
                self.optimizer.acc_stats = False
                self.optimizer.zero_grad()  # clear the gradient for computing true-fisher.

            loss.backward()
            self.optimizer.step()

        # Compute performance
        mean_pen = cum_pen.item() / (i + 1)
        mean_nll = cum_nll.item() / (i + 1)
        mean_loss = cum_loss.item() / (i + 1)

        self.model.eval()

        metrics = {"tr_nll": mean_nll, "tr_pen": mean_pen, "tr_loss": mean_loss}

        if self.classification:
            if len(self.topk_acc) == 1:
                mean_acc = correct[0].item() / total
            else:
                mean_acc = [corr.item() / total for corr in correct]
            metrics["tr_acc"] = mean_acc

        return metrics

    def pre_train(self):
        self.tup_params = tuple(p for n, p in self.model.named_parameters())
        self.tup_names = tuple(n for n, p in self.model.named_parameters())

        if self.args.dataset.autoencoder:
            f_loader_pre_hook = loader_pre_hooks.regression
        else:
            f_loader_pre_hook = loader_pre_hooks.classification

        self.loader_pre_hook = lambda x, y: f_loader_pre_hook(
            x, y, device=self.device, dtype=self.dtype, non_blocking=self.args.dsloader.non_blocking
        )

    def train(self, ckpt_name="last_ckpt", log_name="metrics"):
        self.build_datasets()
        self.train_loader_logs_hg = data.DataLoader(self.trainset, self.args.logs_hg.batch_size)
        self.valid_loader_logs_hg = data.DataLoader(self.validset, self.args.logs_hg.batch_size)
        self.test_loader_logs_hg = data.DataLoader(self.testset, self.args.logs_hg.batch_size)
        self.model = self.build_model()
        self.optimizer = self.build_optimizer(self.model)
        self.use_scheduler = self.hg.dmp_auto.use
        if self.use_scheduler and self.args.optimizer.name.find("NewtonSummary") == 0:
            args_sch = self.hg.dmp_auto
            self.scheduler = ReduceDampingOnPlateau(
                self.optimizer,
                factor=args_sch.factor,
                patience=args_sch.patience,
                cooldown=args_sch.cooldown,
                threshold=args_sch.threshold,
                apply_to=args_sch.apply_to,
                verbose=True,
            )
        self.pre_train()

        time_t0 = time.time()

        print(self.model)
        if self.args.logs_hg.use:
            print("tup_params: ")
            for p in self.tup_params:
                print("    ", p.size())

        # Store the param names - param_groups correspondence
        torch.save(self.name_groups, f"{self.path_artifacts}/ParamNameGroups.pkl")

        # Prepare damping schedule
        damp_sch = self.hg.damping_schedule
        if damp_sch.use:
            damp_sch_epoch = damp_sch.epoch
            damp_sch_factor = (damp_sch.final / self.hg.damping) ** (1 / (damp_sch.epoch + 1))

        # Full training procedure
        for epoch in range(self.args.optimizer.epochs):
            self.epoch = epoch
            print(f"Epoch {self.epoch}")

            # If args.logs_hg.use, then compute H, g and order3 with full-batch
            if self.args.logs_hg.use:
                logs = self.compute_logs_hg()
                torch.save(logs, f"{self.path_artifacts}/Hg_logs_ext.{self.epoch:05}.pkl")

            if self.args.logs_diff.use:
                logs = self.compute_logs_diff()
                torch.save(logs, f"{self.path_artifacts}/Hg_logs_diff.{self.epoch:05}.pkl")

            # Training step
            if self.args.optimizer.name == "NewtonSummaryFB":
                self.model.train()
                self.optimizer.step()
                self.model.eval()

                metrics_tr = self.test_model(self.train_loader, "tr")
            elif self.args.optimizer.name == "LBFGS":

                def closure():
                    self.model.zero_grad()
                    objective = self.model(0)
                    objective.backward()
                    return objective

                metrics_tr = {"tr_loss": self.optimizer.step(closure).item()}
            else:
                try:
                    metrics_tr = self.step_train()
                except:
                    if self.args.optimizer.name.find("NewtonSummary") == 0 and not self.hg.nologs:
                        optim_logs = self.optimizer.logs

                        logs_last = {
                            k: v[-1]
                            for k, v in optim_logs.items()
                            if len(v) > 0 and torch.is_tensor(v[0])
                        }
                        logs_mean = {
                            k: torch.stack(v).mean(0)
                            for k, v in optim_logs.items()
                            if len(v) > 0 and torch.is_tensor(v[0])
                        }
                        logs_total = {
                            k: v for k, v in optim_logs.items() if len(v) > 0 and k != "H"
                        }

                        if False:
                            logs_total["H"] = optim_logs["H"]

                        torch.save(
                            logs_last,
                            f"{self.path_artifacts}/Hg_logs_last.{self.epoch:05}.onexit.pkl",
                        )
                        torch.save(
                            logs_mean,
                            f"{self.path_artifacts}/Hg_logs_mean.{self.epoch:05}.onexit.pkl",
                        )
                        torch.save(
                            logs_total,
                            f"{self.path_artifacts}/Hg_logs_total.{self.epoch:05}.onexit.pkl",
                        )
                        torch.save(
                            self.logs_nlls,
                            f"{self.path_artifacts}/nlls_logs_total.{self.epoch:05}.onexit.pkl",
                        )
                    raise

            # Use scheduler
            if self.use_scheduler and self.args.optimizer.name.find("NewtonSummary") == 0:
                self.scheduler.step(metrics_tr["tr_loss"])

            metrics_va = self.test_model(self.valid_loader, "va")
            metrics_ts = self.test_model(self.test_loader, "ts")

            dct_time = {
                "epoch": self.epoch,
                "time": time.time() - time_t0,
                "memory_peak": torch.cuda.max_memory_allocated(self.device),
            }

            # Logs -- metrics
            metrics = dct_time | metrics_tr | metrics_va | metrics_ts
            print(metrics)

            # self.logger.log_checkpoint(self, log_name = ckpt_name)
            with open(f"{self.path_metrics}/metrics.json", "a") as f:
                f.write(metrics.__repr__() + "\n")

            # Logs -- artifacts
            if self.args.optimizer.name.find("NewtonSummary") == 0 and not self.hg.nologs:
                optim_logs = self.optimizer.logs

                logs_last = {
                    k: v[-1] for k, v in optim_logs.items() if len(v) > 0 and torch.is_tensor(v[0])
                }
                logs_mean = {
                    k: torch.stack(v).mean(0)
                    for k, v in optim_logs.items()
                    if len(v) > 0 and torch.is_tensor(v[0])
                }
                logs_total = {k: v for k, v in optim_logs.items() if len(v) > 0 and k != "H"}

                if False:
                    logs_total["H"] = optim_logs["H"]

                torch.save(logs_last, f"{self.path_artifacts}/Hg_logs_last.{self.epoch:05}.pkl")
                torch.save(logs_mean, f"{self.path_artifacts}/Hg_logs_mean.{self.epoch:05}.pkl")
                torch.save(logs_total, f"{self.path_artifacts}/Hg_logs_total.{self.epoch:05}.pkl")
                nlls = torch.stack(self.logs_nlls).cpu()
                torch.save(nlls, f"{self.path_artifacts}/nlls_logs_total.{self.epoch:05}.pkl")

                self.optimizer.reset_logs()
            elif self.args.optimizer.name == "NewtonSummaryFB":
                torch.save(
                    self.optimizer.logs, f"{self.path_artifacts}/Hg_logs_hgfb.{self.epoch:05}.pkl"
                )

            # Update damping schedule
            if damp_sch.use and self.epoch <= damp_sch_epoch:
                self.optimizer.damping_mul(damp_sch_factor)

        """
        metrics = add_prefix(prefix,metrics)
        print(metrics)
        self.logger.log_checkpoint(self, log_name= ckpt_name)
        self.logger.log_metrics(metrics, log_name=log_name)
        """

    def compute_logs_hg(self):
        logs = {}

        direction = fullbatch_gradient(
            self.param_struct,
            self.loss_fn,
            self.model,
            self.train_loader_logs_hg,
            self.train_size,
            loader_pre_hook=self.loader_pre_hook,
        )

        H, g, order3 = compute_Hg_fullbatch(
            self.param_struct,
            self.full_loss,
            self.train_loader_logs_hg,
            self.train_size,
            direction,
            loader_pre_hook=self.loader_pre_hook,
        )
        order3_ = order3.abs().pow(1 / 3)

        # Compute lrs
        if not self.hg.nesterov.use:
            regul_H = self.hg.ridge * torch.eye(H.size(0), dtype=self.dtype, device=self.device)
            lrs = torch.linalg.solve(H + regul_H, g)
        else:
            lrs, lrs_logs = nesterov_lrs(
                H,
                g,
                order3_,
                damping_int=self.hg.nesterov.damping_int,
                threshold_D_sing=self.hg.nesterov.threshold_D_sing,
                hard_case_rtol=self.hg.nesterov.hard_case_rtol,
                refine=self.hg.nesterov.refine,
            )
            for k, v in lrs_logs.items():
                logs["nesterov." + k] = v

        logs["H"] = H
        logs["g"] = g
        logs["order3"] = order3
        logs["lrs"] = lrs

        return logs

    def compute_logs_diff(self):
        logs = {}

        if self.args.logs_diff.partition == "canonical":
            param_groups, name_groups = build_partition.canonical(self.model)
        elif self.args.logs_diff.partition == "wb":
            param_groups, name_groups = build_partition.wb(self.model)
        elif self.args.logs_diff.partition == "trivial":
            param_groups, name_groups = build_partition.trivial(self.model)
        else:
            raise NotImplementedError(
                f"Not implemented: self.args.logs_diff.partition = {self.args.logs_diff.partition}"
            )

        direction = fullbatch_gradient(
            self.param_struct,
            self.loss_fn,
            self.model,
            self.train_loader_logs_hg,
            self.train_size,
            loader_pre_hook=self.loader_pre_hook,
        )

        lst_diff_n = diff_n_fullbatch(
            self.param_struct,
            self.args.logs_diff.order,
            self.full_loss,
            self.train_loader_logs_hg,
            self.train_size,
            direction,
            loader_pre_hook=self.loader_pre_hook,
        )

        logs["lst_diff_n"] = lst_diff_n

        if self.args.logs_diff.try_descent.use:
            pass

        return logs

    def lr_build_global(self, H, g):
        return g.sum() / H.sum()

    def lr_build_by_layer(self, H, g):
        # regul = self.args.optimizer.hg.epsilon * torch.eye(H.size(0), device = self.device, dtype = self.dtype)
        return torch.linalg.solve(H, g)
