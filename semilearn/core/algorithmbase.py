


import contextlib
import os
from collections import OrderedDict
from inspect import signature
import pandas as pd

import numpy as np
import torch
import torch.nn.functional as F
from semilearn.core.criterions import CELoss, ConsistencyLoss, GCELoss, calculate_ece
from semilearn.core.hooks import (
    AimHook,
    CheckpointHook,
    DistSamplerSeedHook,
    EMAHook,
    EvaluationHook,
    Hook,
    LoggingHook,
    ParamUpdateHook,
    TimerHook,
    WANDBHook,
    get_priority,
)
from semilearn.core.utils import (
    Bn_Controller,
    get_cosine_schedule_with_warmup,
    get_data_loader,
    get_dataset,
    get_optimizer,
)
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from torch.cuda.amp import GradScaler, autocast


class AlgorithmBase:
    """
    Base class for algorithms
    init algorithm specific parameters and common parameters

    Args:
        - args (`argparse`):
            algorithm arguments
        - net_builder (`callable`):
            network loading function
        - tb_log (`TBLog`):
            tensorboard logger
        - logger (`logging.Logger`):
            logger to use
    """

    def __init__(self, args, net_builder, tb_log=None, logger=None, **kwargs):


        self.args = args
        self.num_classes = args.num_classes
        self.ema_m = args.ema_m
        self.epochs = args.epoch
        self.num_train_iter = args.num_train_iter
        self.num_eval_iter = args.num_eval_iter
        self.num_log_iter = args.num_log_iter
        self.num_iter_per_epoch = int(self.num_train_iter // self.epochs)
        self.lambda_u = args.ulb_loss_ratio
        self.use_cat = args.use_cat
        self.use_amp = args.amp
        self.clip_grad = args.clip_grad
        self.save_name = args.save_name
        self.save_dir = args.save_dir
        self.resume = args.resume
        self.algorithm = args.algorithm


        self.tb_log = tb_log
        self.logger = logger

        if logger is None:
            self.print_fn = print
        else:
            def print_wrapper(*args, **kwargs):
                logger.info(*args, **kwargs)

            print_wrapper.info = logger.info
            self.print_fn = print_wrapper
        self.ngpus_per_node = torch.cuda.device_count()
        self.loss_scaler = GradScaler()
        self.amp_cm = autocast if self.use_amp else contextlib.nullcontext
        self.gpu = args.gpu
        self.rank = args.rank
        self.distributed = args.distributed
        self.world_size = args.world_size


        self.it = 0
        self.start_epoch = 0
        self.best_eval_acc, self.best_it = 0.0, 0
        self.bn_controller = Bn_Controller()
        self.net_builder = net_builder
        self.ema = None


        self.dataset_dict = self.set_dataset()

        self.classnames = self.dataset_dict.get('classnames', None)

        self.per_class_prompts = self._load_per_class_prompts(args.per_class_prompts_path)


        self.loader_dict = self.set_data_loader()


        self.model = self.set_model()
        if not self.use_amp:
            self.print_fn("AMP is disabled. Forcing model weights to Float32 to avoid type mismatch.")
            self.model.float()
        self.ema_model = self.set_ema_model()


        self.optimizer, self.scheduler = self.set_optimizer()


        self.ce_loss = CELoss()
        self.consistency_loss = ConsistencyLoss()
        self.gce_loss = GCELoss(num_classes=self.num_classes)





        self._hooks = []
        self.hooks_dict = OrderedDict()
        self.set_hooks()

    def init(self, **kwargs):
        """
        algorithm specific init function, to add parameters into class
        """
        raise NotImplementedError

    def set_dataset(self):
        """
        set dataset_dict
        """
        if self.rank != 0 and self.distributed:
            torch.distributed.barrier()

        if self.algorithm == 'SAST_tl':
            dataset_dict = self.get_dataset_tl()
        else:
            dataset_dict = get_dataset(
                self.args,
                self.algorithm,
                self.args.dataset,
                self.args.num_labels,
                self.args.num_classes,
                self.args.data_dir,
                self.args.include_lb_to_ulb,
            )

        if dataset_dict is None:
            return dataset_dict

        self.args.ulb_dest_len = (
            len(dataset_dict["train_ulb"])
            if dataset_dict["train_ulb"] is not None
            else 0
        )
        self.args.lb_dest_len = len(dataset_dict["train_lb"])

        self.print_fn(
            "unlabeled data number: {}, labeled data number {}".format(
                self.args.ulb_dest_len, self.args.lb_dest_len
            )
        )
        if self.rank == 0 and self.distributed:
            torch.distributed.barrier()
        return dataset_dict
    def get_dataset_tl(self):
        """
        Specific dataset getter for Transductive Learning (TL) algorithms.
        Calls specialized get_*_tl functions in dataset modules.
        """
        from semilearn.datasets import get_dtd_tl, get_cub_tl, get_eurosat_tl, get_resisc_tl, get_flowers_tl, get_fgvc_tl


        dataset = self.args.dataset
        args = self.args
        alg = self.algorithm
        num_classes = self.args.num_classes
        data_dir = self.args.data_dir

        if dataset == 'dtd':
            lb_dset, ulb_dset, test_dset, classnames, seen_classes, unseen_classes = \
                get_dtd_tl(args, alg, 'dtd', num_classes, data_dir)
        elif dataset == 'cub':
            lb_dset, ulb_dset, test_dset, classnames, seen_classes, unseen_classes = \
                get_cub_tl(args, alg, 'cub', num_classes, data_dir)
        elif dataset == 'eurosat':
            lb_dset, ulb_dset, test_dset, classnames, seen_classes, unseen_classes = \
                get_eurosat_tl(args, alg, 'eurosat', num_classes, data_dir)
        elif dataset == 'resisc45':
            lb_dset, ulb_dset, test_dset, classnames, seen_classes, unseen_classes = \
                get_resisc_tl(args, alg, 'resisc45', num_classes, data_dir)
        elif dataset == 'flowers102':
            lb_dset, ulb_dset, test_dset, classnames, seen_classes, unseen_classes = \
                get_flowers_tl(args, alg, 'flowers', num_classes, data_dir)
        elif dataset == 'fgvcaircraft':
            lb_dset, ulb_dset, test_dset, classnames, seen_classes, unseen_classes = \
                get_fgvc_tl(args, alg, 'fgvcaircraft', num_classes, data_dir)
        else:
            raise NotImplementedError(f"TL dataset getter not implemented for {dataset}")

        return {
            'train_lb': lb_dset,
            'train_ulb': ulb_dset,
            'eval': test_dset,
            'test': test_dset,
            'classnames': classnames,
            'seen_classes': seen_classes,
            'unseen_classes': unseen_classes
        }

    def set_data_loader(self):
        """
        set loader_dict
        """
        if self.dataset_dict is None:
            return

        self.print_fn("Create train and test data loaders")
        loader_dict = {}
        loader_dict["train_lb"] = get_data_loader(
            self.args,
            self.dataset_dict["train_lb"],
            self.args.batch_size,
            data_sampler=self.args.train_sampler,
            num_iters=self.num_train_iter,
            num_epochs=self.epochs,
            num_workers=self.args.num_workers,
            distributed=self.distributed,
        )

        loader_dict["train_ulb"] = get_data_loader(
            self.args,
            self.dataset_dict["train_ulb"],
            int(self.args.batch_size * self.args.uratio),
            data_sampler=self.args.train_sampler,
            num_iters=self.num_train_iter,
            num_epochs=self.epochs,
            num_workers=2 * self.args.num_workers,
            distributed=self.distributed,
        )

        loader_dict['eval'] = get_data_loader(
            self.args,
            self.dataset_dict['eval'],
            self.args.eval_batch_size,

            data_sampler=None,
            num_workers=self.args.num_workers,
            drop_last=False)


        loader_dict["warm_up_eval"] = get_data_loader(
            self.args,
            self.dataset_dict["train_ulb"],
            self.args.eval_batch_size,

            data_sampler=None,
            num_workers=self.args.num_workers,
            drop_last=False,
        )

        if self.dataset_dict["test"] is not None:
            loader_dict["test"] = get_data_loader(
                self.args,
                self.dataset_dict["test"],
                self.args.eval_batch_size,

                data_sampler=None,
                num_workers=self.args.num_workers,
                drop_last=False,
            )
        self.print_fn(f"[!] data loader keys: {loader_dict.keys()}")
        return loader_dict

    def set_optimizer(self):
        """
        set optimizer for algorithm
        """
        self.print_fn("Create optimizer and scheduler")
        optimizer = get_optimizer(
            self.model,
            self.args.optim,
            self.args.lr,
            self.args.momentum,
            self.args.weight_decay,
            self.args.layer_decay,
            eps=self.args.eps if hasattr(self.args, 'eps') else 1e-8
        )
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, self.num_train_iter, num_warmup_steps=self.args.num_warmup_iter
        )
        return optimizer, scheduler

    def set_model(self):
        """
        initialize model
        """
        model_args = {
            'num_classes': self.num_classes,
            'pretrained': self.args.use_pretrain,
            'pretrained_path': self.args.pretrain_path,
            'classnames': self.classnames
        }
        if hasattr(self.args, 'prompt_template') and self.args.prompt_template is not None:
            model_args['prompt_template'] = self.args.prompt_template
        if self.per_class_prompts is not None:
            model_args['per_class_prompts'] = self.per_class_prompts

        if hasattr(self.args, 'prompts_path') and self.args.prompts_path is not None:
            model_args['prompts_path'] = self.args.prompts_path

        if hasattr(self.args, 'k'):
            model_args['k'] = self.args.k


        model = self.net_builder(**model_args)
        return model
    def set_ema_model(self):
        """
        initialize ema model from model
        """
        model_args = {
            'num_classes': self.num_classes,
            'pretrained': self.args.use_pretrain,
            'pretrained_path': self.args.pretrain_path,
            'classnames': self.classnames
        }
        if hasattr(self.args, 'prompt_template') and self.args.prompt_template is not None:
            model_args['prompt_template'] = self.args.prompt_template

        if self.per_class_prompts is not None:
            model_args['per_class_prompts'] = self.per_class_prompts

        if hasattr(self.args, 'prompts_path') and self.args.prompts_path is not None:
            model_args['prompts_path'] = self.args.prompts_path

        if hasattr(self.args, 'k'):
            model_args['k'] = self.args.k
        ema_model = self.net_builder(**model_args)
        ema_model.load_state_dict(self.model.state_dict())
        return ema_model

    def set_hooks(self):
        """
        register necessary training hooks
        """

        self.register_hook(ParamUpdateHook(), None, "HIGHEST")
        self.register_hook(EMAHook(), None, "HIGH")
        self.register_hook(EvaluationHook(), None, "HIGH")
        self.register_hook(CheckpointHook(), None, "HIGH")
        self.register_hook(DistSamplerSeedHook(), None, "NORMAL")
        self.register_hook(TimerHook(), None, "LOW")
        self.register_hook(LoggingHook(), None, "LOWEST")
        if self.args.use_wandb:
            self.register_hook(WANDBHook(), None, "LOWEST")
        if self.args.use_aim:
            self.register_hook(AimHook(), None, "LOWEST")

    def process_batch(self, input_args=None, **kwargs):
        """
        process batch data, send data to cuda
        NOTE: **kwargs should have the same arguments to train_step function as keys to
        work properly.
        """
        if input_args is None:
            input_args = signature(self.train_step).parameters
            input_args = list(input_args.keys())

        input_dict = {}

        for arg, var in kwargs.items():
            if arg not in input_args:
                continue

            if var is None:
                continue


            if isinstance(var, dict):
                var = {k: v.cuda(self.gpu) for k, v in var.items()}
            else:
                var = var.cuda(self.gpu)
            input_dict[arg] = var
        return input_dict

    def process_out_dict(self, out_dict=None, **kwargs):
        """
        process the out_dict as return of train_step
        """
        if out_dict is None:
            out_dict = {}

        for arg, var in kwargs.items():
            out_dict[arg] = var


        return out_dict

    def process_log_dict(self, log_dict=None, prefix="train", **kwargs):
        """
        process the tb_dict as return of train_step
        """
        if log_dict is None:
            log_dict = {}

        for arg, var in kwargs.items():
            log_dict[f"{prefix}/" + arg] = var
        return log_dict

    def compute_prob(self, logits):
        return torch.softmax(logits, dim=-1)

    def train_step(self, idx_lb, x_lb, y_lb, idx_ulb, x_ulb_w, x_ulb_s, y_ulb):
        """
        train_step specific to each algorithm
        """





        raise NotImplementedError

    def train(self):
        """
        train function
        """
        self.model.train()
        self.call_hook("before_run")

        for epoch in range(self.start_epoch, self.epochs):
            self.epoch = epoch


            if self.it >= self.num_train_iter:
                break

            self.call_hook("before_train_epoch")

            for data_lb, data_ulb in zip(
                self.loader_dict["train_lb"], self.loader_dict["train_ulb"]
            ):

                if self.it >= self.num_train_iter:
                    break

                self.call_hook("before_train_step")
                self.out_dict, self.log_dict = self.train_step(
                    **self.process_batch(**data_lb, **data_ulb)
                )
                self.call_hook("after_train_step")
                self.it += 1

            self.call_hook("after_train_epoch")

        self.call_hook("after_run")

    def evaluate(self, eval_dest="eval", out_key="logits", return_logits=False):
        self.model.eval()
        self.ema.apply_shadow()

        eval_loader = self.loader_dict[eval_dest]
        total_loss = 0.0
        total_num = 0.0
        y_true = []
        y_pred = []
        y_logits = []
        with torch.no_grad():
            for data in eval_loader:
                x = data["x_lb"]
                y = data["y_lb"]

                if isinstance(x, dict):
                    x = {k: v.cuda(self.gpu) for k, v in x.items()}
                else:
                    x = x.cuda(self.gpu)
                y = y.cuda(self.gpu)

                num_batch = y.shape[0]
                total_num += num_batch

                outputs = self.model(x)
                logits = outputs['logits_text']

                loss = F.cross_entropy(logits, y, reduction="mean", ignore_index=-1)
                y_true.extend(y.cpu().tolist())
                y_pred.extend(torch.max(logits, dim=-1)[1].cpu().tolist())
                y_logits.append(logits.cpu().numpy())
                total_loss += loss.item() * num_batch

        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        y_logits = np.concatenate(y_logits)

        top1 = accuracy_score(y_true, y_pred)
        balanced_top1 = balanced_accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average="macro", zero_division=0)
        recall = recall_score(y_true, y_pred, average="macro", zero_division=0)
        F1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
        ece = calculate_ece(torch.tensor(y_logits), torch.tensor(y_true))

        run_detailed_analysis = False
        stage = ""

        if self.epoch == 1:
            run_detailed_analysis = True
            stage = "early"
        elif self.epoch == self.epochs // 2:
            run_detailed_analysis = True
            stage = "mid"
        elif self.epoch == self.epochs - 1:
            run_detailed_analysis = True
            stage = "late"

        if run_detailed_analysis:
            self.print_fn(f"Running detailed analysis for epoch {self.epoch} ({stage} stage)...")
            cf_mat = confusion_matrix(y_true, y_pred, normalize="true")

            try:
                if not os.path.exists(self.args.save_dir):
                    os.makedirs(self.args.save_dir, exist_ok=True)

                cf_mat_df = pd.DataFrame(cf_mat, index=self.classnames, columns=self.classnames)
                save_path = os.path.join(self.args.save_dir, f"confusion_matrix_{self.save_name}_{eval_dest}_{stage}_epoch{self.epoch}.csv")
                cf_mat_df.to_csv(save_path)
                self.print_fn(f"Confusion matrix ({stage} stage) saved to: {save_path}")
            except Exception as e:
                self.print_fn(f"Failed to save confusion matrix to CSV. Error: {e}")
                self.print_fn("Fallback to printing raw matrix:\n" + np.array_str(cf_mat, max_line_width=180))

            class_recalls = cf_mat.diagonal()
            self.print_fn(f"Per-class Recall ({stage} stage):")
            recall_log = ""
            for i, class_name in enumerate(self.classnames):
                recall_log += f"  - {class_name}: {class_recalls[i]:.4f}\n"
            self.print_fn(recall_log)
        else:
            self.print_fn(f"Running standard evaluation for epoch {self.epoch}.")

        self.ema.restore()
        self.model.train()

        eval_dict = {
            eval_dest + "/loss": total_loss / total_num,
            eval_dest + "/top-1-acc": top1,
            eval_dest + "/balanced_acc": balanced_top1,
            eval_dest + "/precision": precision,
            eval_dest + "/recall": recall,
            eval_dest + "/F1": F1,
            eval_dest + "/ece": ece,
        }

        if return_logits:
            eval_dict[eval_dest + "/logits"] = y_logits
        return eval_dict

    def save_model(self, save_name, save_path):
        """
        save model and specified parameters for resume
        """
        if not os.path.exists(save_path):
            os.makedirs(save_path, exist_ok=True)
        save_filename = os.path.join(save_path, save_name)
        save_dict = self.get_save_dict()
        torch.save(save_dict, save_filename)
        self.print_fn(f"model saved: {save_filename}")

    def load_model(self, load_path):
        """
        Load a model and the necessary parameters for resuming training.
        """
        checkpoint = torch.load(load_path, map_location="cpu")

        self.model.load_state_dict(checkpoint["model"])
        self.ema_model.load_state_dict(checkpoint["ema_model"])
        self.loss_scaler.load_state_dict(checkpoint["loss_scaler"])
        self.it = checkpoint["it"]
        self.start_epoch = checkpoint["epoch"]
        self.epoch = self.start_epoch
        self.best_it = checkpoint["best_it"]
        self.best_eval_acc = checkpoint["best_eval_acc"]
        self.optimizer.load_state_dict(checkpoint["optimizer"])

        if self.scheduler is not None and "scheduler" in checkpoint:

            self.scheduler.load_state_dict(checkpoint["scheduler"])

        if "aim_run_hash" in checkpoint:

            self.aim_run_hash = checkpoint["aim_run_hash"]

        self.print_fn("Model loaded")

        return checkpoint

    def check_prefix_state_dict(self, state_dict):
        """
        remove prefix state dict in ema model
        """
        new_state_dict = dict()
        for key, item in state_dict.items():
            if key.startswith("module"):
                new_key = ".".join(key.split(".")[1:])
            else:
                new_key = key
            new_state_dict[new_key] = item
        return new_state_dict

    def register_hook(self, hook, name=None, priority="NORMAL"):
        assert isinstance(hook, Hook)
        if hasattr(hook, "priority"):
            raise ValueError('"priority" is a reserved attribute for hooks')
        priority = get_priority(priority)
        hook.priority = priority
        hook.name = name if name is not None else type(hook).__name__


        inserted = False
        for i in range(len(self._hooks) - 1, -1, -1):
            if priority >= self._hooks[i].priority:
                self._hooks.insert(i + 1, hook)
                inserted = True
                break

        if not inserted:
            self._hooks.insert(0, hook)


        self.hooks_dict = OrderedDict()
        for hook in self._hooks:
            self.hooks_dict[hook.name] = hook

    def call_hook(self, fn_name, hook_name=None, *args, **kwargs):
        """Call all hooks.
        Args:
            fn_name (str): The function name in each hook to be called, such as
                "before_train_epoch".
            hook_name (str): The specific hook name to be called, such as
                "param_update" or "dist_align", used to call single hook in train_step.
        """

        if hook_name is not None:
            return getattr(self.hooks_dict[hook_name], fn_name)(self, *args, **kwargs)

        for hook in self.hooks_dict.values():
            if hasattr(hook, fn_name):
                getattr(hook, fn_name)(self, *args, **kwargs)

    def registered_hook(self, hook_name):
        """
        Check if a hook is registered
        """
        return hook_name in self.hooks_dict

    @staticmethod
    def get_argument():
        """
        Get specified arguments into argparse for each algorithm
        """
        return {}

    def _load_per_class_prompts(self, path):
        if path is None:
            return None

        prompts = {}
        try:
            with open(path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line or ':' not in line:
                        continue
                    classname, prompt = line.split(':', 1)
                    prompts[classname.strip()] = prompt.strip()
            self.print_fn(f"Successfully loaded {len(prompts)} per-class prompts from {path}")
            return prompts
        except Exception as e:
            self.print_fn(f"Warning: Failed to load per-class prompts from {path}. Error: {e}")
            return None
    def get_save_dict(self):
        """
        Create a dictionary of additional arguments to save for when saving the model.
        """

        save_dict = {
            "model": self.model.state_dict(),
            "ema_model": self.ema_model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "loss_scaler": self.loss_scaler.state_dict(),
            "it": self.it + 1,
            "epoch": self.epoch + 1,
            "best_it": self.best_it,
            "best_eval_acc": self.best_eval_acc,
        }

        if self.scheduler is not None:

            save_dict["scheduler"] = self.scheduler.state_dict()

        if hasattr(self, "aim_run_hash"):

            save_dict["aim_run_hash"] = self.aim_run_hash

        return save_dict


class ImbAlgorithmBase(AlgorithmBase):
    def __init__(self, args, net_builder, tb_log=None, logger=None, **kwargs):
        super().__init__(args, net_builder, tb_log, logger, **kwargs)


        self.lb_imb_ratio = self.args.lb_imb_ratio
        self.ulb_imb_ratio = self.args.ulb_imb_ratio
        self.imb_algorithm = self.args.imb_algorithm

    def imb_init(self, *args, **kwargs):
        """
        initialize imbalanced algorithm parameters
        """
        pass

    def set_optimizer(self):
        if "vit" in self.args.net and self.args.dataset in [
            "cifar100",
            "food101",
            "semi_aves",
            "semi_aves_out",
        ]:
            return super().set_optimizer()
        elif self.args.dataset in ["imagenet", "imagenet127"]:
            return super().set_optimizer()












        else:
            self.print_fn("Create optimizer and scheduler")
            optimizer = get_optimizer(
                self.model,
                self.args.optim,
                self.args.lr,
                self.args.momentum,
                self.args.weight_decay,
                self.args.layer_decay,
                bn_wd_skip=False,
            )
            scheduler = get_cosine_schedule_with_warmup(
                optimizer, self.num_train_iter, num_warmup_steps=self.args.num_warmup_iter
            )
            return optimizer, scheduler