import torch
from torch import nn
import torch.utils.data
from torch.nn import functional as F
import timm

from torchvision.transforms import v2 as transforms
import albumentations as A

import numpy as np

import argparse
import os

from .utils import *
from .model import GtNR
from .logger import MLLogger
from .datasets.hicks_dataset import HicksDataset
from .datasets.visdrone_dataset import VisDroneDataset
from .datasets.visdrone10_dataset import VisDrone10Dataset
from .datasets.isaid_dataset import ISAIDDataset
from .gpu_metric_tracker import GPUMetrics
from .auto_weighted_loss import AutomaticWeightedLoss
from .regional_gau_loss import RegionalGauLoss


import matplotlib
import matplotlib.pyplot as plt

import random

import tqdm

import cv2
import pandas
import shutil
import PIL

import time
import functools


class TrainingApplication():
    """
    The config for training our model
    """
    
    def __init__(self) -> None:
        # The CLU
        args = self._parse_args()

        # Store the arguments
        self.task_name: str              = args.task_name
        self.dataset_pth: str            = args.dataset_path
        self.weights_path_prefix: str    = args.weights_path_prefix
        self.seed: int                   = args.seed
        self.workers: int                = args.workers
        self.lr: float                   = args.learning_rate
        self.mask_loss_scale: float            = args.mask_loss_scale
        self.epochs: int                 = args.epochs
        self.weight_decay: float         = args.weight_decay
        self.batch_size: int             = args.batch_size
        self.update_frequency: int       = args.update_frequency
        self.debug_sample_frequency: int = args.debug_sample_frequency
        self.upload_model_freq: int      = args.upload_model_freq
        self.use_l2_weighting: bool      = args.use_weighted_l2 == 1
        self.drop_empty_images: bool     = args.drop_empty_images == 1

        # Scale learning rate with the square root of the batch size (Granziol et al. 2022)
        self.lr *= self.batch_size**0.5

        # Time-optimisation
        torch.set_num_threads(self.workers)
        # This coud cause false-positives with AMP
        torch.autograd.set_detect_anomaly(False)

        # Setup a logger
        # One of the first things we do so 𝙚𝙫𝙚𝙧𝙮𝙩𝙝𝙞𝙣𝙜 is included in our experiment tracker
        self.logger = MLLogger(task_name=self.task_name, weights_prefix = self.weights_path_prefix)
        self.gpu_metrics = GPUMetrics(self.logger)

        self.iteration       = 0
        self.epoch           = 0
        # Lie so 1 iteration is 1 image seen by the network
        # Makes it so our plots all match up
        self.iteration_step = self.batch_size

        # Setup torch things
        self._set_seeds(self.seed)

        # EXPERIMENT
        # Try initalising with ImageNet classification weights
        self.gnr = GtNR(categories=4, pretrained_is_imagenet=True, pretrained_path="pretrained_weights/imagenet/alt_gvt_base.pth").cuda()
        self.model       = nn.parallel.DataParallel(self.gnr)

        # How to transform the data
        self.crop_transform = A.CenterCrop(1536, 1024, pad_if_needed=True) # We want to resize to a common size
        
        self.augmentation_transforms = A.Compose([
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), # ImageNet
            self.crop_transform,
            A.HorizontalFlip(p=0.5), # Augmentations
            A.Rotate(limit=(-7.5, 7.5)),
            A.ToTensorV2()
        ], additional_targets={"density": "masks"}, seed=self.seed)

        # Validation transforms
        self.validation_transforms = A.Compose([
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), # ImageNet
            self.crop_transform,
            A.ToTensorV2()
        ], additional_targets={"density": "masks"}, seed=self.seed)

        # Data loaders
        self.selected_cats = None # means all 8 categories
        ds_time = time.time()
        self.train_dataset = VisDroneDataset(
            os.path.join(self.dataset_pth, "train_data_class8"), transform=self.augmentation_transforms, gt_scale=4, drop_empty_images=self.drop_empty_images, selected_cats=self.selected_cats)
        # Report
        print(f"Took {time.time() - ds_time} seconds to init training dataset")
        
        ds_time = time.time()
        # This is called "test_data_class8" because we use DSACA's density generaiton script
        self.val_dataset   = VisDroneDataset(
            os.path.join(self.dataset_pth, "test_data_class8"),  transform=self.validation_transforms,   gt_scale=4, drop_empty_images=self.drop_empty_images, selected_cats=self.selected_cats)
        # Report
        print(f"Took {time.time() - ds_time} seconds to init validation dataset")


        # If we've dropped cats, there's info to exploit for logging
        if (self.drop_empty_images): self._report_dataloader_stats()
            
        self.train_loader  = torch.utils.data.DataLoader(dataset=self.train_dataset, shuffle=True, batch_size=self.batch_size)
        self.val_loader    = torch.utils.data.DataLoader(  dataset=self.val_dataset, shuffle=False, batch_size=1)

        # Keep track of validation metrics to save new bests as they come
        self.validation_results = []

        # self.loss_func_den      = nn.MSELoss(reduction="sum").cuda()
        self.loss_func_den      = RegionalGauLoss(reduction="sum", function="mse", use_awl=False).cuda()
        self.loss_func_mask     = nn.CrossEntropyLoss().cuda()
        # self.auto_weighted_loss = AutomaticWeightedLoss(2)

        # Grad
        self.optimiser      = torch.optim.AdamW([
                {"params": self.model.parameters(recurse=True)},
                # {"params": self.loss_func_den.auto_weighted_loss.parameters(), "weight_decay": 0}
                # {"params": self.auto_weighted_loss.parameters(), "weight_decay": 0}
            ], lr=self.lr, weight_decay=self.weight_decay)
        self.shceduler      = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimiser, 'min', patience=5, factor=0.2)
        

        self.scaler = torch.amp.GradScaler(device="cuda")

        matplotlib.use("Agg")
        plt.rc('xtick',labelsize=8)
        plt.rc('ytick',labelsize=8)
        
        print(f"Model has {self.gnr.numparams:,} parameters")
        print(f"{'Using' if self.use_l2_weighting else 'Not using'} L2 loss weighting for densities")

    def train(self) -> None:
        for epoch in range(1, self.epochs + 1):
            # Forward and backward pass
            self.epoch = epoch
            self.train_one_epoch()

            # Validation
            print("Starting validation")
            val_metric, per_cat_val_metric, mask_metric = self.validation()

            # Report overall validation result
            self.logger.report_val_metric(val_metric, self.iteration)
            # Report mask things
            self.logger.report_val_mask_metric(mask_metric, self.iteration)

            # Report per-category validation results
            for ci, cat in enumerate(self.val_dataset.categories):
                self.logger.report_val_metric(per_cat_val_metric[ci], self.iteration, category=cat)

            print(f"Val {val_metric}")

            # The LR scheduler
            self.shceduler.step(val_metric)
            self.logger.report_hyperparams(learning_rate=self.optimiser.param_groups[0]["lr"], iteration=self.iteration)

            # Save weights (and figure out if we've got a new best model)
            if (len(self.validation_results) > 0):
                new_best = (val_metric < np.min(self.validation_results))
            else:
                new_best = True
            
            # Append
            self.validation_results.append(val_metric)
            self.logger.save_weights(self, best=new_best)

    def train_one_epoch(self) -> None:            
        print(f"Begin epoch {self.epoch}")
        losses = []
        losses_den  = []
        losses_mask = []
        
        tqdm_trainloader = tqdm.tqdm(self.train_loader, desc=f"Epoch {self.epoch} batches start")
        for i, batch in enumerate(tqdm_trainloader):
            image   = batch['image'].cuda()
            gt_den  = batch['gt'].cuda()
            gt_mask = batch['gt_mask'].cuda()

            # Zero out gradients from previous iteration
            self.optimiser.zero_grad()
            # Forward pass
            output_den, output_mask = self.model(image)

            mask_loss = 0
            den_loss  = 0

            if (not self.use_l2_weighting):
                den_loss  = self.loss_func_den(output_den, gt_den)

            
            # As DSACA does, we treat the masking as n individual problems
            # So we have a masking layer that predicts car and not car, then a masking layer that predicts person and not person etc.
            for ci in range(self.val_dataset.cat_count):
                mask_pred = output_mask[:, ci*2:(ci+1)*2]
                mask_gt   = gt_mask[:, ci].long() # This needs to be long, as these are effectively class indexes (0 for not person and 1 for person)
                
                mask_loss += self.loss_func_mask(mask_pred, mask_gt.long()) * (self.mask_loss_scale / self.val_dataset.cat_count)

                if (self.use_l2_weighting):
                    den_loss += self.loss_func_den(output_den[:,ci], gt_den[:,ci]) * self.train_dataset.category_importance[ci]
            
            # loss = self.auto_weighted_loss(den_loss, mask_loss)
            loss = den_loss + mask_loss

            losses.append(loss.item())
            losses_den.append(den_loss.item())
            losses_mask.append(mask_loss.item())

            mean_loss = np.mean(losses)

            # Update stats every so often
            if ((i % self.update_frequency) == 0):
                self.logger.report_train_loss(current=loss.item(), mean=mean_loss, iteration=self.iteration)
                self.logger.report_train_loss(current=den_loss.item(), mean=np.mean(losses_den), iteration=self.iteration, prefix="Density L2")
                self.logger.report_train_loss(current=mask_loss.item(), mean=np.mean(losses_mask), iteration=self.iteration, prefix="Mask Loss")
                self.gpu_metrics.gpu_power_report(iteration=self.iteration)
                # Update the TQDM message
                tqdm_trainloader.set_description(f"E{self.epoch} Train loss {loss.item():0.3f} [{mean_loss:0.3f}]")

            
            # Backprop
            loss.backward()
            self.optimiser.step()

            # We just completed an iteration
            self.iteration += self.iteration_step


    @torch.no_grad()
    def validation(self) -> float:
        val_metrics      = []
        val_metrics_mask = []
        cat_val_metrics  = [ [] for c in range(self.val_dataset.cat_count) ]
        
        tqdm_valloader = tqdm.tqdm(self.val_loader, desc="Validation")
        
        for i, batch in enumerate(tqdm_valloader):
            image   = batch['image'].cuda()
            gt      = batch['gt'].cuda()
            gt_mask = batch['gt_mask'].cuda()
            
            output_den, output_mask = self.model(image)
            
            for c in range(self.val_dataset.cat_count):
                cat_val_metrics[c].append(
                    float(abs(gt[:,c,:,:].sum() - output_den[:,c,:,:].sum()).cpu() / gt.shape[0]) # Get MAE for this cat and divide by batch size,
                )

            # Get the mean MAE over all cats
            val_metrics.append(np.array(cat_val_metrics)[:,-1].mean())

            mask_loss = 0

            # Log the mask loss to make sure it's decreasing in validation
            for ci in range(self.val_dataset.cat_count):
                mask_pred = output_mask[:, ci*2:(ci+1)*2, :, :]
                mask_gt   = gt_mask[:, ci, :, :].long() # This needs to be long, as these are effectively class indexes (0 for not person and 1 for person)
                
                mask_loss += self.loss_func_mask(mask_pred, mask_gt.long()) * (self.mask_loss_scale / self.val_dataset.cat_count)

            val_metrics_mask.append(mask_loss.item())
            

            self.gpu_metrics.gpu_power_tick()

            if ((i % self.debug_sample_frequency) == 0):
                self.logger.report_debug_samples(self, batch, output_den, output_mask)

            # Update the TQDM instance
            tqdm_valloader.set_description(f"Validation {np.mean(val_metrics):0.1f} MAE")

        # Get the overall accuracy, and per-category accuracy
        return np.mean(val_metrics), np.array(cat_val_metrics).mean(axis=1), np.mean(val_metrics_mask)

    def _report_dataloader_stats(self):
        """If we're dropping empty images, we have stats to report"""
        
        print(f"Dropped {self.train_dataset.dropped_images} images from testing set and {self.val_dataset.dropped_images} from the validation set")
        print(f"Testing set has {self.train_dataset.counts.sum()} (mean of {self.train_dataset.counts.mean()}) examples, validation has {self.val_dataset.counts.sum()} ({self.val_dataset.counts.mean()})")

        print(f"Train set per cat counts:")
        print(functools.reduce( lambda a,b: a + "\n" + b,
            [
                f"{self.train_dataset.categories[ci]}:\t{count:.1f} ({mean:.3f})"
                for ci, (count, mean) in enumerate(zip(
                    self.train_dataset.count_cats.sum(axis=0).tolist(),
                    self.train_dataset.count_cats.mean(axis=0).tolist())) ] ))
        print("\n")

        print(f"Val set per cat counts:")
        print(functools.reduce( lambda a,b: a + "\n" + b,
            [
                f"{self.val_dataset.categories[ci]}:\t{count:.1f} ({mean:.3f})"
                for ci, (count, mean) in enumerate(zip(
                    self.val_dataset.count_cats.sum(axis=0).tolist(),
                    self.val_dataset.count_cats.mean(axis=0).tolist()))] ))

        print("\n")
        print(f"Mean across all cats in val {self.val_dataset.count_cats.mean(axis=0).mean()}")
        print(f"Caluclated importance in train {self.train_dataset.category_importance}")


    def _set_seeds(self, seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)

    def _parse_args(self) -> argparse.Namespace:
        args = argparse.ArgumentParser(
            prog="GtNRTrainer",
            description="The training script for this Twins-based biodiversity monitoring machine learning model",
            epilog="Please cite our paper!"
        )

        args.add_argument(
            "--task_name", "-n",
            help="The name of the task (used in logging and saving checkpoints)",
            type=lambda n: n if (n.replace("_", "").isalnum()) else raise_(argparse.ArgumentTypeError(f"Task name '{n}' must be better")),
            default="mt_invregional")

        args.add_argument(
            "--dataset_path", "-d",
            help="The directory of the dataset to be used for training (and validation)",
            type=lambda d: d if (os.path.isdir(d)) else raise_(NotADirectoryError(d)),
            default="datasets/VisDrone/")

        args.add_argument(
            "--weights_path_prefix", "-wp",
            help="The directory to save weights into (the task ID is appended to this)",
            type=lambda d: d if (os.path.isdir(d) or (not os.path.exists(d))) else raise_(NotADirectoryError(d)),
            default="weights/")

        args.add_argument(
            "--seed", "-s",
            help="The seed used to initalise many states",
            type=int,
            default=420)

        args.add_argument(
            "--workers", "-w",
            help="Data loaer workers",
            type=int,
            default=8)

        args.add_argument(
            "--learning_rate", "-lr",
            type=float,
            default=1e-4,
            help="Set the initial* learning rate")

        args.add_argument(
            "--mask_loss_scale",
            type=float,
            default=100.0,
            help="The value to multiple the mask loss function by")

        args.add_argument(
            "--epochs", "-e",
            type=int,
            default=150,
            help="Stop training after e epochs")

        args.add_argument(
            "--batch_size", "-bs",
            type=int,
            default=1,
            help="Batch size to use")

        args.add_argument(
            "--weight_decay", "-wd",
            type=float,
            default=0.0001,
            help="Set the weight decay (L2 regularization) for the optimiser")

        args.add_argument(
            "--update_frequency",
            type=int,
            default=1,
            help="Update logging every 𝔫 iterations")

        args.add_argument(
            "--debug_sample_frequency",
            type=int,
            default=32,
            help="For every 𝔫 validation iterations, output 1 debug sample")

        args.add_argument(
            "--upload_model_freq",
            type=int,
            default=2,
            help="Save (upload/report) the model pth every 𝔫 epochs (the model is always uploaded if it's a best)")
        
        args.add_argument(
            "--use_weighted_l2",
            type=int,
            default=0,
            help="Use the weighted L2 loss (1 == True)")

        args.add_argument(
            "--drop_empty_images",
            type=int,
            default=0,
            help="Drop empty images (1 == True)")

        return args.parse_args()

