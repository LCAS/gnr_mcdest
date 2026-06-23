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
from .model import ModelBackbone
from .logger import MLLogger
from .datasets.visdrone_dataset import VisDroneDataset
from .datasets.visdrone10_dataset import VisDrone10Dataset
from .datasets.isaid_dataset import ISAIDDataset
from .datasets.hicks_dataset import HicksDataset
from .gpu_metric_tracker import GPUMetrics

import matplotlib
import matplotlib.pyplot as plt

import random

import tqdm

import cv2
import pandas
import shutil
import PIL
import math


class TestingApplication():
    """
    The config for testing our model
    """
    
    def __init__(self, testing=False) -> None:
        # The CLU
        args = self._parse_args()

        # Store the arguments
        self.dataset_pth: str            = args.dataset_path
        self.weights_path_prefix: str    = args.weights_path_prefix
        self.seed: int                   = args.seed
        self.workers: int                = args.workers
        self.model_testing_dir: str = args.model_state_testing_directory

        # Time-optimisation
        torch.set_num_threads(self.workers)

        # Setup torch things
        self._set_seeds(self.seed)

        self.gnr = GtNR(categories=8).cuda()
        # self.gnr = GtNR(categories=10).cuda()
        # self.gnr = GtNR(categories=8, backbone=ModelBackbone.TWINS_SVT_SMALL, pretrained_path="pretrained_weights/upernet_alt_gvt_s_512x512_160k_ade20k_swin_setting.pth").cuda()
        # self.gnr = GtNR(categories=8, backbone=ModelBackbone.TWINS_SVT_LARGE, pretrained_path="pretrained_weights/upernet_alt_gvt_l_512x512_160k_ade20k_swin_setting.pth").cuda()
        self.model       = nn.parallel.DataParallel(self.gnr)

        # How to transform the data
        self.crop_transform = A.CenterCrop(768, 1024, pad_if_needed=True) # We want to resize to a common size

        # Validation transforms
        self.validation_transforms = A.Compose([
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), # ImageNet
            self.crop_transform,
            A.ToTensorV2()
        ], additional_targets={"density": "masks"}, seed=self.seed)

        # Data loaders

        self.selected_cats = ['leucanthemum_vulgare', 'raununculus_spp', 'heracleum_sphondylium', 'silene_dioica-latifolia', 'trifolium_repens', 'cirsium_arvense', 'stachys_sylvatica', 'rubus_fruticosus_agg']
        self.drop_empty_images = True

        self.test_dataset = HicksDataset(
            os.path.join(self.dataset_pth, "test_c25_data"),  transform=self.validation_transforms,   gt_scale=4, drop_empty_images=self.drop_empty_images, selected_cats=self.selected_cats)
        
        # selected_cats = None means all 8 categories
        # self.test_dataset  = VisDrone10Dataset(os.path.join(self.dataset_pth, "test_den"),  transform=self.validation_transforms,   gt_scale=4, selected_cats = None)
        # self.test_dataset  = VisDroneDataset(os.path.join(self.dataset_pth, "truetest_data_class8"),  transform=self.validation_transforms,   gt_scale=4, selected_cats = None)


        self.test_loader   = torch.utils.data.DataLoader( dataset=self.test_dataset, shuffle=False, batch_size=1)


        # If we've dropped cats, there's info to exploit for logging
        if (self.drop_empty_images):
            print(f"Dropped {self.test_dataset.dropped_images} images from testing set")
            print(f"Testing set has {self.test_dataset.counts.sum()} (mean of {self.test_dataset.counts.mean()}) examples")

            print(f"Test set per cat counts:")
            print(functools.reduce( lambda a,b: a + "\n" + b,
                [
                    f"{self.test_dataset.categories[ci]}:\t{count:.1f} ({mean:.3f})"
                    for ci, (count, mean) in enumerate(zip(
                        self.test_dataset.count_cats.sum(axis=0).tolist(),
                        self.test_dataset.count_cats.mean(axis=0).tolist())) ] ))
            print("\n")

            print("\n")
            print(f"Mean across all cats in test {self.test_dataset.count_cats.mean(axis=0).mean()}")


        matplotlib.use("Agg")
        plt.rc('xtick',labelsize=8)
        plt.rc('ytick',labelsize=8)
        
        print(f"Model has {self.gnr.numparams:,} parameters")


    @torch.no_grad()
    def test_all(self) -> None:
        pth_files = [os.path.join(self.model_testing_dir, f) for f in sorted(os.listdir(self.model_testing_dir)) if f.endswith('.pth')]

        for pth in pth_files:
            # Load it
            state = torch.load(pth, weights_only=False)
            self.model.load_state_dict(state["states"], assign=True, strict=True)

            print(f"Loaded {state['nparams']:,} parameters from {pth}")
            print(f"Testing {state['task_name']} from {state['task_id']}")

            # Test it
            self.test_one(pth, state)


    @torch.no_grad()
    def test_one(self, pth_path: str, state) -> None:
        mae_metrics      = []
        mse_metrics      = []
        rmse_metrics     = []

        masked_mae_metrics      = []
        masked_mse_metrics      = []
        masked_rmse_metrics     = []

        cat_mae_metrics  = [ [] for c in range(self.test_dataset.cat_count) ]
        cat_mse_metrics  = [ [] for c in range(self.test_dataset.cat_count) ]
        cat_masked_mae_metrics  = [ [] for c in range(self.test_dataset.cat_count) ]
        cat_masked_mse_metrics  = [ [] for c in range(self.test_dataset.cat_count) ]

        metrics_df = []
        metrics_image_df = []

        # Get place to save things to
        test_save_dir = os.path.join(self.model_testing_dir, state["task_id"])
        if (os.path.isdir(test_save_dir)):
            print(f"{test_save_dir} exists, skipping")
            return
        else:
            os.makedirs(test_save_dir)
        
        tqdm_testloader = tqdm.tqdm(self.test_loader, desc="Testing")
        
        for i, batch in enumerate(tqdm_testloader):
            image    = batch['image'].cuda()
            gt       = batch['gt'].cuda()
            gt_mask  = batch['gt_mask'].cuda()
            filename = os.path.splitext(os.path.split( batch["img_path"][0] )[1])[0]
            
            samples_path = os.path.join(test_save_dir, "samples", filename)
            os.makedirs(samples_path)
            
            output_den, mask_output = self.model(image)

            # Process the mask logits
            #
            # Take in the logits for each pair of cat/nocat predictions, and produce the masks
            # What we do here:
            #    For every category there is a positive and negative case (the SECOND channel of the pair is positive i.e. 1)
            #    So we apply softmax on each pair to get the probability of either positive or negative
            #    Then we get, for every pixel, which INDEX in the softmax is highest
            #    Then, we stack so we get 1 channel for each category
            mask_output = [ F.softmax(mask_output[0, ci*2:(ci+1)*2], dim=0) for ci in range(self.test_dataset.cat_count) ]
            mask_output = [ torch.max(mo, dim=0).indices for mo in mask_output ]
            mask_output = torch.stack(mask_output)

            # Mask the density
            masked_den  = torch.mul(mask_output, output_den[0]).unsqueeze(dim=0)


            input_image = PIL.Image.open(batch["img_path"][0])
            input_image = self.crop_transform(image=np.array(input_image))["image"]
            cv2.imwrite(os.path.join(samples_path, "00__input.png"), cv2.cvtColor(input_image, cv2.COLOR_RGB2BGR))
            
            for ci, cat in enumerate(self.test_dataset.categories):
                # Samples
                cv2.imwrite(
                    os.path.join(samples_path, f"{ci:02d}_{cat}_masked_sum={masked_den[0,ci].sum():.1f}_gt={gt[0,ci].sum():.1f}.png"),
                    cv2.applyColorMap(
                        normalise_for_cmap(masked_den[0].cpu().numpy(), ci).astype(np.uint8),
                        cv2.COLORMAP_VIRIDIS))

                cv2.imwrite(
                    os.path.join(samples_path, f"{ci:02d}_{cat}_sum={output_den[0,ci].sum():.1f}_gt={gt[0,ci].sum():.1f}.png"),
                    cv2.applyColorMap(
                        normalise_for_cmap(output_den[0].cpu().numpy(), ci).astype(np.uint8),
                        cv2.COLORMAP_VIRIDIS))

                cv2.imwrite(
                    os.path.join(samples_path, f"{ci:02d}_{cat}_mask_out.png"),
                    cv2.applyColorMap(
                        normalise_for_cmap(mask_output.cpu().numpy(), ci).astype(np.uint8),
                        cv2.COLORMAP_VIRIDIS))

                cv2.imwrite(
                    os.path.join(samples_path, f"{ci:02d}_{cat}_mask_gt.png"),
                    cv2.applyColorMap(
                        normalise_for_cmap(gt_mask[0].cpu().numpy(), ci).astype(np.uint8),
                        cv2.COLORMAP_VIRIDIS))

                cv2.imwrite(
                    os.path.join(samples_path, f"{ci:02d}_{cat}_den_gt.png"),
                    cv2.applyColorMap(
                        normalise_for_cmap(gt[0].cpu().numpy(), ci).astype(np.uint8),
                        cv2.COLORMAP_VIRIDIS))


                # Metrics
                
                cat_mae_metrics[ci].append(
                    float(abs(gt[0,ci].sum() - output_den[0,ci].sum()).cpu())
                )

                cat_mse_metrics[ci].append(
                    float((gt[0,ci].sum() - output_den[0,ci].sum()).cpu()) ** 2
                )

                cat_masked_mae_metrics[ci].append(
                    float(abs(gt[0,ci].sum() - masked_den[0,ci].sum()).cpu())
                )

                cat_masked_mse_metrics[ci].append(
                    float((gt[0,ci].sum() - masked_den[0,ci].sum()).cpu()) ** 2
                )

                metrics_df.append({
                    "filename"      : filename,
                    "findex"        : i,
                    "cindex"        : ci,
                    "catname"       : cat,
                    "mae"           : cat_mae_metrics[ci][-1],
                    "mse"           : cat_mse_metrics[ci][-1],
                    "masked_mae"    : cat_masked_mae_metrics[ci][-1],
                    "masked_mse"    : cat_masked_mse_metrics[ci][-1],
                    "gt_count"      : float(gt[0,ci].sum()),
                    "out_count"     : float(output_den[0,ci].sum()),
                    "out_count_mask": float(masked_den[0,ci].sum())
                })


            # Get the mean MAE over all cats
            mae_metrics.append(np.array(cat_mae_metrics)[:,-1].mean())
            mse_metrics.append(np.array(cat_mse_metrics)[:,-1].mean())

            masked_mae_metrics.append(np.array(cat_masked_mae_metrics)[:,-1].mean())
            masked_mse_metrics.append(np.array(cat_masked_mse_metrics)[:,-1].mean())
            
            metrics_image_df.append({
                "filename"      : filename,
                "findex"        : i,
                "mae"           : mae_metrics[-1],
                "mse"           : mse_metrics[-1],
                "masked_mae"    : masked_mae_metrics[-1],
                "masked_mse"    : masked_mse_metrics[-1],
                "gt_count"      : round(float(gt[0].sum())),
                "out_count"     : float(output_den[0].sum()),
                "out_count_mask": float(masked_den[0].sum())
            })

            # Update the TQDM instance
            tqdm_testloader.set_description(f"Testing {np.mean(mae_metrics):0.1f} MAE")

        # Get the overall accuracy, and per-category accuracy
        df = pandas.DataFrame(metrics_df)
        df.to_csv(os.path.join(test_save_dir, "metrics_df.csv"))

        df_img = pandas.DataFrame(metrics_image_df)
        df_img.to_csv(os.path.join(test_save_dir, "metrics_img_df.csv"))

        ranged_results = []

        # Recreate the same talbe in Fraunhofer 2022
        for rng in [(0,1000), (0,10), (11,50), (51,100), (101,1000)]:
            rng_min = rng[0]
            rng_max = rng[1]
            
            # Get examples within the range
            df_within = df_img[(df_img.gt_count >= rng_min) & (df_img.gt_count <= rng_max)]
            
            ranged_results.append({
                "range"            : rng,
                "range_str"        : f"[{rng_min}-{rng_max}]",
                "mae"              : df_within["mae"].mean(),
                "mse"              : df_within["mse"].mean(),
                "rmse"             : df_within["mse"].mean() ** 0.5,
                "examples_in_range": df_within.shape[0]
            })

        ranged_results_df = pandas.DataFrame(ranged_results)
        ranged_results_df.to_csv(os.path.join(test_save_dir, f"results_ranged_{state['task_id']}.csv"))

        results = []
        results.append({
            "catname": "Overall Results",
            "catid": 999,
            "mae": np.mean(mae_metrics),
            "mse": np.mean(mse_metrics),
            "rmse": np.mean(mse_metrics)**0.5,
            "masked_mae": np.mean(masked_mae_metrics),
            "masked_mse": np.mean(masked_mse_metrics),
            "masked_rmse": np.mean(masked_mse_metrics)**0.5
        })
    
        for ci, cat in enumerate(self.test_dataset.categories):
            results.append({
                "catname": cat,
                "catid": ci,
                "mae": df[df.cindex == ci]["mae"].mean(),
                "mse": df[df.cindex == ci]["mse"].mean(),
                "rmse": df[df.cindex == ci]["mse"].mean()**0.5,
                "masked_mae": df[df.cindex == ci]["masked_mae"].mean(),
                "masked_mse": df[df.cindex == ci]["masked_mse"].mean(),
                "masked_rmse": df[df.cindex == ci]["masked_mse"].mean()**0.5
            })

        results_df = pandas.DataFrame(results)
        results_df.to_csv(os.path.join(test_save_dir, f"results_{state['task_id']}.csv"))

    def _set_seeds(self, seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)

    def _parse_args(self) -> argparse.Namespace:
        args = argparse.ArgumentParser(
            prog="GtNRTesting",
            description="The testing script for this Twins-based biodiversity monitoring machine learning model",
            epilog="Please cite our paper!"
        )

        args.add_argument(
            "--dataset_path", "-d",
            help="The directory of the dataset to be used for training (and validation)",
            type=lambda d: d if (os.path.isdir(d)) else raise_(NotADirectoryError(d)),
            # default="datasets/visdrone_10c/")
            # default="datasets/VisDrone/")
            default="datasets/hicks_vdlike/")

        args.add_argument(
            "--weights_path_prefix", "-wp",
            help="The directory to save weights into (the task ID is appended to this)",
            type=lambda d: d if (os.path.isdir(d) or (not os.path.exists(d))) else raise_(NotADirectoryError(d)),
            default="weights/")

        args.add_argument(
            "--model_state_testing_directory",
            help="A directory of model states to test",
            type=lambda d: d if (os.path.isdir(d) or (not os.path.exists(d))) else raise_(NotADirectoryError(d)),
            default="state-testing/")

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
            "--weight_decay", "-wd",
            type=float,
            default=0.0001,
            help="Set the weight decay (L2 regularization) for the optimiser")

        return args.parse_args()
