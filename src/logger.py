import clearml
import datetime
import torch
import os
import shutil
import argparse
import math
import matplotlib
import matplotlib.pyplot as plt
import PIL
import numpy as np

from torch.nn import functional as F

from .utils import get_current_iso_datetime, print_box_outline_text, normalise_for_cmap, fig_to_numpy
# from .application import TrainingApplication

class MLLogger:

    def __init__(self, task_name: str, weights_prefix: str) -> None:
        
        self.task_name      = task_name
        self.task_id        = f"{self.task_name}_{get_current_iso_datetime()}"
        self.weights_prefix = weights_prefix
        self.weights_path   = os.path.join(self.weights_prefix, self.task_id)
        
        self.cmltask = clearml.Task.init(
            task_name=self.task_id,
            project_name="flowers")

        # get logger object for current task
        self.cmllogger: clearml.Logger = self.cmltask.get_logger()
        # Increase logger limit
        self.cmllogger.set_default_debug_sample_history(2000);


    def append_args_to_task_params(self, args: argparse.Namespace):
        pass

    def report_train_loss(self, current: float, mean: float, iteration: int, prefix: str = None) -> None:
        if prefix:
            self.cmllogger.report_scalar(title="Train Loss", series=f"{prefix} - current", iteration=iteration, value=current)
            self.cmllogger.report_scalar(title="Train Loss", series=f"{prefix} - mean",    iteration=iteration, value=mean)
        else:
            self.cmllogger.report_scalar(title="Train Loss", series="current", iteration=iteration, value=current)
            self.cmllogger.report_scalar(title="Train Loss", series="mean",    iteration=iteration, value=mean)
        # print(f"MSE: {mse:0.3f}")

    def report_val_metric(self, acc: float, iteration: int, category: str = None) -> None:
        if category:
            self.cmllogger.report_scalar(title="Validation Metric", series=f"{category} - acc", iteration=iteration, value=acc)
        else:
            self.cmllogger.report_scalar(title="Validation Metric", series="acc", iteration=iteration, value=acc)

    def report_val_mask_metric(self, mask_loss: float, iteration: int) -> None:
        self.cmllogger.report_scalar(title="Validation Metric", series="Mask Loss", iteration=iteration, value=mask_loss)

    def report_hyperparams(self, learning_rate: float, iteration: int) -> None:
        self.cmllogger.report_scalar(title="Hyperparameters", series="learning_rate", iteration=iteration, value=learning_rate)

    def save_weights(self, application, best=False) -> None:
        # Creat if not exist
        if (not os.path.isdir(self.weights_path)):
            os.makedirs(self.weights_path)

        if (best): print_box_outline_text("Saving best weights")

        full_filename = f"{'best_' if best else ''}weights_e_{application.epoch}_i_{application.iteration}.pth"
        save_path     = os.path.join(self.weights_path, full_filename)
        
        torch.save({
                "states": application.model.state_dict(),
                "nparams": application.gnr.numparams,
                "task_name": application.task_name,
                "task_id": self.task_id,
                "validation_results": application.validation_results,
                "epoch": application.epoch,
                # "bce_scale": application.mask_loss_scale,
                "init_lr": application.lr,
                "current_lr": application.optimiser.param_groups[0]["lr"],
                "iteration": application.iteration,
                "batch_size": application.batch_size,
                "gpu_metrics_pwr_lastreport": application.gpu_metrics.pwr_lastreport,
                "gpu_metrics_pwr_kwh": application.gpu_metrics.pwr_kwh,
                "weighted_l2_loss": application.use_l2_weighting
            },
            save_path)

        # Upload to ClearML
        if (((application.epoch % application.upload_model_freq) == 0) or best):
            self.cmltask.update_output_model(
                model_path=save_path,
                name=full_filename,
                comment=f"val: {application.validation_results[-1]}, cats: {application.val_dataset.categories}",
                iteration=application.iteration,
                auto_delete_file=False)


    def report_debug_samples(self, application, batch, density_output, mask_output):
        density_output = density_output.cpu().numpy()
        categories     = application.val_dataset.categories
        category_count = len(categories)
        
        # Take in the logits for each pair of cat/nocat predictions, and produce the masks
        # What we do here:
        #    For every category there is a positive and negative case (the SECOND channel of the pair is positive i.e. 1)
        #    So we apply softmax on each pair to get the probability of either positive or negative
        #    Then we get, for every pixel, which INDEX in the softmax is highest
        #    Then, we stack so we get 1 channel for each category
        mask_output = [ F.softmax(mask_output[0, ci*2:(ci+1)*2], dim=0) for ci in range(category_count) ]
        mask_output = [ torch.max(mo, dim=0).indices for mo in mask_output ]
        mask_output = torch.stack(mask_output).cpu().numpy()

        
        # Make some figures
        plot_square_size = ((category_count*2)+2)**0.5
        plot_shape       = (math.ceil(plot_square_size*(1.5**-1)), math.ceil(plot_square_size*1.5))
        fig_outcount     = plt.figure(figsize=(14*1.5,14*(1.5**-1)), dpi=(400.0 if application.val_dataset.cat_count >= 10 else 250.0))
        fig_outmask      = plt.figure(figsize=(14*1.5,14*(1.5**-1)), dpi=(400.0 if application.val_dataset.cat_count >= 10 else 250.0))

        # Grab the image used as input
        # If we have boxed images, use them
        if ("boxed_img_path" in batch.keys()):
            input_image = PIL.Image.open(batch["boxed_img_path"][0])
            # We need to rezise it to whatever out dataset generation script spat out
            input_image = input_image.resize(PIL.Image.open(batch["img_path"][0]).size, PIL.Image.Resampling.BILINEAR)
        else:
            input_image = PIL.Image.open(batch["img_path"][0])
            
        # Apply the crop transformation
        input_image = application.crop_transform(image=np.array(input_image))["image"]

        # Report the input images
        ax = fig_outcount.add_subplot(plot_shape[0],plot_shape[1],1)
        ax.imshow(input_image)
        ax.set_title("Input")


        ax = fig_outmask.add_subplot(plot_shape[0],plot_shape[1],1)
        ax.imshow(input_image)
        ax.set_title("Input")

        for cid, cat in enumerate(categories):
            gt_index  = (cid*2)+2
            out_index = (cid*2)+3

            # Report density GTs
            ax = fig_outcount.add_subplot(plot_shape[0],plot_shape[1], gt_index)
            heatmap = batch["gt"][0].cpu().numpy()
            ax.imshow(normalise_for_cmap(heatmap, cid))
            ax.set_title(f"GT {cat:10.10} {np.sum(heatmap[cid]):.2f}")

            # Report density outs
            ax = fig_outcount.add_subplot(plot_shape[0],plot_shape[1], out_index)
            heatmap = density_output
            ax.imshow(normalise_for_cmap(heatmap[0], cid))
            ax.set_title(f"{cat:10.10} {np.sum(heatmap[0,cid]):.2f}")

            # Report mask GTs
            ax = fig_outmask.add_subplot(plot_shape[0],plot_shape[1], gt_index)
            heatmap = batch["gt_mask"][0].cpu().numpy()
            ax.imshow(normalise_for_cmap(heatmap, cid))
            ax.set_title(f"GT {cat:10.10} {np.sum(heatmap[cid]):.1f}")

            # Report mask outs
            ax = fig_outmask.add_subplot(plot_shape[0],plot_shape[1], out_index)
            heatmap = mask_output
            ax.imshow(normalise_for_cmap(heatmap, cid))
            ax.set_title(f"{cat:10.10} {np.sum(heatmap[cid]):.1f}")


        # Make the plots look better
        fig_outcount.colorbar(matplotlib.cm.ScalarMappable(
            norm=matplotlib.colors.Normalize(vmin=np.min(density_output), vmax=np.max(density_output))),
            cax=fig_outcount.add_subplot(plot_shape[0],plot_shape[1], np.prod(plot_shape)),
            shrink=0.8)
        fig_outcount.tight_layout()

        # Rasterise and send
        figimg_outcount = fig_to_numpy(fig_outcount)
        figimg_outmask  = fig_to_numpy(fig_outmask)

        self.cmllogger.report_image(title="Density Outputs", series=f"{batch['img_path'][0]}", iteration=application.iteration, image=figimg_outcount)
        self.cmllogger.report_image(title="Mask Outputs",    series=f"{batch['img_path'][0]}", iteration=application.iteration, image=figimg_outmask)

        plt.clf()
        plt.close("all")
