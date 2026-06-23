from enum import Enum
import torch
import torch.nn as nn
import types as T
import collections
import timm
import timm.models
from timm.models import Twins
import numpy as np

from .model_components.mam import MAM
from .model_components.cfm import CFM

from .utils import print_box_outline_text


class ModelBackbone(Enum):
    """
    The available backbones (encoders) to be used with this model
    """

    # AKA Twins ALTGVT-Small
    TWINS_SVT_SMALL = 1
    # AKA Twins ALTGVT-Base
    TWINS_SVT_BASE  = 2
    # AKA Twins ALTGVT-Large
    TWINS_SVT_LARGE = 3



class GtNR(nn.Module):
    """
    The crowd counting architecture implemented as a PyTorch module
    """

    def __init__(self,
        backbone: ModelBackbone          = ModelBackbone.TWINS_SVT_BASE,
        pretrained_path: str             = "pretrained_weights/upernet_alt_gvt_b_512x512_160k_ade20k_swin_setting.pth",
        pretrained_is_imagenet: bool     = False,
        categories: int                  = 1
    ):
        """
        Configure the backbone, initialise weights and load weights
        """

        # PyTorch module init
        super().__init__()

        # Parameters
        self.categories = categories
        
        # Setup the backbone
        match backbone:
            case ModelBackbone.TWINS_SVT_SMALL:
                # This is useful information
                self.encoder_channels = [64, 128, 256, 512]
                
                self.backbone = Twins(
                    patch_size=4, embed_dims=self.encoder_channels, num_heads=[2, 4, 8, 16], mlp_ratios=[4, 4, 4, 4], depths=[2, 2, 10, 4], wss=[7, 7, 7, 7], sr_ratios=[8, 4, 2, 1]
                )

            case ModelBackbone.TWINS_SVT_BASE:
                # This is useful information
                self.encoder_channels = [96, 192, 384, 768]
                
                self.backbone = Twins(
                    patch_size=4, embed_dims=self.encoder_channels, num_heads=[3, 6, 12, 24], mlp_ratios=[4, 4, 4, 4], depths=[2, 2, 18, 2], wss=[7, 7, 7, 7], sr_ratios=[8, 4, 2, 1]
                )
                
            case ModelBackbone.TWINS_SVT_LARGE:
                # This is useful information
                self.encoder_channels = [128, 256, 512, 1024]
                
                self.backbone = Twins(
                    patch_size=4, embed_dims=self.encoder_channels, num_heads=[4, 8, 16, 32], mlp_ratios=[4, 4, 4, 4], depths=[2, 2, 18, 2], wss=[7, 7, 7, 7], sr_ratios=[8, 4, 2, 1]
                )

            case _:
                raise Exception(f"Backbone must be one of the items in {[value for name,value in ModelBackbone.__members__.items()]}")

        # Create out counting head
        self._create_head()

        # self.apply(self._initialize_weights)
        # Initialise only the backbone's weights using their function before we override them, in case there are missing weights in the pretrained pth
        # We let pytorch decide how our weights should be initalised
        self.backbone.apply(self.backbone._init_weights)

        # If a pretrained path is specified, load it
        if (pretrained_path):
            # Load the weights from the segmentation problem, and select only the backbone weights
            if (not pretrained_is_imagenet):
                backbone_weights = self._process_backbone_weights_dict(
                    torch.load(pretrained_path, weights_only=True)["state_dict"]
                );
            else:
                backbone_weights = torch.load(pretrained_path, weights_only=True)

            # Load the backbone weights
            # Strict false means we don't need all the weights
            # Assign means we absolutely do want to overrride the weights we already have
            missing_keys, unexpected_keys = self.backbone.load_state_dict(backbone_weights, strict=False, assign=True)
            
            print_box_outline_text(f"Loaded backbone weights with the following missing keys:\n{missing_keys}\nFound the following unexpected keys:\n{unexpected_keys}")


    @property
    @torch.jit.ignore
    def numparams(self):
        return int(np.sum([np.prod(p.size()) for p in self.parameters()]))

    @torch.jit.ignore
    def _process_backbone_weights_dict(self, full_state_dict: collections.OrderedDict):
        """
        Extract only items starting with `backbone.` (and remove that), so we can load weights for our backbone

        Based on https://stackoverflow.com/a/70988261
        """
        
        processed_dict = collections.OrderedDict()

        for k in full_state_dict.keys():
            decomposed_key = k.split(".")
            # Only extract backbone states
            if (decomposed_key[0] == "backbone"):
                # Remove the backbone part and copy the old data
                processed_dict[".".join(decomposed_key[1:])] = full_state_dict[k]

        return processed_dict

    def _initialize_weights(self, m):
        """Taken from VGG"""
        
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0, 0.01)
            nn.init.constant_(m.bias, 0)

    def _create_head(self):
        # MAM for layers 2-4 of Twins
        self.mam = MAM(channels=self.encoder_channels[1])
        self.cfm = CFM(channels=self.encoder_channels[1])

        # We combine the last three levels into the MAM using element-wise add, so they need to be of the same dimensions and depth
        self.l3_upsampleconv = nn.Sequential(
            nn.Upsample(scale_factor=4.0, mode="bilinear"),
            nn.Conv2d(in_channels=self.encoder_channels[3], out_channels=self.encoder_channels[1], kernel_size=1, padding="same")
        )
        self.l2_upsampleconv = nn.Sequential(
            nn.Upsample(scale_factor=2.0, mode="bilinear"),
            nn.Conv2d(in_channels=self.encoder_channels[2], out_channels=self.encoder_channels[1], kernel_size=1, padding="same")
        )

        # The final output conv operates on the MAM after 2nd layer of Twins and the results of l3 and l4 MAMs after the upscaling and conv
        # We output 1 channel for 1 count map
        self.output_layer = nn.Sequential(
            nn.Upsample(scale_factor=2.0),
            nn.Conv2d(in_channels=self.encoder_channels[1], out_channels=self.encoder_channels[1]//2, kernel_size=3, padding="same"),
            nn.BatchNorm2d(num_features=self.encoder_channels[1]//2),
            nn.Conv2d(in_channels=self.encoder_channels[1]//2, out_channels=self.encoder_channels[1]//2, kernel_size=3, padding="same"),
            nn.Softplus(),
            nn.Conv2d(in_channels=self.encoder_channels[1]//2, out_channels=self.categories, kernel_size=1, padding="same"),
            nn.Softplus()
        )

        # The mask layer, to be applied on concated last 3 layers
        self.mask_output_layer = nn.Sequential(
            nn.Upsample(scale_factor=2.0),
            nn.Conv2d(in_channels=self.encoder_channels[1], out_channels=self.encoder_channels[1]//2, kernel_size=3, padding="same"),
            nn.Sigmoid(),
            # Output logits for BCE
            nn.Conv2d(in_channels=self.encoder_channels[1]//2, out_channels=self.categories*2, kernel_size=1, padding="same")
        )
    
    def forward_head(self, l1: torch.Tensor, l2: torch.Tensor, l3: torch.Tensor) -> torch.Tensor:
        # Get the different levels to the same dimensions
        f3 = self.l3_upsampleconv(l3)
        f2 = self.l2_upsampleconv(l2)

        # Elementwise addition to combine the three layers
        f1 = torch.sum(torch.stack([l1, f2, f3]), dim=0)

        # Multi-scale stacking
        denpre  = self.mam(f1)
        maskpre = self.cfm(f1)

        # Output the count map
        return self.output_layer(denpre), self.mask_output_layer(maskpre)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for our network, applying our heads on Twins.

        Model outputs a tensor [b, 1, h/4, w/4]
        """

        # Get the intermediates from the backbone
        # Returns a list of tensors
        stages: list[torch.Tensor] = self.backbone.forward_intermediates(
            x                  = x,
            indices            = range(4),  # The first 4 layers (the backbone)
            norm               = False,     # Don't apply norm layer to all intermediates
            stop_early         = True,      # We only need to compute the layers we care about
            output_fmt         = "NCHW",    # Output shape for Twins must be NCHW
            intermediates_only = True       # We only care about intermediate features
        )
        
        # Output the head
        return self.forward_head(l1 = stages[1], l2 = stages[2], l3 = stages[3])
