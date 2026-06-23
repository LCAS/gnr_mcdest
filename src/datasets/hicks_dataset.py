import torch
from torch import nn
from torch.utils.data import Dataset

import os
import h5py
import numpy as np
from PIL import Image
from torchvision.transforms import PILToTensor, Resize

class HicksDataset(Dataset):

    categories = ['leucanthemum_vulgare', 'raununculus_spp', 'heracleum_sphondylium', 'silene_dioica-latifolia', 'trifolium_repens', 'cirsium_arvense', 'stachys_sylvatica', 'rubus_fruticosus_agg', 'vicia_cracca', 'yellow_composite', 'angelica_sylvestris', 'achillea_millefolium', 'senecio_jacobaea', 'prunella_vulgaris', 'trifolium_pratense', 'lotus_spp', 'centaurea_nigra', 'vicia_sepium-sativa', 'bellis_perennis', 'symphytum_officinale', 'knautia_arvensis', 'rhinanthus_minor', 'cirsium_vulgare', 'lathyrus_pratensis', 'taraxacum_agg']

    category_importance = [0.27715895629160753, 0.3590297590986135, 0.4080553832876439, 0.49556850788299894, 0.5274664144952425, 0.5437017008511151, 0.6580230849704721, 0.7059430609629932, 0.7246051838143394, 0.7650547513419925, 0.7808368238156487, 0.7808368238156487, 0.8045464439719943, 0.8323475252820213, 0.8706373347692555, 0.9462998650527742, 1.0612708766947, 1.0756317816567393, 1.0874034017022303, 1.209881106003547, 1.3052411931762404, 1.3204184163527082, 1.4169195840362394, 2.0486904294956974, 3.9944315911775394]


    def __init__(self, root_dir, selected_cats = None, gt_scale = 8, pad_to_dims = (), transform=None, drop_empty_images=False):
        """
        Args:
            root_dir (str): Directory with the HDF5 files (create two datasets, one for val and one for train)
            selected_cats (list(str), optional): Select only these categores for the GT (SET TO NONE TO USE DEFAULT)
            gt_scale (int): The ground-truth matrix is n times larger on each axis (so scale it to the inverse of this)
            transform (callable, optional): Optional transform to be applied on a sample (both GT AND the image)
            transform_img (callable, optional): Optional transform to be applied JUST to the image
        """
        self.root_dir          = root_dir
        self.gt_dir            = os.path.join(root_dir, "gt_density_map")
        self.img_dir           = os.path.join(root_dir, "images")
        self.bbox_dir          = os.path.abspath(os.path.join(self.root_dir, "../bbox"))
        self.gt_scale          = gt_scale
        self.gt_scale_inv      = self.gt_scale ** -1
        self.gt_scale_pow2     = self.gt_scale ** 2
        self.gt_scale_sampler  = nn.Upsample(scale_factor=self.gt_scale_inv, mode="bilinear")
        self.drop_empty_images = drop_empty_images

        self.transform         = transform
        self.gt_files          = [os.path.join(self.gt_dir, f) for f in sorted(os.listdir(self.gt_dir)) if f.endswith('.h5')]
        self.img_files         = [os.path.join(self.img_dir, f) for f in sorted(os.listdir(self.img_dir)) if f.endswith('.jpg')]

        # Figure out the mapping, this SHOULD create an error if it's not found
        if (selected_cats != None):
            self.category_mapping = [ self.categories.index(sc) for sc in selected_cats ]
            self.categories       = selected_cats
        else:
            self.category_mapping = None

        self.cat_count        = len(self.categories)

        self.dropped_images    = 0
        # Observes category mapping
        if (drop_empty_images):
            before_image_count = len(self.img_files)
            self.gt_files, self.img_files, self.counts, self.count_cats = self._select_only_relevant_images()

            self.dropped_images = before_image_count - len(self.img_files)

            # Calculate the category importance if we've dropped images
            self.category_importance = self.count_cats.mean(axis=0)
            # Invert so least common classes are more important
            self.category_importance **= -1
            # Normalise so categories with higher representaion contribute less to the loss
            self.category_importance *= self.cat_count / self.category_importance.sum()

        
        # Get just the filename without the extension or path
        self.img_filenames    = [os.path.splitext(os.path.split(f)[1])[0] for f in self.img_files]
        self.gt_filenames     = [os.path.splitext(os.path.split(f)[1])[0] for f in self.gt_files]
        # Find the images with bounding boxes
        self.boxed_files      = [os.path.join(self.bbox_dir, img + ".JPG.boxed.jpg") for img in self.img_filenames]
        self.pil_to_tensor    = PILToTensor()

        for i, img in enumerate(self.boxed_files):
            if (not os.path.isfile(img)):
                raise RuntimeError(f"Boxed image {img} does not exist for image {self.img_filenames[i]}")

        # Check that all GTs have an input
        for i, gtf in enumerate(self.gt_filenames):
            # If the GT has a corresponding input, IN THE CORRECT INDEX
            assert(gtf == self.img_filenames[i])

    def _apply_mapping(self, density):
        if (self.category_mapping == None):
            return np.array(density)
        else:
            den_shape  = density.shape
            target     = np.zeros((len(self.category_mapping), den_shape[1],  den_shape[2] ), dtype=density.dtype)

            for i, mapping in enumerate(self.category_mapping):
                target[i] = density[mapping]
            
            return target


    def _select_only_relevant_images(self):
        new_gts = []
        new_imgs = []
        counts = []
        counts_cat = []
        
        for gt_f, img_f in zip(self.gt_files, self.img_files):
            with h5py.File(gt_f, 'r') as file:
                dmap = self._apply_mapping(file["density_map"])
                
                if (dmap.sum() > 0):
                    new_gts.append(gt_f)
                    new_imgs.append(img_f)
                    counts.append(dmap.sum())
                    counts_cat.append(dmap.sum(axis=(1,2)))

        return new_gts, new_imgs, np.array(counts), np.array(counts_cat)

    def __len__(self):
        return len(self.gt_files)

    def __getitem__(self, idx):
        # Get the corresponding paths
        gt_path  = self.gt_files[idx]
        img_path = self.img_files[idx]
        box_path = self.boxed_files[idx]

        with h5py.File(gt_path, 'r') as file:
            image   = Image.open(img_path)
            image_first_size = image.size
            gt      = self._apply_mapping(file["density_map"])
            gt_mask = self._apply_mapping(file["mask"])
            # The mask here is stored as probabilities, fix that
            gt_mask[gt_mask<=4] = 1
            gt_mask[gt_mask>4]  = 0
            
        
        if self.transform:
            transformed = self.transform(image=np.array(image), density=gt, masks=gt_mask)

            gt      = transformed["density"]
            gt_mask = transformed["masks"]
            image   = transformed["image"]

        # Get a scaling factor to correct for the error in bilinear
        before_scale_gt_sums = gt.sum(dim=(1,2))

        # Apply the scaling to the GTs to match the output of the model
        gt      = self.gt_scale_sampler(gt.unsqueeze(0)).squeeze(0) * self.gt_scale_pow2
        gt_mask = self.gt_scale_sampler(gt_mask.unsqueeze(0)).squeeze(0)

        for ci in range(self.cat_count):
            # Apply the correction factor, just make sure we're not dividing by zero
            if ((before_scale_gt_sums[ci] != 0) and (gt[ci].sum() != 0)): gt[ci] *= before_scale_gt_sums[ci] / gt[ci].sum()

        # Sizes in this dataset
        # Counter({(1024, 768): 3271, (1024, 576): 2382, (1023, 576): 537, (960, 540): 250, (1024, 767): 30, (480, 360): 1})

        return {
            'image'           : image,
            'gt'              : gt,
            'gt_mask'         : gt_mask,
            'cats'            : self.categories,
            'gt_path'         : gt_path,
            'img_path'        : img_path,
            'boxed_img_path'  : box_path,
            'image_first_size': image_first_size
        }
