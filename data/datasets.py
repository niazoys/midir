import os
import os.path as path
import random

import numpy as np
import torch
import torch.utils.data as ptdata

from data.synthesis import synthesis_elastic_deformation, GaussianFilter
from utils.image import normalise_intensity, crop_and_pad, bbox_from_mask, bbox_crop
from utils.image_io import load_nifti

"""
Data object:
- Construct Datasets and Dataloaders
- Standardize data interface
"""

def worker_init_fn(worker_id):
    """Callback function passed to DataLoader to initialise the workers"""
    # # generate a random sequence of seeds for the workers
    # print(f"Random state before generating the random seed: {random.getstate()}")
    random_seed = random.randint(0, 2 ** 32 - 1)
    # ##debug
    # print(f"Random state after generating the random seed: {random.getstate()}")
    # print(f"Random seed for worker {worker_id} is: {random_seed}")
    # ##
    np.random.seed(random_seed)


class BrainData(object):
    def __init__(self, args, params):
        self.args = args
        self.params = params

        # training data
        train_data_path = self.params.train_data_path
        assert path.exists(train_data_path), f"Training data path does not exist: \n{train_data_path}, not generated?"

        if self.params.dim == 3:
            # pre-generated training data
            self.train_dataset = BrainDataset(train_data_path,
                                              run="train",
                                              dim=self.params.dim)
        else:
            # synthesis on-the-fly training data
            self.train_dataset = BrainSynthDataset(self.params.dataset_name,
                                                   train_data_path,
                                                   dim=self.params.dim,
                                                   run="train",
                                                   cps=self.params.synthesis_cps,
                                                   sigma=self.params.synthesis_sigma,
                                                   disp_max=self.params.disp_max,
                                                   crop_size=self.params.crop_size,
                                                   slice_range=tuple(self.params.slice_range))

        self.train_dataloader = ptdata.DataLoader(self.train_dataset,
                                                  batch_size=self.params.batch_size,
                                                  shuffle=True,
                                                  num_workers=self.args.num_workers,
                                                  pin_memory=self.args.cuda,
                                                  worker_init_fn=worker_init_fn  # todo: fix random seeding
                                                  )


        # validation data
        val_data_path = self.params.val_data_path
        assert path.exists(val_data_path), f"Validation data path does not exist: \n{val_data_path}, not generated?"

        self.val_dataset = BrainDataset(val_data_path,
                                        run="val",
                                        dim=self.params.dim)

        self.val_dataloader = ptdata.DataLoader(self.val_dataset,
                                                batch_size=1,
                                                shuffle=False,
                                                num_workers=self.args.num_workers,
                                                pin_memory=self.args.cuda
                                                )


        # testing data
        test_data_path = self.params.test_data_path
        assert path.exists(test_data_path), f"Testing data path does not exist: \n{test_data_path}, not generated?"

        self.test_dataset = BrainDataset(test_data_path,
                                         run="test",
                                         dim=self.params.dim)

        self.test_dataloader = ptdata.DataLoader(self.test_dataset,
                                                 batch_size=1,
                                                 shuffle=False,
                                                 num_workers=self.args.num_workers,
                                                 pin_memory=self.args.cuda
                                                 )



"""
Datasets
"""

class BrainSynthDataset(ptdata.Dataset):
    """
    Loading, processing and synthesising transformation
    for training (on-the-fly) and for generating val/test data
    """
    def __init__(self,
                 dataset_name,
                 data_path,
                 run,
                 dim,
                 sigma=8,
                 cps=10,
                 disp_max=1.,
                 crop_size=192,
                 slice_range=(70, 90),
                 device=torch.device('cpu')
                 ):
        super(BrainSynthDataset, self).__init__()

        self.dataset_name = dataset_name
        self.data_path = data_path
        self.subject_list = sorted(os.listdir(self.data_path))

        self.run = run
        self.dim = dim
        self.crop_size = crop_size
        self.slice_range = slice_range
        self.device = device

        # elastic parameters
        self.sigma = sigma
        self.cps = cps
        self.disp_max = disp_max

        # Gaussian smoothing filter for random transformation generation
        self.smooth_filter = GaussianFilter(dim=self.dim, sigma=self.sigma)

    def __getitem__(self, index):
        data_dict = {}
        subject_id = self.subject_list[index]

        # specify path to target, source and roi mask (only data-specific part in the pipeline)
        if self.dataset_name == "Brats17":
            target_original_path = f"{self.data_path}/{subject_id}/{subject_id}_t1.nii.gz"
            source_path = f"{self.data_path}/{subject_id}/{subject_id}_t2.nii.gz"
            roi_mask_path = f"{self.data_path}/{subject_id}/{subject_id}_brainmask.nii.gz"

        elif self.dataset_name == "IXI":
            target_original_path = f"{self.data_path}/{subject_id}/T1-brain.nii.gz"
            source_path = f"{self.data_path}/{subject_id}/T2-brain.nii.gz"
            roi_mask_path = f"{self.data_path}/{subject_id}/T1-brain_mask.nii.gz"

        else:
            raise RuntimeError("Dataset name not recognised")


        """ 
        Load T1 &/ T2 image and brain ROI mask, shape (N, *sizes) 
        """
        # 2D axial slices, data shape (N=#slices, H, W)
        if self.dim == 2:
            data_dict["target_original"] = load_nifti(target_original_path).transpose(2, 0, 1)
            data_dict["source"] = load_nifti(source_path).transpose(2, 0, 1)
            data_dict["roi_mask"] = load_nifti(roi_mask_path).transpose(2, 0, 1)

            # slice selection
            if self.run == "train":
                # randomly select a slice within range
                z = random.randint(self.slice_range[0], self.slice_range[1])
                slicer = slice(z, z+1)  # keep dim
            else: # generate
                # take all slices within range
                slicer = slice(self.slice_range[0], self.slice_range[1])

            for name, data in data_dict.items():
                data_dict[name] = data[slicer, ...]  # (N/1, H, W)

        # 3D volumes, extend shape to (N=1, H, W, D)
        else:
            data_dict["target_original"] = load_nifti(target_original_path)[np.newaxis, ...]
            data_dict["source"] = load_nifti(source_path)[np.newaxis, ...]
            data_dict["roi_mask"] = load_nifti(roi_mask_path)[np.newaxis, ...]
        """"""

        """ Intensity normalisation, cropping, synthesize deformation """
        # crop by brain mask bounding box for IXI dataset
        if self.dataset_name == "IXI":
            bbox, _ = bbox_from_mask(data_dict["roi_mask"], pad_ratio=0.0)
            for name, data in data_dict.items():
                data_dict[name] = bbox_crop(data[:, np.newaxis, ...], bbox)[:, 0, ...]


        # cropping
        for name in ["target_original", "source", "roi_mask"]:
            data_dict[name] = crop_and_pad(data_dict[name], new_size=self.crop_size)

        # intensity normalisation to [0, 1]
        for name in ["target_original", "source"]:
            data_dict[name] = normalise_intensity(data_dict[name],
                                                  min_out=0., max_out=1.,
                                                  mode="minmax", clip=True)

        # generate synthesised DVF and deformed T1 image
        data_dict["target"], data_dict["dvf_gt"] = synthesis_elastic_deformation(data_dict["target_original"],
                                                                                 data_dict["roi_mask"],
                                                                                 smooth_filter=self.smooth_filter,
                                                                                 cps=self.cps,
                                                                                 disp_max=self.disp_max,
                                                                                 device=self.device)

        # cast to Pytorch Tensor
        for name, data in data_dict.items():
            data_dict[name] = torch.from_numpy(data).float()
        return data_dict

    def __len__(self):
        return len(self.subject_list)



class BrainDataset(ptdata.Dataset):
    """
    Loading pre-generated synthesised training data
    or validation and testing data
    (normalised & cropped)
    """
    def __init__(self, data_path, run, dim):
        """data_path should contain the subject directories"""
        super(BrainDataset, self).__init__()
        self.data_path = data_path
        self.subject_list = sorted(os.listdir(self.data_path))
        self.run = run  # "train" or else ("val"/"test")
        self.dim = dim

    def __getitem__(self, index):
        """Shape of output data in data_dict : images (N, *size), DVF (N, dim, *size)"""
        data_dict = {}
        subject_id = self.subject_list[index]

        for name in ["target", "source", "target_original", "roi_mask", "dvf_gt"]:
            data_path = f"{self.data_path}/{subject_id}/{name}.nii.gz"
            if name == "dvf_gt":
                # skip loading ground truth DVF for training data
                if self.run == "train":
                    continue

                if self.dim == 2:  # 2D
                    # dvf is saved in shape (H, W, N, 2) -> (N, 2, H, W)
                    data_dict[name] = load_nifti(data_path).transpose(2, 3, 0, 1)
                else:  # 3D
                    # dvf is saved in shape (H, W, D, 3) -> (N=1, 3, H, W, D)
                    data_dict[name] = load_nifti(data_path).transpose(3, 0, 1, 2)[np.newaxis, ...]

            else:
                if self.dim == 2:  # 2D
                    # image is saved in shape (H, W, N) ->  (N, H, W)
                    data_dict[name] = load_nifti(data_path).transpose(2, 0, 1)
                else:  # 3D
                    # image is saved in shape (H, W, D) -> (N=1, H, W, D)
                    data_dict[name] = load_nifti(data_path)[np.newaxis, ...]

        # cast to Pytorch Tensor
        for name, data in data_dict.items():
            data_dict[name] = torch.from_numpy(data).float()

        return data_dict

    def __len__(self):
        return len(self.subject_list)

