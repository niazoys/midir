import os
import random
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
from data.utils import _load3d, _crop_and_pad, _normalise_intensity, _to_tensor, _load2d, _magic_slicer


class _BaseDataset(Dataset):
    """Base dataset class"""
    def __init__(self, data_dir_path):
        super(_BaseDataset, self).__init__()
        self.data_dir = data_dir_path
        assert os.path.exists(data_dir_path), f"Data dir does not exist: {data_dir_path}"
        self.subject_list = sorted(os.listdir(self.data_dir))
        self.data_path_dict = dict()

    def _set_path(self, index):
        """ Set the paths of data files to load and the keys in data_dict"""
        raise NotImplementedError

    def __getitem__(self, index):
        """ Load data and pre-process """
        raise NotImplementedError

    def __len__(self):
        return len(self.subject_list)


class BrainMRInterSubj3D(_BaseDataset):
    def __init__(self,
                 data_dir_path,
                 crop_size,
                 evaluate=False,
                 modality='t1t1',
                 atlas_path=None):
        super(BrainMRInterSubj3D, self).__init__(data_dir_path)
        self.evaluate = evaluate
        self.crop_size = crop_size
        self.img_keys = ['target', 'source']

        self.modality = modality
        self.atlas_path = atlas_path

    def _set_path(self, index):
        # choose the target and source subjects/paths
        if self.atlas_path is None:
            self.tar_subj_id = self.subject_list[index]
            self.tar_subj_path = f'{self.data_dir}/{self.tar_subj_id}'
        else:
            self.tar_subj_path = self.atlas_path

        self.src_subj_id = random.choice(self.subject_list)
        self.src_subj_path = f'{self.data_dir}/{self.src_subj_id}'

        self.data_path_dict['target'] = f'{self.tar_subj_path}/T1_brain.nii.gz'

        # modality
        if self.modality == 't1t1':
            self.data_path_dict['source'] = f'{self.src_subj_path}/T1_brain.nii.gz'
        elif self.modality == 't1t2':
            self.data_path_dict['source'] = f'{self.src_subj_path}/T2_brain.nii.gz'
        else:
            raise ValueError(f'Modality ({self.modality}) not recognised.')

        # eval data
        if self.evaluate:
            # T1w image of source subject for visualisation
            self.data_path_dict['target_original'] = f'{self.src_subj_path}/T1_brain.nii.gz'
            self.img_keys.append('target_original')
            # segmentation
            self.data_path_dict['target_seg'] = f'{self.tar_subj_path}/T1_brain_MALPEM_tissues.nii.gz'
            self.data_path_dict['source_seg'] = f'{self.src_subj_path}/T1_brain_MALPEM_tissues.nii.gz'

    def __getitem__(self, index):
        self._set_path(index)
        data_dict = _load3d(self.data_path_dict)
        data_dict = _crop_and_pad(data_dict, self.crop_size)
        data_dict = _normalise_intensity(data_dict, self.img_keys)
        return _to_tensor(data_dict)


class CardiacMR2D(_BaseDataset):
    def __init__(self,
                 data_dir_path,
                 evaluate=False,
                 slice_range=None,
                 slicing=None,
                 crop_size=(192, 192),
                 batch_size=None,
                 ):
        super(CardiacMR2D, self).__init__(data_dir_path)
        self.evaluate = evaluate
        self.crop_size = crop_size
        self.img_keys = ['target', 'source']

        self.slice_range = slice_range
        self.slicing = slicing
        if batch_size is not None:
            self.subject_list = self.subject_list * batch_size

    def _set_path(self, index):
        self.subj_id = self.subject_list[index]
        self.subj_path = f'{self.data_dir}/{self.subj_id}'
        self.data_path_dict['target'] = f'{self.subj_path}/sa_ED.nii.gz'
        self.data_path_dict['source'] = f'{self.subj_path}/sa_ES.nii.gz'
        if self.evaluate:
            self.data_path_dict['target_original'] = self.data_path_dict['source']
            self.img_keys.append('target_original')
            self.data_path_dict['target_seg'] = f'{self.subj_path}/label_sa_ED.nii.gz'
            self.data_path_dict['source_seg'] = f'{self.subj_path}/label_sa_ES.nii.gz'

    def __getitem__(self, index):
        self._set_path(index)
        data_dict = _load2d(self.data_path_dict)
        data_dict = _magic_slicer(data_dict, slice_range=self.slice_range, slicing=self.slicing)
        data_dict = _crop_and_pad(data_dict, self.crop_size)
        data_dict = _normalise_intensity(data_dict, self.img_keys)
        return _to_tensor(data_dict)
    


class LumireDataset(Dataset):
    def __init__(self,
                 data_dir_path,
                 evaluate=False,
                 slice_range=None,
                 slicing=None,
                 crop_size=(256, 256),
                 ):
        """
        Initialize the LumireDataset.

        Args:
            data_dir_path (str): Path to the directory containing all samples.
            evaluate (bool): Flag to indicate evaluation mode.
            slice_range (tuple, optional): Range for slicing the data.
            slicing (callable, optional): Slicing function.
            crop_size (tuple): Desired crop size for images.
        """
        super(LumireDataset, self).__init__()
        self.evaluate = evaluate
        self.crop_size = crop_size
        self.img_keys = ['target', 'source']
        
        # Load the list of sample paths
        # Ensure that 'all_samples_full.pt' contains a list of file paths
        self.all_samples = torch.load(os.path.join(data_dir_path, 'all_samples_full.pt'))
        
        if not isinstance(self.all_samples, list):
            raise ValueError("all_samples_full.pt should contain a list of file paths.")
        
        self.pairs = []
        self.shuffle_pairs()  # Initialize pairs

    def shuffle_pairs(self):
        """
        Shuffle the samples and create pairs for the current epoch.
        """
        shuffled = self.all_samples.copy()
        np.random.shuffle(shuffled)
        
        # If the number of samples is odd, drop the last sample
        if len(shuffled) % 2 != 0:
            print("Odd number of samples, dropping the last one to form pairs.")
            shuffled = shuffled[:-1]
        
        self.pairs = [(shuffled[i], shuffled[i + 1]) for i in range(0, len(shuffled), 2)]

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        """
        Retrieve a pair of images and process them.

        Args:
            index (int): Index of the pair.

        Returns:
            dict: Dictionary containing 'target' and 'source' tensors.
        """
        target_pth, source_pth = self.pairs[index]
        
        # Load and convert images to grayscale
        target_image = np.asarray(Image.open(target_pth).convert('L'))
        source_image = np.asarray(Image.open(source_pth).convert('L'))
        

        data_dict = {'target': target_image, 'source': source_image}
        
        # Normalize and convert to tensors
        data_dict = _normalise_intensity(data_dict, self.img_keys)
        data_dict = _to_tensor(data_dict)
        
        return data_dict


    def on_epoch_start(self):
        """
        Reshuffle pairs at the start of each epoch.
        """
        self.shuffle_pairs()


