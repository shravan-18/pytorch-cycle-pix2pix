import os
from data.base_dataset import BaseDataset, get_params, get_transform
from data.image_folder import make_dataset
from PIL import Image

class AlignedDataset(BaseDataset):
    """A dataset class for paired images in separate directories A and B."""

    def __init__(self, opt):
        """Initialize this dataset class.
        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.dir_A = os.path.join(opt.dataroot, opt.phase, 'A')  # get the image directory for domain A
        self.dir_B = os.path.join(opt.dataroot, opt.phase, 'B')  # get the image directory for domain B
        
        self.A_paths = sorted(make_dataset(self.dir_A, opt.max_dataset_size))  # get image paths for A
        self.B_paths = sorted(make_dataset(self.dir_B, opt.max_dataset_size))  # get image paths for B
        
        # Ensure the number of images in both directories matches
        assert len(self.A_paths) == len(self.B_paths), "Number of images in A and B directories must match"
        
        # Verify that the filenames match (optional, but helps catch errors)
        for a_path, b_path in zip(self.A_paths, self.B_paths):
            a_name = os.path.basename(a_path)
            b_name = os.path.basename(b_path)
            assert a_name == b_name, f"Filename mismatch: {a_name} != {b_name}"
        
        self.input_nc = self.opt.output_nc if self.opt.direction == 'BtoA' else self.opt.input_nc
        self.output_nc = self.opt.input_nc if self.opt.direction == 'BtoA' else self.opt.output_nc

    def __getitem__(self, index):
        """Return a data point and its metadata information.
        Parameters:
            index - - a random integer for data indexing
        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor) - - an image in the input domain
            B (tensor) - - its corresponding image in the target domain
            A_paths (str) - - image paths
            B_paths (str) - - image paths
        """
        # Read images from separate paths
        A_path = self.A_paths[index]
        B_path = self.B_paths[index]
        
        A_img = Image.open(A_path).convert('RGB')
        B_img = Image.open(B_path).convert('RGB')
        
        # Apply the same transform to both A and B
        transform_params = get_params(self.opt, A_img.size)
        A_transform = get_transform(self.opt, transform_params, grayscale=(self.input_nc == 1))
        B_transform = get_transform(self.opt, transform_params, grayscale=(self.output_nc == 1))
        
        A = A_transform(A_img)
        B = B_transform(B_img)
        
        return {'A': A, 'B': B, 'A_paths': A_path, 'B_paths': B_path}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.A_paths)    
