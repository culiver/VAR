import os.path as osp
import json
import PIL.Image as PImage
from torchvision.datasets.folder import DatasetFolder, IMG_EXTENSIONS
from torchvision.transforms import InterpolationMode, transforms
import torch
import os


def normalize_01_into_pm1(x):  # normalize x from [0, 1] to [-1, 1] by (x*2) - 1
    return x.add(x).add_(-1)


def build_dataset(
    data_path: str, final_reso: int,
    hflip=False, mid_reso=1.125,
    dataset_type="imagenet"
):
    # build augmentations
    mid_reso = round(mid_reso * final_reso)  # first resize to mid_reso, then crop to final_reso
    train_aug, val_aug = [
        transforms.Resize(mid_reso, interpolation=InterpolationMode.LANCZOS), # transforms.Resize: resize the shorter edge to mid_reso
        transforms.RandomCrop((final_reso, final_reso)),
        transforms.ToTensor(), normalize_01_into_pm1,
    ], [
        transforms.Resize(mid_reso, interpolation=InterpolationMode.LANCZOS), # transforms.Resize: resize the shorter edge to mid_reso
        transforms.CenterCrop((final_reso, final_reso)),
        transforms.ToTensor(), normalize_01_into_pm1,
    ]
    if hflip: train_aug.insert(0, transforms.RandomHorizontalFlip())
    train_aug, val_aug = transforms.Compose(train_aug), transforms.Compose(val_aug)
    
    if dataset_type == "imagenet-a":
        # For ImageNetA dataset testing
        return build_imagenet_a_dataset(data_path, val_aug)
    else:
        # build regular ImageNet dataset
        train_set = DatasetFolder(root=osp.join(data_path, 'train'), loader=pil_loader, extensions=IMG_EXTENSIONS, transform=train_aug)
        val_set = DatasetFolder(root=osp.join(data_path, 'val'), loader=pil_loader, extensions=IMG_EXTENSIONS, transform=val_aug)
        num_classes = 1000
        print(f'[Dataset] {len(train_set)=}, {len(val_set)=}, {num_classes=}')
        print_aug(train_aug, '[train]')
        print_aug(val_aug, '[val]')
        
        return num_classes, train_set, val_set


class ImageNetADataset(torch.utils.data.Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.samples = []
        self.targets = []
        self.class_to_idx = {}
        
        # Load the mapping between class IDs and ImageNet classes
        imagenet_map_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                                        "imagenet_class_index.json")
        with open(imagenet_map_path, 'r') as f:
            imagenet_class_map = json.load(f)
        
        # Create a mapping from folder name (like n01498041) to class index (0-999)
        folder_to_idx = {}
        for idx, (_, folder_info) in enumerate(imagenet_class_map.items()):
            wnid = folder_info[0]  # like n01498041
            folder_to_idx[wnid] = int(idx)
        
        # Collect samples from the ImageNetA directory
        subfolders = [d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))]
        for folder in sorted(subfolders):
            if folder in folder_to_idx:  # Check if this folder is in our mapping
                class_idx = folder_to_idx[folder]
                self.class_to_idx[folder] = class_idx
                
                folder_path = os.path.join(root, folder)
                for img_name in os.listdir(folder_path):
                    if any(img_name.lower().endswith(ext) for ext in IMG_EXTENSIONS):
                        img_path = os.path.join(folder_path, img_name)
                        self.samples.append((img_path, class_idx))
                        self.targets.append(class_idx)
                        
        print(f"Found {len(self.samples)} samples in ImageNetA dataset across {len(self.class_to_idx)} classes")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, target = self.samples[idx]
        img = pil_loader(img_path)
        
        if self.transform is not None:
            img = self.transform(img)
            
        return img, target


def build_imagenet_a_dataset(data_path, transform):
    """Build the ImageNetA dataset with proper class mapping to original ImageNet"""
    # ImageNetA only has test samples, no train set
    imagenet_a_set = ImageNetADataset(root=data_path, transform=transform)
    
    # Use empty train set to maintain API compatibility
    empty_train_set = torch.utils.data.TensorDataset(
        torch.tensor([]), torch.tensor([])
    )
    
    # For ImageNet-A, we have 200 classes but they map to specific indices in the 1000 ImageNet classes
    # Return both num_classes=200 and the actual class indices for proper alignment
    num_classes = 200  # ImageNet-A has 200 classes
    # Get the actual class indices used in ImageNet-A
    imagenet_a_class_indices = sorted(list(imagenet_a_set.class_to_idx.values()))
    
    print(f'[Dataset] ImageNetA: {len(imagenet_a_set)=}, {len(imagenet_a_set.class_to_idx)=} unique classes')
    print_aug(transform, '[ImageNetA]')
    
    return num_classes, empty_train_set, imagenet_a_set, imagenet_a_class_indices


def pil_loader(path):
    with open(path, 'rb') as f:
        img: PImage.Image = PImage.open(f).convert('RGB')
    return img


def print_aug(transform, label):
    print(f'Transform {label} = ')
    if hasattr(transform, 'transforms'):
        for t in transform.transforms:
            print(t)
    else:
        print(transform)
    print('---------------------------\n')
