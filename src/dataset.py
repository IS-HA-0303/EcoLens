import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from pathlib import Path
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
from collections import Counter

# ── Image size we will resize everything to ──────────────────
IMG_SIZE = 224  # EfficientNet-B4 default input size

# ── Albumentations pipeline for TRAINING images ──────────────
# Applied randomly each time an image is loaded
# This artificially increases dataset diversity
train_transforms = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),
    A.HorizontalFlip(p=0.5),              # 50% chance mirror image
    A.VerticalFlip(p=0.1),               # 10% chance flip upside down
    A.Rotate(limit=15, p=0.4),           # random rotation up to 15 degrees
    A.RandomBrightnessContrast(          # random brightness and contrast
        brightness_limit=0.2,
        contrast_limit=0.2,
        p=0.4
    ),
    A.HueSaturationValue(               # random color jitter
        hue_shift_limit=10,
        sat_shift_limit=20,
        val_shift_limit=10,
        p=0.3
    ),
    A.GaussianBlur(blur_limit=3, p=0.2), # slight blur occasionally
    A.CoarseDropout(                     # randomly black out small patches
        max_holes=8,
        max_height=16,
        max_width=16,
        p=0.2
    ),
    A.Normalize(                         # normalize using ImageNet stats
        mean=[0.485, 0.456, 0.406],      # because EfficientNet was pretrained
        std=[0.229, 0.224, 0.225]        # on ImageNet with these values
    ),
    ToTensorV2()                         # convert numpy array → PyTorch tensor
])

# ── Albumentations pipeline for VALIDATION and TEST images ───
# No augmentation — only resize and normalize
# We want to evaluate on real unmodified images
val_transforms = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),
    A.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
    ToTensorV2()
])


# ── Custom PyTorch Dataset class ─────────────────────────────
class WasteDataset(Dataset):
    """
    Custom Dataset that:
    1. Reads images from train/ val/ test/ folder structure
    2. Applies Albumentations transforms
    3. Returns (image_tensor, label_integer) pairs
    """

    def __init__(self, data_dir, class_to_idx, transform=None):
        """
        data_dir     : Path to train/ or val/ or test/ folder
        class_to_idx : dict mapping class name → integer label
        transform    : Albumentations transform pipeline
        """
        self.data_dir     = Path(data_dir)
        self.transform    = transform
        self.class_to_idx = class_to_idx
        self.valid_ext    = {'.jpg','.jpeg','.png','.JPG','.JPEG','.PNG'}

        # Build list of (image_path, label) pairs
        self.samples = []
        for class_folder in sorted(self.data_dir.iterdir()):
            if not class_folder.is_dir():
                continue
            if class_folder.name not in class_to_idx:
                continue
            label = class_to_idx[class_folder.name]
            for img_path in class_folder.iterdir():
                if img_path.suffix in self.valid_ext:
                    self.samples.append((img_path, label))

        print(f"Dataset loaded: {len(self.samples)} images "
              f"from {self.data_dir.name}/")

    def __len__(self):
        # PyTorch calls this to know how many samples exist
        return len(self.samples)

    def __getitem__(self, idx):
        # PyTorch calls this to get one sample by index
        img_path, label = self.samples[idx]

        # Open image and convert to RGB
        # (some images can be RGBA or grayscale — RGB ensures 3 channels)
        image = Image.open(str(img_path)).convert("RGB")
        image = np.array(image)  # Albumentations needs numpy array

        # Apply transforms
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]  # now a PyTorch tensor

        return image, label

    def get_labels(self):
        # Returns list of all labels — needed for WeightedRandomSampler
        return [label for _, label in self.samples]


# ── Weighted Random Sampler ───────────────────────────────────
def make_weighted_sampler(dataset):
    """
    Creates a WeightedRandomSampler that oversamples minority classes.
    This means small classes (like hazardous with 945 images) get
    sampled more often so the model sees them as frequently as
    large classes (like non_recyclable_trash with 3834 images).
    """
    labels      = dataset.get_labels()
    class_counts = Counter(labels)
    n_classes   = len(class_counts)

    # Weight for each class = 1 / count
    # Rare classes get higher weight → sampled more often
    class_weights = {
        cls: 1.0 / count
        for cls, count in class_counts.items()
    }

    # Assign weight to each individual sample
    sample_weights = [class_weights[label] for label in labels]
    sample_weights = torch.DoubleTensor(sample_weights)

    sampler = WeightedRandomSampler(
        weights     = sample_weights,
        num_samples = len(sample_weights),
        replacement = True   # allows same image to be picked multiple times
    )

    print("WeightedRandomSampler created")
    print("Class weights (higher = sampled more):")
    for cls, w in sorted(class_weights.items()):
        print(f"  Class {cls:>3}: weight = {w:.6f}")

    return sampler


# ── DataLoader builder function ───────────────────────────────
def get_dataloaders(base_dir, class_to_idx, batch_size=32):
    """
    Creates train, val, test DataLoaders.
    Train uses WeightedRandomSampler to fix class imbalance.
    Val and test use simple sequential loading.
    """
    base_dir = Path(base_dir)

    train_dataset = WasteDataset(
        base_dir / "data" / "train",
        class_to_idx,
        transform=train_transforms
    )
    val_dataset = WasteDataset(
        base_dir / "data" / "val",
        class_to_idx,
        transform=val_transforms
    )
    test_dataset = WasteDataset(
        base_dir / "data" / "test",
        class_to_idx,
        transform=val_transforms   # no augmentation for test
    )

    # Weighted sampler only for training
    train_sampler = make_weighted_sampler(train_dataset)

    train_loader = DataLoader(
        train_dataset,
        batch_size  = batch_size,
        sampler     = train_sampler,  # replaces shuffle=True
        num_workers = 0,              # set to 0 on Windows to avoid errors
        pin_memory  = False
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size  = batch_size,
        shuffle     = False,
        num_workers = 0
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size  = batch_size,
        shuffle     = False,
        num_workers = 0
    )

    return train_loader, val_loader, test_loader
