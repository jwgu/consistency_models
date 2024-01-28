import os, json
from dataclasses import asdict, dataclass
from typing import Any, Callable, List, Optional, Tuple, Union

import torch
from lightning import LightningDataModule, LightningModule, Trainer, seed_everything
from lightning.pytorch.callbacks import LearningRateMonitor
from lightning.pytorch.loggers import TensorBoardLogger
from matplotlib import pyplot as plt
from torch import Tensor, nn
from torch.utils.data import DataLoader
from torchinfo import summary
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchvision import transforms as T
from torchvision.datasets import ImageFolder
from torchvision.utils import make_grid

from consistency_models import (
    ConsistencySamplingAndEditing,
    ConsistencyTraining,
    ImprovedConsistencyTraining,
    pseudo_huber_loss,
    ema_decay_rate_schedule,
)
from consistency_models.utils import update_ema_model_

from unet import UNet, UNetConfig  
from train import ImageDataModule, ImageDataModuleConfig

#------------------------------------------------------------------------------------------
def plot_images(images: Tensor, cols: int = 4) -> None:
    rows = max(images.shape[0] // cols, 1)
    fig, axs = plt.subplots(rows, cols)
    axs = axs.flatten()
    for i, image in enumerate(images):
        axs[i].imshow(image.permute(1, 2, 0).numpy() / 2 + 0.5)
        axs[i].set_axis_off()
        
def sample_consistency_model():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    unet = UNet.from_pretrained("checkpoints/cm").eval().to(device=device, dtype=dtype)
    
    dm = ImageDataModule(ImageDataModuleConfig("butterflies256", batch_size=4))
    dm.setup()
    
    batch, _ = next(iter(dm.train_dataloader()))
    batch = batch.to(device=device, dtype=dtype)
    plot_images(batch.float().cpu())

    consistency_sampling_and_editing = ConsistencySamplingAndEditing()

    with torch.no_grad():
        samples = consistency_sampling_and_editing(
            unet,
            torch.randn((4, 3, 32, 32), device=device, dtype=dtype),
            sigmas=[80.0],  # Use more steps for better samples e.g 2-5
            clip_denoised=True,
            verbose=True,
        )

    plot_images(samples.float().cpu())


    # inpaint
    random_erasing = T.RandomErasing(p=1.0, scale=(0.2, 0.5), ratio=(0.5, 0.5))
    masked_batch = random_erasing(batch)
    mask = torch.logical_not(batch == masked_batch)

    plot_images(masked_batch.float().cpu())

    with torch.no_grad():
        inpainted_batch = consistency_sampling_and_editing(
            unet,
            masked_batch,
            sigmas=[5.23, 2.25],
            mask=mask.to(dtype=dtype),
            clip_denoised=True,
            verbose=True,
        )

    plot_images(torch.cat((masked_batch, inpainted_batch), dim=0).float().cpu())


    # interpolate
    batch_a = batch.clone()
    batch_b = torch.flip(batch, dims=(0,))
    plot_images(torch.cat((batch_a, batch_b), dim=0).float().cpu()) 

    with torch.no_grad():
        interpolated_batch = consistency_sampling_and_editing.interpolate(
            unet,
            batch_a,
            batch_b,
            ab_ratio=0.5,
            sigmas=[5.23, 2.25],
            clip_denoised=True,
            verbose=True,
        )

    plot_images(torch.cat((batch_a, batch_b, interpolated_batch), dim=0).float().cpu())