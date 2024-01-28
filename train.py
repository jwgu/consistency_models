import json
import os
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

@dataclass
class ImageDataModuleConfig:
    data_dir: str = "butterflies256"
    image_size: Tuple[int, int] = (32, 32)
    batch_size: int = 32
    num_workers: int = 8
    pin_memory: bool = True
    persistent_workers: bool = True


class ImageDataModule(LightningDataModule):
    def __init__(self, config: ImageDataModuleConfig) -> None:
        super().__init__()

        self.config = config

    def setup(self, stage: str = None) -> None:
        transform = T.Compose(
            [
                T.Resize(self.config.image_size),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                T.Lambda(lambda x: (x * 2) - 1),
            ]
        )
        self.dataset = ImageFolder(self.config.data_dir, transform=transform)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
            persistent_workers=self.config.persistent_workers,
        )


#----------------------- consistency model training ----------------------
@dataclass
class LitConsistencyModelConfig:
    initial_ema_decay_rate: float = 0.95
    student_model_ema_decay_rate: float = 0.99993
    lr: float = 1e-4
    betas: Tuple[float, float] = (0.9, 0.995)
    lr_scheduler_start_factor: float = 1e-5
    lr_scheduler_iters: int = 10_000
    sample_every_n_steps: int = 10_000
    num_samples: int = 8
    sampling_sigmas: Tuple[Tuple[int, ...], ...] = (
        (80,),
        (80.0, 0.661),
        (80.0, 24.4, 5.84, 0.9, 0.661),
    )


class LitConsistencyModel(LightningModule):
    def __init__(
        self,
        consistency_training: ConsistencyTraining,
        consistency_sampling: ConsistencySamplingAndEditing,
        student_model: UNet,
        teacher_model: UNet,
        ema_student_model: UNet,
        config: LitConsistencyModelConfig,
    ) -> None:
        super().__init__()

        self.consistency_training = consistency_training
        self.consistency_sampling = consistency_sampling
        self.student_model = student_model
        self.teacher_model = teacher_model
        self.ema_student_model = ema_student_model
        self.config = config
        self.num_timesteps = self.consistency_training.initial_timesteps

        self.lpips = LearnedPerceptualImagePatchSimilarity(net_type="alex")

        # Freeze teacher and EMA student models and set to eval mode
        for param in self.teacher_model.parameters():
            param.requires_grad = False
        for param in self.ema_student_model.parameters():
            param.requires_grad = False
        self.teacher_model = self.teacher_model.eval()
        self.ema_student_model = self.ema_student_model.eval()

    def training_step(self, batch: Union[Tensor, List[Tensor]], batch_idx: int) -> None:
        if isinstance(batch, list):
            batch = batch[0]

        output = self.consistency_training(
            self.student_model,
            self.teacher_model,
            batch,
            self.global_step,
            self.trainer.max_steps,
        )
        self.num_timesteps = output.num_timesteps

        lpips_loss = self.lpips(
            output.predicted.clamp(-1.0, 1.0), output.target.clamp(-1.0, 1.0)
        )
        overflow_loss = F.mse_loss(
            output.predicted, output.predicted.detach().clamp(-1.0, 1.0)
        )
        loss = lpips_loss + overflow_loss

        self.log_dict(
            {
                "train_loss": loss,
                "lpips_loss": lpips_loss,
                "overflow_loss": overflow_loss,
                "num_timesteps": output.num_timesteps,
            }
        )

        return loss

    def on_train_batch_end(
        self, outputs: Any, batch: Union[Tensor, List[Tensor]], batch_idx: int
    ) -> None:
        # Update teacher model
        ema_decay_rate = ema_decay_rate_schedule(
            self.num_timesteps,
            self.config.initial_ema_decay_rate,
            self.consistency_training.initial_timesteps,
        )
        update_ema_model_(self.teacher_model, self.student_model, ema_decay_rate)
        self.log_dict({"ema_decay_rate": ema_decay_rate})

        # Update EMA student model
        update_ema_model_(
            self.ema_student_model,
            self.student_model,
            self.config.student_model_ema_decay_rate,
        )

        if (
            (self.global_step + 1) % self.config.sample_every_n_steps == 0
        ) or self.global_step == 0:
            self.__sample_and_log_samples(batch)

    def configure_optimizers(self):
        opt = torch.optim.Adam(
            self.student_model.parameters(), lr=self.config.lr, betas=self.config.betas
        )
        sched = torch.optim.lr_scheduler.LinearLR(
            opt,
            start_factor=self.config.lr_scheduler_start_factor,
            total_iters=self.config.lr_scheduler_iters,
        )
        sched = {"scheduler": sched, "interval": "step", "frequency": 1}

        return [opt], [sched]

    @torch.no_grad()
    def __sample_and_log_samples(self, batch: Union[Tensor, List[Tensor]]) -> None:
        if isinstance(batch, list):
            batch = batch[0]

        # Ensure the number of samples does not exceed the batch size
        num_samples = min(self.config.num_samples, batch.shape[0])
        noise = torch.randn_like(batch[:num_samples])

        # Log ground truth samples
        self.__log_images(
            batch[:num_samples].detach().clone(), f"ground_truth", self.global_step
        )

        for sigmas in self.config.sampling_sigmas:
            samples = self.consistency_sampling(
                self.ema_student_model, noise, sigmas, clip_denoised=True, verbose=True
            )
            samples = samples.clamp(min=-1.0, max=1.0)

            # Generated samples
            self.__log_images(
                samples,
                f"generated_samples-sigmas={sigmas}",
                self.global_step,
            )

    @torch.no_grad()
    def __log_images(self, images: Tensor, title: str, global_step: int) -> None:
        images = images.detach().float()

        grid = make_grid(
            images.clamp(-1.0, 1.0), value_range=(-1.0, 1.0), normalize=True
        )
        self.logger.experiment.add_image(title, grid, global_step)


@dataclass
class ConsistencyModelTrainingConfig:
    image_dm_config: ImageDataModuleConfig
    unet_config: UNetConfig
    consistency_training: ConsistencyTraining
    consistency_sampling: ConsistencySamplingAndEditing
    lit_cm_config: LitConsistencyModelConfig
    trainer: Trainer
    seed: int = 42
    model_ckpt_path: str = "checkpoints/cm"
    resume_ckpt_path: Optional[str] = None


#--------------------- Improved Consistency Model Training --------------
@dataclass
class LitImprovedConsistencyModelConfig:
    ema_decay_rate: float = 0.99993
    lr: float = 1e-4
    betas: Tuple[float, float] = (0.9, 0.995)
    lr_scheduler_start_factor: float = 1e-5
    lr_scheduler_iters: int = 10000
    sample_every_n_steps: int = 10000
    num_samples: int = 8
    sampling_sigmas: Tuple[Tuple[int, ...], ...] = (
        (80,),
        (80.0, 0.661),
        (80.0, 24.4, 5.84, 0.9, 0.661),
    )


class LitImprovedConsistencyModel(LightningModule):
    def __init__(
        self,
        consistency_training: ImprovedConsistencyTraining,
        consistency_sampling: ConsistencySamplingAndEditing,
        model: UNet,
        ema_model: UNet,
        config: LitImprovedConsistencyModelConfig,
    ) -> None:
        super().__init__()

        self.consistency_training = consistency_training
        self.consistency_sampling = consistency_sampling
        self.model = model
        self.ema_model = ema_model
        self.config = config

        # Freeze the EMA model and set it to eval mode
        for param in self.ema_model.parameters():
            param.requires_grad = False
        self.ema_model = self.ema_model.eval()

    def training_step(self, batch: Union[Tensor, List[Tensor]], batch_idx: int) -> None:
        if isinstance(batch, list):
            batch = batch[0]

        output = self.consistency_training(
            self.model, batch, self.global_step, self.trainer.max_steps
        )

        loss = (
            pseudo_huber_loss(output.predicted, output.target) * output.loss_weights
        ).mean()

        self.log_dict({"train_loss": loss, "num_timesteps": output.num_timesteps})

        return loss

    def on_train_batch_end(
        self, outputs: Any, batch: Union[Tensor, List[Tensor]], batch_idx: int
    ) -> None:
        update_ema_model_(self.model, self.ema_model, self.config.ema_decay_rate)

        if (
            (self.global_step + 1) % self.config.sample_every_n_steps == 0
        ) or self.global_step == 0:
            self.__sample_and_log_samples(batch)

    def configure_optimizers(self):
        opt = torch.optim.Adam(
            self.model.parameters(), lr=self.config.lr, betas=self.config.betas
        )
        sched = torch.optim.lr_scheduler.LinearLR(
            opt,
            start_factor=self.config.lr_scheduler_start_factor,
            total_iters=self.config.lr_scheduler_iters,
        )
        sched = {"scheduler": sched, "interval": "step", "frequency": 1}

        return [opt], [sched]

    @torch.no_grad()
    def __sample_and_log_samples(self, batch: Union[Tensor, List[Tensor]]) -> None:
        if isinstance(batch, list):
            batch = batch[0]

        # Ensure the number of samples does not exceed the batch size
        num_samples = min(self.config.num_samples, batch.shape[0])
        noise = torch.randn_like(batch[:num_samples])

        # Log ground truth samples
        self.__log_images(
            batch[:num_samples].detach().clone(), f"ground_truth", self.global_step
        )

        for sigmas in self.config.sampling_sigmas:
            samples = self.consistency_sampling(
                self.ema_model, noise, sigmas, clip_denoised=True, verbose=True
            )
            samples = samples.clamp(min=-1.0, max=1.0)

            # Generated samples
            self.__log_images(
                samples,
                f"generated_samples-sigmas={sigmas}",
                self.global_step,
            )

    @torch.no_grad()
    def __log_images(self, images: Tensor, title: str, global_step: int) -> None:
        images = images.detach().float()

        grid = make_grid(
            images.clamp(-1.0, 1.0), value_range=(-1.0, 1.0), normalize=True
        )
        self.logger.experiment.add_image(title, grid, global_step)
        
@dataclass
class ImprovedConsistencyModelTrainingConfig:
    image_dm_config: ImageDataModuleConfig
    unet_config: UNetConfig
    consistency_training: ImprovedConsistencyTraining
    consistency_sampling: ConsistencySamplingAndEditing
    lit_icm_config: LitImprovedConsistencyModelConfig
    trainer: Trainer
    seed: int = 42
    model_ckpt_path: str = "checkpoints/icm"
    resume_ckpt_path: Optional[str] = None

#------------------- main interface --------------------   
def show_unet():
    # for debug purpose
    summary(UNet(UNetConfig()), input_size=((1, 3, 32, 32), (1,)))


def train_consistency_model():
    """
    args TODO
        data -- bufferflies or cifar10 or mnist, etc.
        loss -- lpips or dists    
        steps --- xxx
    """
    config = ConsistencyModelTrainingConfig(
        image_dm_config=ImageDataModuleConfig("butterflies255"),
        unet_config=UNetConfig(),
        consistency_training=ConsistencyTraining(final_timesteps=16),
        consistency_sampling=ConsistencySamplingAndEditing(),
        lit_cm_config=LitConsistencyModelConfig(
            sample_every_n_steps=999, lr_scheduler_iters=1000
        ),
        trainer=Trainer(
            max_steps=10000,
            log_every_n_steps=9,
            logger=TensorBoardLogger(".", name="logs", version="cm"),
            callbacks=[LearningRateMonitor(logging_interval="step")],
        ),
    )

    # Set seed
    seed_everything(config.seed)

    # Create data module
    dm = ImageDataModule(config.image_dm_config)

    # Create student and teacher models and EMA student model
    student_model = UNet(config.unet_config)
    teacher_model = UNet(config.unet_config)
    teacher_model.load_state_dict(student_model.state_dict())
    ema_student_model = UNet(config.unet_config)
    ema_student_model.load_state_dict(student_model.state_dict())

    # Create lightning module
    lit_cm = LitConsistencyModel(
        config.consistency_training,
        config.consistency_sampling,
        student_model,
        teacher_model,
        ema_student_model,
        config.lit_cm_config,
    )

    # Run training
    config.trainer.fit(lit_cm, dm, ckpt_path=config.resume_ckpt_path)

    # Save model
    lit_cm.ema_student_model.save_pretrained(config.model_ckpt_path)


def train_consistency_model_improved():
    """
    args TODO
        data -- bufferflies or cifar10 or mnist, etc.
        loss -- lpips or dists    
        steps --- xxx
    """ 
    config = ImprovedConsistencyModelTrainingConfig(
        image_dm_config=ImageDataModuleConfig("butterflies256"),
        unet_config=UNetConfig(),
        consistency_training=ImprovedConsistencyTraining(final_timesteps=11),
        consistency_sampling=ConsistencySamplingAndEditing(),
        lit_icm_config=LitImprovedConsistencyModelConfig(
            sample_every_n_steps=1000, lr_scheduler_iters=1000
            ),
        trainer=Trainer(
            max_steps=10000,
            log_every_n_steps=10,
            logger=TensorBoardLogger(".", name="logs", version="icm"),
            callbacks=[LearningRateMonitor(logging_interval="step")],
        ),
    )

    # Set seed
    seed_everything(config.seed)

    # Create data module
    dm = ImageDataModule(config.image_dm_config)

    # Create model and its EMA
    model = UNet(config.unet_config)
    ema_model = UNet(config.unet_config)
    ema_model.load_state_dict(model.state_dict())

    # Create lightning module
    lit_icm = LitImprovedConsistencyModel(
        config.consistency_training,
        config.consistency_sampling,
        model,
        ema_model,
        config.lit_icm_config,
    )

    # Run training
    config.trainer.fit(lit_icm, dm, ckpt_path=config.resume_ckpt_path)

    # Save model
    lit_icm.model.save_pretrained(config.model_ckpt_path)


import argparse

if __name__=='__main__':

    parser = argparse.ArgumentParser()
    
    parser.add_argument('-m','--mode', type=str, help="mode of show, cm, or icm")
    args = parser.parse_args()
    
    if args.mode=='show':
        show_unet()
    elif args.mode=='cm':
        train_consistency_model()
    elif args.mode=='icm':
        train_consistency_model_improved()
