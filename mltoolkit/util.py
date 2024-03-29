import os

import wandb

from mltoolkit.arguments import DataUseArguments, WandBArguments


def download_dataset(dataset_args: DataUseArguments, wandb_args: WandBArguments):
    artifact = wandb_args.run.use_artifact(os.path.join(wandb_args.project_path,
                                                        dataset_args.wandb_artifact_path),
                                           type='dataset')
    dataset_artifact_dir = artifact.download()
    return dataset_artifact_dir


def log_dataset_reference(dataset_args: DataUseArguments, wandb_args: WandBArguments, max_objects=10000, checksum=True):
    init_wandb(wandb_args)
    artifact = wandb.Artifact(dataset_args.name, type='dataset')
    path = dataset_args.file_uri
    if not path.startswith('file://'):
        path = f'file://{path}'
    artifact.add_reference(path, max_objects=max_objects, checksum=checksum)
    wandb_args.run.log_artifact(artifact, aliases=[dataset_args.version])
    return artifact


def init_wandb(wandb_args: WandBArguments):
    import wandb
    wandb.init(project=wandb_args.project,
               group=wandb_args.group,
               job_type=wandb_args.job_type,
               name=wandb_args.run_name,
               tags=wandb_args.tags,
               resume=wandb_args.resume)
    wandb_args._run = wandb.run


def fix_seeds(seed):
    import random
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
