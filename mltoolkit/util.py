import os

from arguments import DataUseArguments, WandBArguments


def download_dataset(dataset_args: DataUseArguments, wandb_args: WandBArguments):
    artifact = wandb_args.run.use_artifact(os.path.join(wandb_args.project_path, dataset_args.artifact_path),
                                           type='dataset')
    dataset_artifact_dir = artifact.download()
    return dataset_artifact_dir
