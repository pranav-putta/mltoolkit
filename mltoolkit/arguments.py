import os
from dataclasses import field
from typing import Optional, Any, Sequence

from mltoolkit.argparser import argclass
import torch
import similaritymeasures


@argclass
class GeneralArguments:
    no_gpu: bool = field(default=False, metadata={'help': 'do not use GPU acceleration when available'})
    workers: int = field(default=1, metadata={'help': 'number of workers to use'})
    _device: str = None

    config: Optional[str] = field(default=None, metadata={'help': 'load arguments from config file/folder'})

    def __post_init__(self):
        # check for devices in order ['cuda', 'mps', 'cpu']
        if torch.cuda.is_available() and not self.no_gpu:
            self._device = 'cuda'
        elif 'mps' in dir(torch.backends) and torch.backends.mps.is_available() and not self.no_gpu:
            self._device = 'mps'
        else:
            self._device = 'cpu'

    @property
    def device(self):
        return self._device


@argclass
class WandBArguments:
    project: Optional[str] = field(default=None, metadata={'help': 'wandb project name'})
    group: Optional[str] = field(default=None, metadata={'help': 'wandb run group name'})
    job_type: Optional[str] = field(default=None,
                                    metadata={'help': 'wandb run job type name (e.g. train, eval, data processing)'})
    run_name: Optional[str] = field(default=None, metadata={'help': 'wandb run name'})
    workspace: str = field(default=None, metadata={'help': 'wandb workspace name'})
    tags: Sequence[str] = field(default=None, metadata={'help': 'wandb tags'})
    resume: str = field(default=None, metadata={'help': 'run id to resume'})

    _run: Any = field(default=None)

    @property
    def run(self):
        return self._run

    @property
    def project_path(self):
        return os.path.join(self.workspace, self.project)


@argclass
class DataUseArguments:
    """
    dataset arguments ; compatible with wandb utils
    """
    name: str = field(default=None, metadata={'help': 'data name to use'})
    version: str = field(default=None, metadata={'help': 'data version to use'})

    @property
    def artifact_path(self):
        return f'{self.name}:{self.version}'
