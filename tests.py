from dataclasses import field
from typing import Dict, List

from mltoolkit import argclass, parse_args, GeneralArguments, asdict

from mltoolkit import GeneralArguments
from mltoolkit.argparser import argclass
from dataclasses import field


@argclass
class TrainingArguments:
    num_steps: int = field(default=1000)
    num_minibatches: int = field(default=4)
    epsilon: float = field(default=0.2)
    value_coef: float = field(default=0.5)
    entropy_coef: float = field(default=0.01)
    max_steps: int = field(default=500)
    num_envs: int = field(default=8)
    hidden_size: int = field(default=256)
    scale: float = field(default=0.5)
    lr: float = field(default=1e-4)
    ppo_epochs: int = field(default=4)
    gru_layers: int = field(default=1)
    gamma: float = field(default=0.99)
    embd_dim: int = field(default=32)


@argclass
class EnvironmentArguments:
    max_steps: int = field(default=500)
    reward_type: str = field(default='sparse')
    num_objects: int = field(default=1)
    agent_visibility: int = field(default=3)
    grid_size: int = field(default=5)
    difficulty: str = field(default='hard')
    static_env: bool = field(default=False)


@argclass
class MazeArguments(GeneralArguments):
    train: TrainingArguments
    env: EnvironmentArguments

    log_every: int = field(default=10)
    verbose: bool = field(default=True)
    trainer_name: str = field(default='ppo')
    net_name: str = field(default='simple')

    @property
    def state_dim(self):
        if self.agent_visibility == -1:
            return self.size
        else:
            return self.agent_visibility


def main():
    args = parse_args(MazeArguments)
    import pickle
    pickle.loads(pickle.dumps(args))
    print(asdict(args))




main()
