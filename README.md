# mltoolkit
set of universal tools for personal machine learning projects

## Argparse -- powerful argument generation
### Initialization
Simply create a class with a set of fields. Wrap the class with the @argclass annotation and that's it!

```python
@argclass
class GPT2MTCArguments(GeneralArguments):
    privacy: PrivacyArguments
    training: LMTrainingArguments
    wandb: WandBArguments
    dataset: LMDatasetUseArguments
    gen: GPT2MTCGenerationArguments
    model: ModelCreateArguments
    prefix_step_size: int = field(default=0)
    prefix_decay: str = field(default='const')

    lm_loss_weight: float = field(default=1.)
    class_loss_weight: float = field(default=1.)

    def __post_init__(self):
        super().__post_init__()
        if self.wandb.job_type is None:
            self.wandb.job_type = 'train'
        if self.wandb.run_name is None:
            self.wandb.run_name = f'gpt2 dp train {self.dataset.name} eps={self.privacy.eps}'
```
You can nest arguments by type annotating names of other argclasses. These arguments will be auto-initialized to defaults specified in their definition. Inherit arguments to the same level as the defined class by extending another argclass.

Nested arguments can be accessed through commandline arguments through dot flattening. For example, `--training.batch_size 8` would change the value of the `batch_size` field stored nested in the training field.

### Usage
Either type in commandline arguments or specify a configuration file (.yaml or .json). Set `resolve_config=True` to automatically load configuration arguments. First loads config file args, then overrides with command line args. After defining an argclass, populating is simple:
```python
args: GPT2MTCArguments = parse_args(GPT2MTCArguments, resolve_config=True)
```
