import os
from dataclasses import dataclass, field

from trl.trainer.utils import OnPolicyConfig


@dataclass
class CoCoSConfig(OnPolicyConfig):
    exp_name: str = field(
        default=os.path.basename(__file__)[:-3],
        metadata={"help": "Name of this experiment."},
    )
    reward_model_path: str = field(
        default="EleutherAI/pythia-160m",
        metadata={"help": "Path to the reward model."},
    )
    num_ppo_epochs: int = field(
        default=4,
        metadata={"help": "Number of epochs to train."},
    )
    whiten_rewards: bool = field(
        default=False,
        metadata={"help": "Whether to whiten the rewards."},
    )
    kl_coef: float = field(
        default=0.05,
        metadata={"help": "KL coefficient."},
    )
    cliprange: float = field(
        default=0.2,
        metadata={"help": "Clip range."},
    )
    rloo_k: int = field(
        default=2,
        metadata={"help": "REINFORCE Leave-One-Out (RLOO) number of online samples per prompt."},
    )
    normalize_reward: bool = field(
        default=False,
        metadata={"help": "Whether to normalize rewards"},
    )
    reward_clip_range: float = field(
        default=10.0,
        metadata={"help": "Clip range for rewards"},
    )
    normalize_advantage: bool = field(
        default=False,
        metadata={"help": "Whether to normalize advantages"},
    )
    token_level_kl: bool = field(
        default=False,
        metadata={"help": "Whether to use token-level KL penalty or sequence-level KL penalty"},
    )
    ds3_gather_for_generation: bool = field(
        default=True,
        metadata={
            "help": "This setting applies to DeepSpeed ZeRO-3. If enabled, the policy model weights are gathered for "
            "generation, improving generation speed. However, disabling this option allows training models that "
            "exceed the VRAM capacity of a single GPU, albeit at the cost of slower generation."
        },
    )
    max_seq_len: int = field(
        default=8192,
        metadata={"help": "Maximum sequence length."},
    )
    temperature: float = field(
        default=0.9,
        metadata={"help": "Temperature."},
    )
    num_turns: int = field(
        default=2,
        metadata={"help": "Number of turns."},
    )
    weight: float = field(
        default=0.5,
        metadata={"help": "Reward weight."},
    )