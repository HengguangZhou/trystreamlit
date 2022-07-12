from dataclasses import dataclass, field
from typing import Optional

from dataclasses import dataclass, field
from typing import Optional



@dataclass
class ReSerArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    re_output_dir: str = field(
        metadata={"help": "The output directory where the model predictions and checkpoints will be written."},
    )

    ser_output_dir: str = field(
        metadata={"help": "The output directory where the model predictions and checkpoints will be written."},
    )

    re_model_name_or_path: str = field(
        metadata={"help": "Path to pretrained re model or re model identifier from huggingface.co/models"}
    )

    ser_model_name_or_path: str = field(
        metadata={"help": "Path to pretrained ser model or ser model identifier from huggingface.co/models"}
    )

    re_do_train: bool = field(default=False, metadata={"help": "Whether to run training."})
    re_do_eval: bool = field(default=None, metadata={"help": "Whether to run eval on the dev set."})
    re_do_predict: bool = field(default=False, metadata={"help": "Whether to run predictions on the test set."})

    ser_do_train: bool = field(default=False, metadata={"help": "Whether to run training."})
    ser_do_eval: bool = field(default=None, metadata={"help": "Whether to run eval on the dev set."})
    ser_do_predict: bool = field(default=False, metadata={"help": "Whether to run predictions on the test set."})

    overwrite_re_output_dir: bool = field(
        default=False,
        metadata={
            "help": (
                "Overwrite the content of the output directory."
                "Use this to continue training if output_dir points to a checkpoint directory."
            )
        },
    )

    overwrite_ser_output_dir: bool = field(
        default=False,
        metadata={
            "help": (
                "Overwrite the content of the output directory."
                "Use this to continue training if output_dir points to a checkpoint directory."
            )
        },
    )
    
    re_config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    ser_config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )

    re_tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    ser_tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )

    ser_max_steps: int = field(
        default=-1,
        metadata={"help": "If > 0: set total number of training steps to perform. Override num_train_epochs."},
    )

    re_max_steps: int = field(
        default=-1,
        metadata={"help": "If > 0: set total number of training steps to perform. Override num_train_epochs."},
    )