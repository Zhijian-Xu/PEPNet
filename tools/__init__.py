# from .runner import run_net
from .runner_finetune_epitope_ddp import run_net as pretrain_run_net
from .runner_finetune_epitope_ddp import run_net as finetune_run_net