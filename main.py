from tools import pretrain_run_net as pretrain
from tools import finetune_run_net as finetune
from utils import parser, dist_utils, misc
from utils.logger import *
from utils.config import *
import time
import os
import torch
from tensorboardX import SummaryWriter
# 过滤掉关于 DDP 梯度步长不匹配的警告
import warnings
warnings.filterwarnings("ignore", message="Grad strides do not match bucket view strides")

def main():
    # args
    args = parser.get_args()
    args.local_rank = int(os.environ["LOCAL_RANK"]) 
    # CUDA
    args.use_gpu = torch.cuda.is_available()
    if args.use_gpu:
        torch.backends.cudnn.benchmark = True
    if args.launcher == 'none':
        args.distributed = False
    else:
        args.distributed = True
        dist_utils.init_dist(args.launcher)
        # re-set gpu_ids with distributed training mode
        _, world_size = dist_utils.get_dist_info()
        args.world_size = world_size
    # logger
    if args.local_rank == 0:
        timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
        log_file = os.path.join(args.experiment_path, f'{timestamp}.log')
        logger = get_root_logger(log_file=log_file, name=args.log_name)
    else:
        logger = None
    # define the tensorboard writer
    if not args.test:
        if args.local_rank == 0: 
            train_writer = SummaryWriter(os.path.join(args.tfboard_path, 'train')) 
            val_writer = SummaryWriter(os.path.join(args.tfboard_path, 'test'))
        else:
            train_writer = None
            val_writer = None

    # config
    config = get_config(args, logger = logger) 
    config.dataset.train.others.log_name = args.log_name
    config.dataset.val.others.log_name = args.log_name
    # batch size
    if args.distributed:
        assert config.total_bs % world_size == 0
        config.dataset.train.others.bs = config.total_bs // world_size
        if config.dataset.get('extra_train'):
            config.dataset.extra_train.others.bs = config.total_bs // world_size * 2
        config.dataset.val.others.bs = config.total_bs // world_size * 2
        if config.dataset.get('test'):
            config.dataset.test.others.bs = config.total_bs // world_size 
    else:
        config.dataset.train.others.bs = config.total_bs
        if config.dataset.get('extra_train'):
            config.dataset.extra_train.others.bs = config.total_bs * 2
        config.dataset.val.others.bs = config.total_bs * 2 
        if config.dataset.get('test'):
            config.dataset.test.others.bs = config.total_bs 


    # set random seeds
    if args.seed is not None:
        misc.set_random_seed(args.seed, deterministic=args.deterministic) 
        # recommend to set seed for all processes
        if args.local_rank == 0:
            logger.info(f'Set random seed to {args.seed}, '
                        f'deterministic: {args.deterministic}')
    if args.distributed:
        assert args.local_rank == torch.distributed.get_rank() 
        if args.local_rank == 0:
            log_args_to_file(args, 'args', logger = logger)
            log_config_to_file(config, 'config', logger = logger)
            logger.info(f'Distributed training: {args.distributed}')
        
    # run
    if args.test:
        raise NotImplementedError('Test mode is not implemented yet.')
    else:
        if args.finetune_model or args.scratch_model:
            finetune(args, config, train_writer, val_writer)
        else:
            pretrain(args, config, train_writer, val_writer)


if __name__ == '__main__':
    main()
