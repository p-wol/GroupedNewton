import os
import time
import random
import numpy as np
import torch
import mlxpy
from training import Trainer


def set_seeds(seed):    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

@mlxpy.launch(config_path='./configs', seeding_function = set_seeds)
def main(ctx):
    try:
        trainer = ctx.logger.load_checkpoint(log_name = 'last_ckpt') 
        print("Loading from latest checkpoint")
    except:
        print("Failed to load checkpoint, Starting from scratch")
        trainer = Trainer(ctx.config, ctx.logger)

    trainer.train()

if __name__ == "__main__":
    main()
