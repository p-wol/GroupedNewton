import os
import time
import random
import numpy as np
import torch
import hydra
from omegaconf import DictConfig, OmegaConf
from training_hydra import Trainer


@hydra.main(version_base=None, config_path="./configs", config_name="config_hydra")
def main(cfg):
    """
    try:
        trainer = ctx.logger.load_checkpoint(log_name = 'last_ckpt') 
        print("Loading from latest checkpoint")
    except:
        print("Failed to load checkpoint, Starting from scratch")
        trainer = Trainer(ctx.config, ctx.logger)
    """
    trainer = Trainer(cfg, hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    trainer.train()

if __name__ == "__main__":
    main()

