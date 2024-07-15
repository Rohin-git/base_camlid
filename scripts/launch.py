# Copyright 2023 Toyota Research Institute.  All rights reserved.
# Copyright 2023 Toyota Research Institute.  All rights reserved.

import os

import fire
import torch
print("Starting the script...")

try:
    from vidar.core.trainer import Trainer
    print("Trainer imported successfully")
except ImportError as e:
    print(f"Failed to import Trainer: {e}")
    raise

try:
    from vidar.utils.config import read_config
    print("read_config imported successfully")
except ImportError as e:
    print(f"Failed to import read_config: {e}")
    raise

try:
    from vidar.core.wrapper import Wrapper
    print("Wrapper imported successfully")
except ImportError as e:
    print(f"Failed to import Wrapper: {e}")
    raise

print("All imports successful, proceeding with the rest of the script...")

def train():
    

    os.environ['DIST_MODE'] = 'gpu' if torch.cuda.is_available() else 'cpu'
    config_file_path = '/workspace/vidar/configs/overfit/kitti/selfsup_resnet18.yaml'
    print(f"Using configuration file at {config_file_path}")

    cfg = read_config(config_file_path)
    #cfg = read_config(cfg, **kwargs)

    wrapper = Wrapper(cfg, verbose=True)
    trainer = Trainer(cfg)
    trainer.learn(wrapper)

def main():
    train()

    
if __name__ == '__main__':
    main()

