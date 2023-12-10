import os
import pickle
import json
from src.accelerator import get_accelerator
from src.trainer import Trainer
from src.options import args_parser

if __name__ == "__main__":
    args = args_parser()
    accelerator = get_accelerator(args)
    
    trainer = Trainer(args, accelerator)
    trainer.fit()