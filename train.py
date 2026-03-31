"""
    Main Training Script
    Entry point for training neural networks with command-line arguments
"""

import argparse
import wandb
import yaml
import logging
import json
from pathlib import Path
from src.classification import TrainVGG11

parent = Path(__file__).resolve().parent

logging.basicConfig(filename = parent / "prog_log.log",
                format = '%(levelname)s: %(message)s',
                filemode = 'w')
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

with open(parent / "config.yaml", "r") as f:
    configs = yaml.full_load(f)

def learning_rate(lr):
    try:
        lr = float(lr)
    except:
        raise argparse.ArgumentTypeError(f"{lr} should be a floating-point value")
    
    if not (0.0 < lr < 1.0):
        raise argparse.ArgumentTypeError(f"{lr} not in (0, 1) range")
    return lr

def parse_arguments():
    """
        Parse command-line arguments.

        - task: [1, 2, 3, 4]
        - device: "cpu" or "cuda"
        - weight_decay: Weight decay factor
        - epochs: Number of training epochs
        - batch_size: Mini-batch size
        - learning_rate: Learning rate for optimizer
        - wandb_project: W&B project name
        - model_save_path: Path to save trained model (do not give absolute path,
          rather provide relative path)
    """

    parser = argparse.ArgumentParser(description='Train a neural network')
    
    parser.add_argument("-t", "--task",
                        type = int,
                        default = 1,
                        choices = [1, 2, 3, 4],
                        action = "store",
                        dest = "task",
                        help = "specify task"
    )

    parser.add_argument("-d", "--device",
                        type = str,
                        default = "cuda",
                        choices = ["cuda", "cpu"],
                        action = "store",
                        dest = "device",
                        help = "specify device"
    )

    parser.add_argument("-e", "--epochs",
                        type = int,
                        default = 15,
                        action = "store",
                        dest = "epochs",
                        help = "specify the number of epochs"
    )

    parser.add_argument("-b", "--batch_size",
                        type = int,
                        default = 128,
                        action = "store",
                        dest = "batch_size",
                        help = "specify the batch size"
    )

    parser.add_argument("-lr", "--learning_rate",
                        type = learning_rate,
                        default = 0.1,
                        action = "store",
                        dest = "lr",
                        help = "specify learning rate"
    )

    parser.add_argument("-wd", "--weight_decay",
                        type = float,
                        default = 0.1,
                        action = "store",
                        dest = "wgt_decay",
                        help = "specify weight decay for L2 regularization"
    )
    
    parser.add_argument("-w_p", "--wandb_project",
                        type = str,
                        default="da6401_ass1",
                        action = "store",
                        dest = "wandb",
                        help = "specify name of the wandb project"
    )

    parser.add_argument("-pth", "--model_save_path",
                        type = str,
                        action = "store",
                        dest = "path",
                        help = "specify relative path of the model"
    )

    return parser.parse_args()

def main():
    """
        Main training function.
    """

    args = parse_arguments()
    dic = vars(args)
    print(dic)
    print(args)
    configs = {
        "task": int(args.task if "task" in dic else args.t),
        "device": str(args.device if "device" in dic else args.d),
        "epochs": int(args.epochs if "epochs" in dic else args.e),
        "batch_size": int(args.batch_size if "batch_size" in dic else args.b),
        "lr": float(args.learning_rate if "learning_rate" in dic else args.lr),
        "wgt_decay": float(args.wgt_decay), # if "weight_decay" in dic else args.wd,)
        "wandb": str(args.wandb), # if "wandb_project" in dic else args.w_p,
        "path": str(args.path) # if "model_save_path" in dic else args.pth
    }
    configs["name"] = f"{configs["task"]}_lr{configs["lr"]}_wtd{configs["wgt_decay"]}_{configs["path"][:-4]}" # type: ignore

    wandb.init(
        project=configs["wandb"], # type: ignore
        config=configs,
        name = configs["name"] # type: ignore
    )

    with open(parent / f"train_{configs["task"]}.yaml", "w") as f:
        yaml.dump(configs, f, default_flow_style=False, sort_keys=False)

    logger.info("TRAIN CONFIGS CREATED")
    wandb_project = configs["wandb"]

    match configs["task"]:
        case 1:
            trainer = TrainVGG11(wandb_project)
            trainer.train_model()

    logger.info("TRAINING COMPLETE")

if __name__ == '__main__':
    main()

# -t 1 -e 1 -b 128 -lr 0.01 -wd 0.0001 -pth test1.pth -d cuda