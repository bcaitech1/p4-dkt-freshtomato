import os
from args import parse_args
from dkt.dataloader import Preprocess
from dkt import trainer
import torch
from dkt.utils import set_seed
import wandb


def main(args):
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    args.device = device

    preprocess = Preprocess(args)

    print("loading train data...")
    preprocess.load_train_data(args.file_name)

    print("get preprocessed data...")
    train_data = preprocess.get_train_data()

    print("split train, valid dataset")
    train_data, valid_data = preprocess.split_data(train_data)

    # wandb setting
    wandb.login()
    wandb.init(project=args.wandb_project_name, name=args.wandb_run_name, config=vars(args))

    args.model_dir = f"{args.model_dir}/{args.wandb_run_name}"
    os.makedirs(args.model_dir, exist_ok=True)

    trainer.run(args, train_data, valid_data)


if __name__ == "__main__":
    args = parse_args(mode="train")
    main(args)
