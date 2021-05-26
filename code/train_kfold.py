import os
from args import parse_args
from dkt.dataloader import Preprocess
from dkt import trainer
import torch
from dkt.utils import set_seed
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold
import wandb


def main(args):
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    args.device = device

    preprocess = Preprocess(args)

    print("loading train data...")
    preprocess.load_train_data(args.file_name)

    print("get preprocessed data...")
    train_all_data = preprocess.get_train_data()

    wandb.login()
    user_stratified_key = preprocess.get_user_stratified_key()

    skf = StratifiedKFold(n_splits=args.kfold, shuffle=True, random_state=0)
    fold_counter=0

    args.model_dir = os.path.join(args.model_dir, args.wandb_run_name)
    model_dir = args.model_dir

    for train_idx, test_idx in skf.split(train_all_data, user_stratified_key):
        train_data = train_all_data[train_idx]
        valid_data = train_all_data[test_idx]

        wandb.init(project="minyong", name=args.wandb_run_name + f"_fold_{fold_counter}", config=vars(args))

        args.model_dir = f"{model_dir}/fold_{fold_counter}"

        print(f"Start training fold {fold_counter}")
        trainer.run(args, train_data, valid_data)

        fold_counter += 1
        wandb.finish()

if __name__ == "__main__":
    args = parse_args(mode="train")
    main(args)
