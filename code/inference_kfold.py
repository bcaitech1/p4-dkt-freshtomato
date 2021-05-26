import os
from args import parse_args
from dkt.dataloader import Preprocess
from dkt import trainer
import torch

def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    args.device = device

    preprocess = Preprocess(args)
    preprocess.load_test_data(args.test_file_name)
    test_data = preprocess.get_test_data()
    
    model_dir = args.model_dir
    args.output_dir = os.path.join(args.output_dir, args.wandb_run_name)
    
    for fold in range(args.kfold):
        args.model_dir = f"{model_dir}/fold_{fold}"
        args.fold = fold
        trainer.inference(args, test_data)


if __name__ == "__main__":
    args = parse_args(mode='train_kfold')
    args.model_dir = os.path.join(args.model_dir, args.wandb_run_name)

    main(args)