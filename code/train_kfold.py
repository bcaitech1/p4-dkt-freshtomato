import os
import numpy as np
import pickle
import json

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
    fold_counter = 0

    args.model_dir = os.path.join(args.model_dir, args.wandb_run_name)
    model_dir = args.model_dir

    # fold별 acc&auc와 average acc&auc를 저장할 dictionary
    final_score_dict = {}
    # K-Fold 학습 시작
    for train_idx, test_idx in skf.split(train_all_data, user_stratified_key):
        train_data = train_all_data[train_idx]
        valid_data = train_all_data[test_idx]

        wandb.init(
            project=args.wandb_project_name,
            name=args.wandb_run_name + f"_fold_{fold_counter}",
            config=vars(args),
        )

        args.model_dir = f"{model_dir}/fold_{fold_counter}"
        print(f"Start training fold {fold_counter}")
        trainer.run(args, train_data, valid_data)

        # fold별로 저장된 dict 불러와서 AUC, ACC를 불러오는 코드
        with open(f"{model_dir}/fold_{fold_counter}/best_dict.json", "r") as best_dict:
            load_dict_per_fold = json.load(best_dict)
            final_score_dict[
                f"{fold_counter} Fold Best Valid AUC"
            ] = load_dict_per_fold["best_valid_auc"]
            final_score_dict[
                f"{fold_counter} Fold Best Valid ACC"
            ] = load_dict_per_fold["best_valid_acc"]

        fold_counter += 1
        wandb.finish()

    final_score_dict["Average AUC"] = np.mean(
        [val for key, val in final_score_dict.items() if key.endswith("AUC")]
    )
    final_score_dict["Average ACC"] = np.mean(
        [val for key, val in final_score_dict.items() if key.endswith("ACC")]
    )

    # AUC & ACC를 저장한 dictionary json 형태로 저장
    print("Saving Final Score Dict...")
    ## json파일 보기 좋게 저장 ##
    final_score_dict = {
        key: value
        for key, value in sorted(final_score_dict.items(), key=lambda x: x[0][-3:])
    }
    with open(f"{model_dir}/final_score_dict.json", "w") as fp:
        json.dump(final_score_dict, fp, indent=4)

    ### 여기가 Score 출력하는 부분 ###
    print("=" * 30)
    for key, value in final_score_dict.items():
        print(f"{key} : {value}")
    print("=" * 30)


if __name__ == "__main__":
    args = parse_args(mode="train")
    main(args)
