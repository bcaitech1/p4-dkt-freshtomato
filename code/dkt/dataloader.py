import os
from datetime import datetime
import time
import tqdm
import pandas as pd
import random
from sklearn.preprocessing import LabelEncoder
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from collections import defaultdict
import pickle


class Preprocess:
    def __init__(self, args):
        self.args = args
        self.train_data = None
        self.test_data = None
        self.user_stratified_key = None

    def get_train_data(self):
        return self.train_data

    def get_test_data(self):
        return self.test_data

    def get_user_stratified_key(self):
        return self.user_stratified_key

    def split_data(self, data, ratio=0.9, shuffle=True):
        """
        split data into two parts with a given ratio.
        """
        seed = self.args.seed

        if shuffle:
            random.seed(seed)  # fix to default seed 0
            random.shuffle(data)

        size = int(len(data) * ratio)
        data_1 = data[:size]
        data_2 = data[size:]

        return data_1, data_2

    def __make_stratified_key(self, df):
        # TODO
        pass

    def __feature_engineering(self, df):
        # TODO
        df = df.sort_values(by=["userID", "Timestamp"], axis=0)

        # (1) 풀고 있는 assessmentItemID 의 정답률을 나타내는 feature

        # -> train dataset에서 구한 정답률을 미리 저장하는 코드입니다.
        # assessmentItemID_mean_sum = df.groupby(['assessmentItemID'])['answerCode'].agg(['mean', 'sum']).to_dict()
        # le_path = os.path.join(self.args.asset_dir, "assessmentItemID_mean_sum.pk")
        # with open(le_path, 'wb') as f:
        #     pickle.dump(assessmentItemID_mean_sum, f)

        # -> 저장된 정답률을 가져와서 mapping하는 코드입니다.
        le_path = os.path.join(self.args.asset_dir, "assessmentItemID_mean_sum.pk")
        with open(le_path, 'rb') as f:
            assessmentItemID_mean_sum = pickle.load(f)
        df["assessmentItemID_mean"] = df.assessmentItemID.map(assessmentItemID_mean_sum['mean'])


        # (2) 해당 시험지를 몇번째 푸는 것인지 나타내는 feature
        # test_prob_count = defaultdict(set)
        # for test_id, assessmentItemID in zip(df["testId"], df["assessmentItemID"]):
        #     test_prob_count[test_id].add(assessmentItemID[-3:])
        # test_prob_count = {key: len(value) for key, value in test_prob_count.items()}
        # le_path = os.path.join(self.args.asset_dir, "test_prob_count.pk")
        # with open(le_path, 'wb') as f:
        #     pickle.dump(test_prob_count, f)

        # -> 저장된 test_id별 문항수를 가져와서 mapping하는 코드입니다.
        le_path = os.path.join(self.args.asset_dir, "test_prob_count.pk")
        with open(le_path, 'rb') as f:
            test_prob_count = pickle.load(f)

        prev_user_id = df["userID"][0]
        user_current_test = dict()
        how_many_times = defaultdict(int)
        new_feature = []

        for cur_user_id, cur_test_id  in zip(df["userID"], df["testId"]):
            if prev_user_id != cur_user_id: # 만약 사용자가 바뀌는 타이밍이라면, dict를 새로 선언해줍니다. 왜냐하면 사용자별로 파악하고 있으니까요!
                prev_user_id = cur_user_id
                user_current_test = dict()
                how_many_times = defaultdict(int)

            if cur_test_id not in user_current_test: # 만약 사용자가 지금 풀고 있는 문제 목록에 없는 거라면,
                user_current_test[cur_test_id] = test_prob_count[cur_test_id] - 1 # 문제 목록에 추가해주면서, 남아있는 문항수도 추가합니다. (-1을 하는 이유는 현재 row를 확인했을 때 한문제를 푼거랑 동일하니까요!)
                new_feature.append(how_many_times[cur_test_id]) # 새로운 feature에 지금 푸는 시험지는 몇번째 푼것인지 기록해줍니다.
                
            else: # 만약 사용자가 지금 풀고 있는 문제라면,
                user_current_test[cur_test_id] -= 1 # 해당 시험지의 남아있는 문항수를 하나 빼줍니다.
                new_feature.append(how_many_times[cur_test_id]) # 새로운 feature에 지금 푸는 시험지는 몇번째 푼것인지 기록해줍니다.

                if user_current_test[cur_test_id] == 0: # 근데 확인해보니 이미 사용자가 다 풀어버려서 남아있는 문항수가 0이라면,
                    how_many_times[cur_test_id] += 1 # 사용자가 해당 id에 대한 시험지 하나를 다 푼것이므로, test_id를 몇번 풀었는지 나타내는 value를 +1 해줍니다.
                    del user_current_test[cur_test_id] # 현재 풀고 있는 문제에서 제거하기 위해 test_id에 해당하는 key값을 삭제합니다.
            
        new_feature = pd.DataFrame(new_feature, columns=['nth_test'])
        df = pd.concat([df, new_feature], axis=1)

        return df

    def __cate_preprocessing(self, df, cate_cols, is_train=True):
        def save_labels(encoder, name):
            le_path = os.path.join(self.args.asset_dir, name + "_classes.npy")
            np.save(le_path, encoder.classes_)

        if not os.path.exists(self.args.asset_dir):
            os.makedirs(self.args.asset_dir)

        for col in cate_cols:
            le = LabelEncoder()
            if is_train:
                # For UNKNOWN class
                a = df[col].unique().tolist() + ["unknown"]
                le.fit(a)
                save_labels(le, col)
            else:
                label_path = os.path.join(self.args.asset_dir, col + "_classes.npy")
                le.classes_ = np.load(label_path)

                df[col] = df[col].astype(str).apply(lambda x: x if x in le.classes_ else "unknown")

            df[col] = df[col].astype(str)
            test = le.transform(df[col])
            df[col] = test

        return df

    def __cont_preprocessing(self, df, cont_cols, is_train=True):
        # TODO

        # continuous feature의 결측치를 0으로 채웁니다.
        for col in df[cont_cols]:
            df[col].fillna(0, inplace=True)

        return df

    def load_data_from_file(self, file_name, is_train=True):
        csv_file_path = os.path.join(self.args.data_dir, file_name)
        df = pd.read_csv(csv_file_path)

        # stratified K-Fold를 위한 key value를 만듭니다.
        self.__make_stratified_key(df)

        # row에 추가할 feature column을 생성합니다.
        df = self.__feature_engineering(df)

        # ========================== !! 이 부분을 만드신 feature의 column name으로 채워주세요 !! ==========================
        # feature engineering을 수행한 뒤, feature로 사용할 column을 적어줍니다.
        cate_cols = ["testId", "assessmentItemID", "KnowledgeTag", "nth_test"]
        cont_cols = ["assessmentItemID_mean"]
        # ==================================================================================================================
        self.args.n_cates = len(cate_cols) # 사용할 categorical feature 개수
        self.args.n_conts = len(cont_cols) # 사용할 continuous feature 개수

        # categorical & continuous feature의 preprocessing을 진행합니다.
        df = self.__cate_preprocessing(df, cate_cols, is_train)
        df = self.__cont_preprocessing(df, cont_cols, is_train)
        df = df.sort_values(by=["userID", "Timestamp"], axis=0)

        # categorical feature의 embedding 차원을 저장 (각 feature가 가지는 총 class의 개수)
        self.args.cate_embs = []
        for cate in cate_cols:
            self.args.cate_embs.append(len(np.load(os.path.join(self.args.asset_dir, f"{cate}_classes.npy"))))
        
        # interaction(이전 문제를 맞췄는지) feature를 위한 추가
        self.args.n_cates += 1
        self.args.cate_embs.append(3)

        # 각 사용자별로 feature를 tuple 형태로 묶은 형태로 만든다.
        columns = ["userID"] + cate_cols + cont_cols + ["answerCode"]
        group = (
            df[columns]
            .groupby("userID")
            .apply(
                lambda r: tuple([r[column].values for column in columns[1:]])
            )
        )
        return group.values

    def load_train_data(self, file_name):
        self.train_data = self.load_data_from_file(file_name)

    def load_test_data(self, file_name):
        self.test_data = self.load_data_from_file(file_name, is_train=False)


class DKTDataset(torch.utils.data.Dataset):
    def __init__(self, data, args):
        self.data = data
        self.args = args

    def __getitem__(self, index):
        row = list(self.data[index])

        # 각 data의 sequence length
        seq_len = len(row[0])

        # max seq len을 고려하여서 이보다 길면 자르고 아닐 경우 그대로 냅둔다
        if seq_len > self.args.max_seq_len:
            for i, col in enumerate(row):
                row[i] = col[-self.args.max_seq_len :]
            mask = np.ones(self.args.max_seq_len, dtype=np.int16)
        else:
            mask = np.zeros(self.args.max_seq_len, dtype=np.int16)
            mask[-seq_len:] = 1

        # mask도 columns 목록에 포함시킴
        row.append(mask)

        # np.array -> torch.tensor 형변환
        for i, col in enumerate(row):
            row[i] = torch.tensor(col)

        return row

    def __len__(self):
        return len(self.data)


def collate(batch):
    col_n = len(batch[0])
    col_list = [[] for _ in range(col_n)]
    max_seq_len = len(batch[0][-1])

    # batch의 값들을 각 column끼리 그룹화
    for row in batch:
        for i, col in enumerate(row):
            pre_padded = torch.zeros(max_seq_len)
            pre_padded[-len(col) :] = col
            col_list[i].append(pre_padded)

    for i, _ in enumerate(col_list):
        col_list[i] = torch.stack(col_list[i])

    return tuple(col_list)


def get_loaders(args, train, valid):

    pin_memory = True
    train_loader, valid_loader = None, None

    if train is not None:
        trainset = DKTDataset(train, args)
        train_loader = torch.utils.data.DataLoader(
            trainset,
            num_workers=args.num_workers,
            shuffle=True,
            batch_size=args.batch_size,
            pin_memory=pin_memory,
            collate_fn=collate,
        )
    if valid is not None:
        valset = DKTDataset(valid, args)
        valid_loader = torch.utils.data.DataLoader(
            valset,
            num_workers=args.num_workers,
            shuffle=False,
            batch_size=args.batch_size,
            pin_memory=pin_memory,
            collate_fn=collate,
        )

    return train_loader, valid_loader
