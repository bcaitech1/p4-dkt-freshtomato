import os
from datetime import datetime
import time
import tqdm
import pandas as pd
import random
from sklearn.preprocessing import LabelEncoder
import numpy as np
import torch


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
    

    def __save_labels(self, encoder, name):
        le_path = os.path.join(self.args.asset_dir, name + "_classes.npy")
        np.save(le_path, encoder.classes_)

    def __preprocessing(self, df, is_train=True):
        cate_cols = ["assessmentItemID", "testId", "KnowledgeTag", "correctRate"]

        if not os.path.exists(self.args.asset_dir):
            os.makedirs(self.args.asset_dir)

        for col in cate_cols:
            le = LabelEncoder()
            if is_train:
                # For UNKNOWN class
                a = df[col].unique().tolist() + ["unknown"]
                le.fit(a)
                self.__save_labels(le, col)
            else:
                label_path = os.path.join(self.args.asset_dir, col + "_classes.npy")
                le.classes_ = np.load(label_path)

                df[col] = df[col].apply(lambda x: x if x in le.classes_ else "unknown")

            # 모든 컬럼이 범주형이라고 가정
            df[col] = df[col].astype(str)
            test = le.transform(df[col])
            df[col] = test

        def convert_time(s):
            timestamp = time.mktime(
                datetime.strptime(s, "%Y-%m-%d %H:%M:%S").timetuple()
            )
            return int(timestamp)

        df["Timestamp"] = df["Timestamp"].apply(convert_time)

        return df

    def __feature_engineering(self, df):
        # TODO

        # --- user별 Test의 정답률 feature 추가
        total_count, answer_count = 1, 0
        pr_user_id, pr_test_id, pr_answer_code = 0, 'A060000001', 1
        user_test_correct_ratio = []

        for user_id, test_id, answer_code in zip(df['userID'], df['testId'], df['answerCode']):
            if (user_id != pr_user_id) or (test_id != pr_test_id): # 다른 user 시작 or 다른 시험지 시작
                total_count, answer_count = 1, 0
                pr_user_id = user_id
                pr_test_id = test_id
            if answer_code == 1:
                answer_count += 1
            user_test_correct_ratio.append(answer_count/total_count)
            total_count += 1
        
        user_test_correct_ratio = pd.DataFrame(user_test_correct_ratio, columns=['correctRate'])
        df = df.join(user_test_correct_ratio)

        # 유저들의 문제 풀이수, 정답 수, 정답률을 시간순으로 누적해서 계산
        df['correctAnswer'] = df.groupby('userID')['answerCode'].transform(lambda x: x.cumsum().shift(1))
        df['totalAnswer'] = df.groupby('userID')['answerCode'].cumcount()
        df['userAcc'] = df['correctAnswer']/df['totalAnswer']
        
        # testId와 KnowledgeTag의 전체 정답률은 한번에 계산
        # 아래 데이터는 제출용 데이터셋에 대해서도 재사용
        correct_t = df.groupby(['testId'])['answerCode'].agg(['mean'])
        correct_t.columns = ["test_mean"]
        correct_k = df.groupby(['KnowledgeTag'])['answerCode'].agg(['mean'])
        correct_k.columns = ["tag_mean"]
        
        df = pd.merge(df, correct_t, on=['testId'], how="left")
        df = pd.merge(df, correct_k, on=['KnowledgeTag'], how="left")
        return df

    def __make_stratified_key(self, df):
        def assessmentItemID2item(x):
            return x[-3:]
        def assessmentItemID2test(x):
            return x[1:-3]
        def test2test_pre(x):
            return x[:3]    
        def test2test_post(x):
            return x[3:]

        stratified = df.copy()
        
        stratified['problem_number'] = stratified.assessmentItemID.map(assessmentItemID2item)
        stratified['test'] = stratified.assessmentItemID.map(assessmentItemID2test)
        stratified['test_pre'] = stratified.test.map(test2test_pre)
        stratified['test_post'] = stratified.test.map(test2test_post)
        stratified.drop('test', axis=1, inplace=True)

        user_test_group = stratified.groupby("userID")

        result = []

        for key, group in user_test_group:
            test_pres = np.sort(group.test_pre.unique())
            key = ''
            for test_pre in test_pres:
                key += test_pre[1]

            result.append(key)
        
        self.user_stratified_key = result


    def load_data_from_file(self, file_name, is_train=True):
        csv_file_path = os.path.join(self.args.data_dir, file_name)

        df = pd.read_csv(csv_file_path)
        # self.__make_stratified_key(df)
        df = self.__feature_engineering(df)
        df = self.__preprocessing(df, is_train)

        df = df.sort_values(by=["userID", "Timestamp"], axis=0)
        # =============================== !!!!여기만 주의하자!!!! ===============================
        columns = ["userID", "testId", "assessmentItemID", "KnowledgeTag", 
                   "correctRate", "correctAnswer", "totalAnswer", "userAcc", "test_mean", "tag_mean", "answerCode"]
        args.n_cates = 3
        args.n_cons = 6
        # ========================================================================================
        args.cate_embs = []
        for c in columns[1: args.n_cates+1]:
            args.cate_embs.append(len(np.load(os.path.join(self.args.asset_dir, f"{c}_classes.npy"))))
        
        args.n_cates += 1
        args.cate_embs.append(3)

        for col in df.columns:
            df[col].fillna(0, inplace=True)
        
        group = (
            df[columns]
            .groupby("userID")
            .apply(
                lambda r: tuple([r[c].values for c in columns[1:]])
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
        row = self.data[index]

        # 각 data의 sequence length
        seq_len = len(row[0])

        test, question, tag, correct = row

        cate_cols = [test, question, tag, rate, correct]

        # max seq len을 고려하여서 이보다 길면 자르고 아닐 경우 그대로 냅둔다
        if seq_len > self.args.max_seq_len:
            for i, col in enumerate(cate_cols):
                cate_cols[i] = col[-self.args.max_seq_len :]
            mask = np.ones(self.args.max_seq_len, dtype=np.int16)
        else:
            mask = np.zeros(self.args.max_seq_len, dtype=np.int16)
            mask[-seq_len:] = 1

        # mask도 columns 목록에 포함시킴
        cate_cols.append(mask)

        # np.array -> torch.tensor 형변환
        for i, col in enumerate(cate_cols):
            cate_cols[i] = torch.tensor(col)

        return cate_cols

    def __len__(self):
        return len(self.data)


from torch.nn.utils.rnn import pad_sequence


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
