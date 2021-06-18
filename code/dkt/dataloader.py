import os
from datetime import datetime
import pandas as pd
import random
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, OneHotEncoder, LabelBinarizer, StandardScaler
import numpy as np
import torch

from .feature import (minyong_feature_engineering,
                      ara_feature_engineering,
                      jaehoon_feature_engineering,
                      yuura_feature_engineering)


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

    def __data_augmentation(self, df):
        total_row = df.shape[0] - 1 # 전체 row 개수
        prev_user = df.userID[total_row]
        userid = [] # 새롭게 만들 userid
        user_count = 10000 # 새로운 user를 만들 때 id를 10000부터 부여합니다.
        row_count = 0

        # dataframe의 역순으로 구해줍니다. 앞에서부터 자르기보다는 뒤에서부터 자르는게 더 좋은 정보를 살릴 수 있을 것 같아요.
        for i in range(total_row, -1, -1):
            cur_user = df.userID[i]
            row_count += 1

            if prev_user != cur_user: # 사용자가 아예 바뀌면 새로운 사용자로 인식합니다.
                user_count += 1
                prev_user = cur_user
                row_count = 1
            
            elif row_count % self.args.max_seq_len == 0: # 만약 row count가 max_seq_len가 되면
                user_count += 1 # 새로운 사용자로 인식합니다.
                row_count = 0
            
            userid.append(user_count)
        
        # 새로운 userID를 제공하기 위해 기존에 가지고 있던 userID column을 drop 합니다.
        df.drop(columns=["userID"], inplace=True)

        # 역순으로 userID를 구했으므로 reverse 시킵니다.
        userid.reverse()

        new_user_id = pd.DataFrame(userid, columns=['userID']) # 새롭게 만든 userid를
        df = pd.concat([df, new_user_id], axis=1)  # 기존 dataframe에 concat 해줍니다.

        return df

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
        stratified = df.copy()
        result = []

        prev_userID = stratified.userID[0]
        for userID, answerCode in zip(stratified.userID, stratified.answerCode.shift(1)):
            if prev_userID != userID:
                result.append(answerCode)
                prev_userID = userID
        
        result.append(stratified.answerCode.loc[stratified.shape[0] -1])
        self.user_stratified_key = result

        
    def __split_id_number(self, df):
        def assessmentItemID2problem_number(x):
            return x[-3:]
        def testId2test_pre(x):
            return x[1:4]
        def testId2test_post(x):
            return x[-3:]

        df['problem_number'] = df.assessmentItemID.map(assessmentItemID2problem_number)
        df['test_pre'] = df.testId.map(testId2test_pre)
        df['test_post'] = df.testId.map(testId2test_post)

        return df

    def __cate_preprocessing(self, df, cate_cols, is_train=True):
        def save_labels(encoder, name):
            le_path = os.path.join(self.args.asset_dir, name + "_classes.npy")
            np.save(le_path, encoder.classes_)

        os.makedirs(self.args.asset_dir, exist_ok=True)
        
        '''
        # Label Encoder
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
        
        '''
        
        # one hot encoding
        final_cate_cols = []
        for col in cate_cols:
            ohe = LabelBinarizer()
            ohe.fit(df[col])
            transformed = ohe.transform(df[col])
            ohe_df = pd.DataFrame(transformed, columns=ohe.classes_)
            ohe_df = ohe_df.add_prefix(col)

            df = pd.concat([df, ohe_df], axis=1)
            final_cate_cols.extend(list(ohe_df.columns))

        return df, final_cate_cols

    def __cont_preprocessing(self, df, cont_cols, is_train=True):

        # scaler를 통해 전처리 해줍니다.
        # scaler_col = cont_cols
        # scaler = StandardScaler()
        
        scaler_col = ["time_elapsed", "user_problem_cumcount"] # "user_weighted_score", "correctAnswerByTime", "totalAnswerByTime"
        scaler = MinMaxScaler()
        for col in scaler_col:
            scaler.fit(pd.DataFrame(df[col]))
            transformed = scaler.transform(pd.DataFrame(df[col]))
            df[col] = transformed
        
        # continuous feature의 결측치를 0으로 채웁니다.
        for col in df[cont_cols]:
            df[col].fillna(0, inplace=True)

        return df
        
    def load_data_from_file(self, file_name, is_train=True):
        
        csv_file_path = os.path.join(self.args.data_dir, file_name)
        df = pd.read_csv(csv_file_path)
        df = df.sort_values(by=["userID", "Timestamp"], axis=0)

        if is_train and self.args.aug:
            # data augmentation
            df = self.__data_augmentation(df)

        # userID, 시간 순으로 dataframe 정렬

        if is_train:
            # stratified K-Fold를 위한 key value를 만듭니다.
            self.__make_stratified_key(df)

        # feature engineering할 때 더 편리하게 사용할 수 있도록 testid와 assessmentItemId에서 number를 추출합니다.
        df = self.__split_id_number(df)

        # row에 추가할 feature column을 생성합니다.
        df = minyong_feature_engineering(self.args, df, is_train)
        df = ara_feature_engineering(self.args, df, is_train)
        df = jaehoon_feature_engineering(self.args, df, is_train)
        df = yuura_feature_engineering(self.args, df, is_train)

        # ========================== !! 이 부분을 만드신 feature의 column name으로 채워주세요 !! ==========================
        # feature engineering을 수행한 뒤, feature로 사용할 column을 적어줍니다.
        cate_cols = ["problem_number", "nth_test", "time_elapsed_cat"]
        cont_cols = ["assessmentItemID_mean", "KnowledgeTag_mean", "testId_mean", "time_elapsed"]
        
        # ==================================================================================================================
        
        # categorical & continuous feature의 preprocessing을 진행합니다.
        df = self.__cont_preprocessing(df, cont_cols, is_train)
        # df = self.__cate_preprocessing(df, cate_cols, is_train) # LabelEncoder
        df, cate_cols = self.__cate_preprocessing(df, cate_cols, is_train) # OneHotEncoder
        
        self.args.n_cates = len(cate_cols) # 사용할 categorical feature 개수
        self.args.n_conts = len(cont_cols) # 사용할 continuous feature 개수

        # categorical feature의 embedding 차원을 저장 (각 feature가 가지는 총 class의 개수)
        self.args.cate_embs = []
        for cate in cate_cols:
            # self.args.cate_embs.append(len(np.load(os.path.join(self.args.asset_dir, f"{cate}_classes.npy"))))
            self.args.cate_embs.append(3)
        
        # interaction(이전 문제를 맞췄는지) feature를 위한 추가
        self.args.n_cates += 1
        self.args.cate_embs.append(3)

        # 각 사용자별로 feature를 tuple 형태로 묶은 형태로 만든다.
        columns = ["userID"] + cate_cols + cont_cols + ["answerCode"]

        # check all features
        print(df[columns].head(20))
        print(df[columns].tail(20))
        print(df[columns].describe())
        # exit(1)

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
