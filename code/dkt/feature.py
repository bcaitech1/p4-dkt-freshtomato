import math
import pickle
import os
import pandas as pd
from collections import defaultdict

def save_pickle(args, name, object):
    pickle_path = os.path.join(args.asset_dir, name)
    with open(pickle_path, 'wb') as f:
        pickle.dump(object, f)


def load_pickle(args, name):
    pickle_path = os.path.join(args.asset_dir, name)
    with open(pickle_path, 'rb') as f:
        object = pickle.load(f)
    return object


def minyong_feature_engineering(args, df, is_train):
    # (1) 풀고 있는 assessmentItemID의 정답률을 나타내는 feature
    if is_train:
        assessmentItemID_mean_sum = df.groupby(['assessmentItemID'])['answerCode'].agg(['mean', 'sum']).to_dict()
        save_pickle(args, "assessmentItemID_mean_sum.pk", assessmentItemID_mean_sum)

    assessmentItemID_mean_sum = load_pickle(args, "assessmentItemID_mean_sum.pk")
    df["assessmentItemID_mean"] = df.assessmentItemID.map(assessmentItemID_mean_sum['mean'])


    # (2) 풀고 있는 testId의 정답률을 나타내는 feature
    if is_train:
        testId_mean_sum = df.groupby(['testId'])['answerCode'].agg(['mean', 'sum']).to_dict()
        save_pickle(args, "testId_mean_sum.pk", testId_mean_sum)

    testId_mean_sum = load_pickle(args, "testId_mean_sum.pk")
    df["testId_mean"] = df.testId.map(testId_mean_sum['mean'])


    # (3) 풀고 있는 KnowledgeTag의 정답률을 나타내는 feature
    if is_train:
        KnowledgeTag_mean_sum = df.groupby(['KnowledgeTag'])['answerCode'].agg(['mean', 'sum']).to_dict()
        save_pickle(args, "KnowledgeTag_mean_sum.pk", KnowledgeTag_mean_sum)

    KnowledgeTag_mean_sum = load_pickle(args, "KnowledgeTag_mean_sum.pk")
    df["KnowledgeTag_mean"] = df.KnowledgeTag.map(KnowledgeTag_mean_sum['mean'])


    # (4) 해당 시험지를 몇번째 푸는 것인지 나타내는 feature
    if is_train:
        test_prob_count = defaultdict(set)
        for test_id, assessmentItemID in zip(df["testId"], df["assessmentItemID"]):
            test_prob_count[test_id].add(assessmentItemID[-3:])
        test_prob_count = {key: len(value) for key, value in test_prob_count.items()}
        save_pickle(args, "test_prob_count.pk", test_prob_count)

    test_prob_count = load_pickle(args, "test_prob_count.pk")
    prev_user_id = df["userID"][0]
    user_current_test = dict()
    how_many_times = defaultdict(int)
    nth_test_count_feature = []

    for cur_user_id, cur_test_id  in zip(df["userID"], df["testId"]):
        if prev_user_id != cur_user_id: # 만약 사용자가 바뀌는 타이밍이라면, dict를 새로 선언해줍니다. 왜냐하면 사용자별로 파악하고 있으니까요!
            prev_user_id = cur_user_id
            user_current_test = dict()
            how_many_times = defaultdict(int)

        if cur_test_id not in user_current_test: # 만약 사용자가 지금 풀고 있는 문제 목록에 없는 거라면,
            user_current_test[cur_test_id] = test_prob_count[cur_test_id] - 1 # 문제 목록에 추가해주면서, 남아있는 문항수도 추가합니다. (-1을 하는 이유는 현재 row를 확인했을 때 한문제를 푼거랑 동일하니까요!)
            nth_test_count_feature.append(how_many_times[cur_test_id]) # 새로운 feature에 지금 푸는 시험지는 몇번째 푼것인지 기록해줍니다.
            
        else: # 만약 사용자가 지금 풀고 있는 문제라면,
            user_current_test[cur_test_id] -= 1 # 해당 시험지의 남아있는 문항수를 하나 빼줍니다.
            nth_test_count_feature.append(how_many_times[cur_test_id]) # 새로운 feature에 지금 푸는 시험지는 몇번째 푼것인지 기록해줍니다.

            if user_current_test[cur_test_id] == 0: # 근데 확인해보니 이미 사용자가 다 풀어버려서 남아있는 문항수가 0이라면,
                how_many_times[cur_test_id] += 1 # 사용자가 해당 id에 대한 시험지 하나를 다 푼것이므로, test_id를 몇번 풀었는지 나타내는 value를 +1 해줍니다.
                del user_current_test[cur_test_id] # 현재 풀고 있는 문제에서 제거하기 위해 test_id에 해당하는 key값을 삭제합니다.
        
    nth_test_count_feature = pd.DataFrame(nth_test_count_feature, columns=['nth_test'])
    df = pd.concat([df, nth_test_count_feature], axis=1)
    

    # (5) User의 실력을 정답률에 가중치를 적용해서 알아보자
    prev_user_id = -1
    user_weighted_score = []
    weighted_score = 0

    for cur_user_id, answerCode, assessmentItemID_mean in zip(df["userID"], df["answerCode"].shift(1), df["assessmentItemID_mean"].shift(1)):
        if prev_user_id != cur_user_id: # 만약 사용자가 바뀌는 타이밍이라면, list 새로 선언해줍니다. 왜냐하면 사용자별로 파악하고 있으니까요!
            weighted_score = 0
            prev_user_id = cur_user_id
            user_weighted_score.append(weighted_score)
            continue
        
        if answerCode == 1:
            weighted_score += (1 - assessmentItemID_mean)/4
        else:
            weighted_score += (-1) * (assessmentItemID_mean/8)

        user_weighted_score.append(weighted_score)
    
    user_weighted_score = pd.DataFrame(user_weighted_score, columns=['user_weighted_score'])
    df = pd.concat([df, user_weighted_score], axis=1)
    

    # (6) 지금 User가 어떤 조합의 시험지를 푸는 사람인지 (풀수 있는 문제지 대분류[post]는 최대 3가지)
    if is_train:
        user_test_group = df.groupby("userID")["test_pre"].unique()
        user_test_group = user_test_group.reset_index().rename(columns={"index": "userID"})

        def make_test_comb(x):
            comb = ''
            x = sorted(x)
            for test in x:
                comb += test[1]
            return comb

        user_test_group["test_pre_comb"] = user_test_group["test_pre"].apply(make_test_comb)
        user_test_comb = {key: value for key, value in zip(user_test_group["userID"], user_test_group["test_pre_comb"])}
        save_pickle(args, "user_test_comb.pk", user_test_comb)

    user_test_comb = load_pickle(args, "user_test_comb.pk")
    df["user_test_comb"] = df.userID.map(user_test_comb)


    # (7) 한 문제를 푸는데 얼마나 시간이 걸렸는지
    user_current_test = dict()
    time_elapsed = []
    test_start_idx = 0
    df = df.astype({'Timestamp': 'datetime64[ns]'})

    for idx, (cur_user_id, next_user_id, cur_test_id, cur_timestamp, next_timestamp) in enumerate(zip(df["userID"], df["userID"].shift(-1), df["testId"], df["Timestamp"], df["Timestamp"].shift(-1))):
        if cur_user_id != next_user_id: # 만약 사용자가 바뀌는 타이밍이라면, 
            time_elapsed.append(int(sum(time_elapsed[test_start_idx - idx:]) / len(time_elapsed[test_start_idx - idx:]))) # 이전까지 문제를 풀때 걸린 시간의 평균을 구합니다.
            test_start_idx = idx
            user_current_test = dict() # dict를 새로 선언해줍니다. 
            continue
            
        if cur_test_id not in user_current_test: # 만약 사용자가 지금 풀고 있는 문제 목록에 없는 거라면,
            user_current_test[cur_test_id] = test_prob_count[cur_test_id] - 1 # 문제 목록에 추가해주면서, 남아있는 문항수도 추가합니다. (-1을 하는 이유는 현재 row를 확인했을 때 한문제를 푼거랑 동일하니까요!)
            time_elapsed.append((next_timestamp - cur_timestamp).total_seconds()) # 지금 푸는 문제와 다음 문제 사이의 시간 간격을 기록합니다.
            
        else: # 만약 사용자가 지금 풀고 있는 문제라면,
            user_current_test[cur_test_id] -= 1 # 해당 시험지의 남아있는 문항수를 하나 빼줍니다.

            if user_current_test[cur_test_id] == 0: # 근데 확인해보니 해당 시험지에 대해 이미 사용자가 다 풀어버려서 남아있는 문항수가 0이라면,
                del user_current_test[cur_test_id] # 현재 풀고 있는 문제에서 제거하기 위해 test_id에 해당하는 key값을 삭제합니다.

                if len(user_current_test.keys()) == 0: # 확인해보니 아예 사용자가 풀어야할 시험지가 하나도 없다면,
                    time_elapsed.append(int(sum(time_elapsed[test_start_idx - idx:]) / len(time_elapsed[test_start_idx - idx:]))) # 이전까지 문제를 풀때 걸린 시간의 평균을 구합니다.
                    test_start_idx = idx
                
                else: # 아직 풀어야할 시험지가 남아있다면 시험을 보고 있는 중이므로,
                    time_elapsed.append((next_timestamp - cur_timestamp).total_seconds()) # 지금 푸는 문제와 다음 문제 사이의 시간 간격을 기록합니다.
            
            else:
                time_elapsed.append((next_timestamp - cur_timestamp).total_seconds()) # 지금 푸는 문제와 다음 문제 사이의 시간 간격을 기록합니다.

    time_elapsed = pd.DataFrame(time_elapsed, columns=['time_elapsed'])
    def max_limit(x):
        if x > 300:
            x = 300
        return x
    time_elapsed = time_elapsed["time_elapsed"].apply(max_limit)

    df = pd.concat([df, time_elapsed], axis=1)

    # (8) 현재까지 몇문제를 풀었는지
    df["user_problem_cumcount"] = df.groupby("userID")["assessmentItemID"].cumcount()

    return df


def ara_feature_engineering(args, df, is_train):
    # 유저들의 문제 풀이수, 정답수, 정답률을 시간순으로 누적해서 계산
    df['correctAnswerByTime'] = df.groupby('userID')['answerCode'].transform(lambda x: x.cumsum().shift(1))
    df['totalAnswerByTime'] = df.groupby('userID')['answerCode'].cumcount()
    df['userAccByTime'] = df['correctAnswerByTime']/df['totalAnswerByTime']
    return df


def jaehoon_feature_engineering(args, df, is_train):
    '''
    ## 문제번호 (시험지 내에서) (범주형? 연속형?) - 3자리
    # 0 부터 시작하도록 
    df['problem_number_jae'] = df.assessmentItemID.map(lambda x: int(x[-3:])-1)

    ## 시험지의(testID 별) 문제 최대개수가 몇인지 dict
    #  (이를 파악하여 중복 풀이하는 친구들을 걸러낼수있음)
    total_num_prob_in_test = df[['assessmentItemID', 'testId']].drop_duplicates().groupby('testId').size()
    testid_maxlen_dict = total_num_prob_in_test.to_dict()

    # 시험지별 문제번호 최대값
    max_prob_in_test = df.groupby('testId').problem_number_jae.max()
    # 시험 문제번호의 최대값과 시험지 내의 문제개수가 일치하지 않는 시험지의 testId들을 추출
    inconsistent_index = max_prob_in_test[max_prob_in_test +1 != total_num_prob_in_test].index  # max_prob_in_test는 0부터 시작하고, num_prob_in_test는 1부터 시작
    # 문제 max번호와 문제개수가 일치하지 않는 시험지에 해당하는 데이터프레임으로부터 (시험지 Id와 문제 Id) 추출 후 문제 Id의 오름차순 순서대로 정렬
    inconsistent_df = df.loc[df.testId.isin(inconsistent_index),['assessmentItemID', 'testId']].drop_duplicates().sort_values('assessmentItemID')      

    # 순서대로 안 푼 시험지에서의 본래 문제의 순서가 dict에 저장된다    (inconsistent_Itemid_item_dict)
    inconsistent_Itemid_item_dict = {}
    inconsistent_df_group = inconsistent_df.groupby('testId')
    # 순서대로 안 푼 시험지들에 대하여 for문
    for key in inconsistent_df_group.groups:
        for i, (k,_) in enumerate(inconsistent_df_group.get_group(key).values):
            inconsistent_Itemid_item_dict[k] = i
    
    # origin_problem_order : 한 시험지 내에서 해당 문제 Id의 본래 순서 (중간에 비어있는 문제는 없는 문제로 생각하여 다시 순서를 매김)
    # ex) A080096003,A080096001,A080096005,A080096006,A080096007,A080096008,A080096002 
    # 문제번호 : 3, 1, 5, 6, 7, 8, 2 => 문제번호 중 4번이 빠져있음
    # 따라서 4번을 제외하고 문제번호를 본래순서에 대응시키면 => (1:0,2:1,3:2,5:3,6:4,7:5,8:6)
    # 변수생성 => origin_problem_order : (2,0,3,4,5,6,1)
    # 시험 문제번호의 최대값과 시험지 내의 문제개수가 일치하지 않는 시험지의 testId들을 추출        
    df['origin_problem_order'] =  df.assessmentItemID.map(lambda x: int(inconsistent_Itemid_item_dict[x]) if x in inconsistent_Itemid_item_dict else int(x[-3:]) - 1) # 0부터 시작하도록 -1 해줌.
    '''

    return df


def yuura_feature_engineering(args, df, is_train):
    # 유저별 전체 정답률을 나타내는 수치
    if is_train:
        user_total_accs = df.groupby(['userID'])['answerCode'].agg(['mean', 'sum']).to_dict()
        save_pickle(args, "user_total_accs.pk", user_total_accs)

    user_total_accs = load_pickle(args, "user_total_accs.pk")
    df["user_total_accs"] = df.userID.map(user_total_accs['mean'])

    return df