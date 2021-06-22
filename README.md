# ⭕❌ _**Deep Knowledge Tracing**_
<p align='center'>
<img src=https://user-images.githubusercontent.com/63627253/119375251-8f2c8100-bcf5-11eb-89b7-583434f96171.gif width=75%>
</p>

</br>

## **Task Description**

#### Task
학생이 푼 문제 리스트와 정답 여부가 담긴 데이터를 바탕으로 **학생의 지식상태를 추적**하고, 미래에 학생이 특정 문제를 맞출지 틀릴지를 예측합니다.
이를 통해 학생에게 **개인 맞춤형 교육**을 제공합니다.
#### Metric
**AUROC**, Accuracy

</br>

## Pipeline
![freshtomato_pipeline](https://user-images.githubusercontent.com/46676700/122864044-da6e9980-d35e-11eb-8cc7-75a24b1d3771.png)

## **Command Line Interface**

## **1️⃣ Train/Valid Ratio 9:1 (random split)**

#### Train Phase
```python
>>> cd code
>>> python train.py --wandb_project_name [PROJECT_NAME] --wandb_run_name [RUN_NAME] --model [MODEL]
```

#### Inference Phase
```python
>>> cd code
>>> python inference.py --wandb_run_name [RUN_NAME] --model [MODEL]
```

## **2️⃣ K-fold**

#### Train Phase
```python
>>> cd code
>>> python train_kfold.py --wandb_project_name [PROJECT_NAME] --wandb_run_name [RUN_NAME] --model [MODEL] --kfold 10
```

#### Inference Phase
```python
>>> cd code
>>> python inference_kfold.py --wandb_run_name [RUN_NAME] --model [MODEL] --kfold 10
```

## **3️⃣ Stratified K-fold**

#### Train Phase
```python
>>> cd code
>>> python train_stfkfold.py --wandb_project_name [PROJECT_NAME] --wandb_run_name [RUN_NAME] --model [MODEL] --kfold 10
```

#### Inference Phase
```python
>>> cd code
>>> python inference_kfold.py --wandb_run_name [RUN_NAME] --model [MODEL] --kfold 10
```

</br>

## **Implemented models**
- LSTM (lstm)
- LSTM + Attention (lstmattn)
- Bert (bert)
- GRUATTN (gruattn)
- ATTNGRU (attngru)
- Saint (saint)
- Saint_custom (saintcustom)
- LastQuery (lastquery)
- BaseCNN (cnn)
- DeepCNN (deepcnn)


## Directory structure

```bash
├── README.md                 - 리드미 파일
│
├── requirements.md           - 필요한 library
|
├── dkt/                      - DL팀 utils 파일
│   ├── criterion.py           
│   ├── custom_model.py           
│   │── dataloader.py         
│   │── feature.py       
│   │── model.py
│   │── optimizer.py         
│   │── scheduler.py       
│   │── trainer.py                  
|   └── utils.py
|
├── code/                     - DL팀 코드 폴더
│   ├── args.py           
│   ├── inference.py           
│   │── inference_kfold.py         
│   │── train.py       
│   │── train_kfold.py                  
|   └── train_stfkfold.py
│
├── notebook_pycaret          - ML팀 코드 폴더
|   ├── Add_Feature_with_Groupby.ipynb
|   ├── get_logs연습해보기.ipynb
|   ├── kaggle_riid_전처리.ipynb
|   ├── LGBM_Valid원하는대로구축성공.ipynb
│   ├── Optuna_LightGBM_문제시간간격후처리X.ipynb
│   ├── output파일_best랑비교해보기.ipynb
|   └── PermutationImportance.ipynb
|
└── notebooks              
    ├── baseline.ipynb
    ├── EDA_Minyong.ipynb            
    ├── EDA-arabae.ipynb
    ├── hard_and_soft_ensemble.ipynb
    ├── output_confidence.ipynb
    └── Riiid_pretrain.ipynb

```


# 🍅 _Members_

**강민용 T1001**
**[[Github](https://github.com/MignonDeveloper)]  [[Blog](https://mignontraveler.tistory.com/)]**

- EDA
- GRU + Attention & SAINT Modeling
- User Data Augmentation, Pseudo Labeling
- Deep Learning Code 개선
- DKT, DKT+, DKVMN 논문 정리 및 공유

> #️⃣**463번 실험**  #️⃣**일단 공유해** #️⃣**어피치**

---

**김진현 T1248**
**[[Github](https://github.com/KimJinHye0n)]  [Blog]**

- EDA
- Saint, Saint+ Modeling
- Feature Searching (문항별 난이도 / KnowledgeTag)
- Ensemble (Hard + Soft voting)

---
**문재훈 T1058**

**[[Github](https://github.com/MoonJaeHoon)]  [[Blog](https://moonjaehoon.github.io/)]**

- ML (with Customized Optuna & Pycaret)
- 검증전략 (HoldOut Set, Customized CV)
- Efficient Feature Engineering (with Pandas method)
- Feature Selection (with Permutation Importance)
- Riiid Dataset Processing for Pre-training

---

**배아라 T1084**

**[[Github](https://github.com/arabae)]  [[Blog](https://arabae.github.io/)]**

- LSTM, LSTM+Attention, BERT, CNN, Last Query, SAINT 등
다양한 모델 구현 및 실험
- User별 Feature Engineering
- Deep Learning Code 개선
- Riiid 데이터를 활용한 pre-train 시도
- Ensemble (soft-voting, weighted soft-voting)

---


**이정현 T1160**

**[[Github](https://github.com/gvsteve24)]  [[Blog](https://dipndeep.tistory.com/)]**

- EDA
- RNN계열(LSTM, LSTM+Attention) 모델 실험
- DKT, DKVMN 논문정리
- Riiid Competition Data Analysis for Transfer Learning
- 회의 내용 정리

---

**최유라 T1212**    

**[[Github](https://github.com/Yuuraa)]  [[Blog](https://velog.io/@yoorachoi)]**

- EDA - 학습 데이터, 테스트 데이터 분포 파악
- Feature engineering - 풀이 시간, 정확도 평균 feature 추가
- ML 모델 학습 - LightGBM, XGBoost, CatBoost
- Validation set 찾기
