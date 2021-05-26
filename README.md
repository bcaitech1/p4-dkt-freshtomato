# ⭕❌ **Deep Knowledge Tracing**
![ezgif-1-dfc7d299ff28](https://user-images.githubusercontent.com/63627253/119375251-8f2c8100-bcf5-11eb-89b7-583434f96171.gif)

</br>

## **Task Description**

- Task: 학생이 맞춘 문제의 이력을 보고, 다음 문제를 맞출지 틀릴지를 예측합니다.
- Metric: AUROC, Accuracy
  
</br>

## **Command Line Interface**

### **Train Phase**
```python
>>> cd code
>>> python train.py --wandb_project_name [PROJECT_NAME] --wandb_run_name [RUN_NAME] --model [MODEL]
>>> python train_kfold.py --wandb_project_name [PROJECT_NAME] --wandb_run_name [RUN_NAME] --model [MODEL] --kfold 5
```

### **Inference Phase**
```python
>>> cd code
>>> python inference.py --wandb_run_name [RUN_NAME] --model [MODEL]
```

</br>
  
## **Implemented models**
- LSTM (lstm)
- LSTM + Attention (lstmattn)
- Bert encoder (bert)
