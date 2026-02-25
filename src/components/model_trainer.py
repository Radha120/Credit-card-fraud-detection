import os
import sys
from dataclasses import dataclass
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from src.exception import customException
from src.logger import logging
from src.utils import save_object,evaluate_model
@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join("artifacts","model.pl")
class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()
    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("Split training and test input data")
            x_train, y_train, x_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )
            model =XGBClassifier()
            model_report:dict=evaluate_model(x_train=x_train,y_train=y_train,x_test=x_test,y_test=y_test,model=model)
            model.fit(x_train, y_train)
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=model
            )
            predicted=model.predict(x_test)
            Accuracy_score=accuracy_score(y_test,predicted)
            return Accuracy_score

        except Exception as e:
            raise customException(e,sys)