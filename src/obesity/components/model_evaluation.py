from obesity.entity import ModelEvaluationConfig
from obesity.utils import save_json
import os
import pandas as pd
from urllib.parse import urlparse
import mlflow
import mlflow.sklearn
import numpy as np
import joblib
from pathlib import Path
from sklearn.metrics import (accuracy_score,classification_report,confusion_matrix,
                             precision_score,recall_score,f1_score,roc_auc_score,mean_squared_error,mean_absolute_error,r2_score)

class ModelEvaluation:
    def __init__(self, config: ModelEvaluationConfig):
        self.config = config

    
    
    
    def classification_performace_matric(self,actual, pred,avg):
        acc=accuracy_score(actual,pred)
        f1=f1_score(actual,pred,average=avg)
        precission=precision_score(actual,pred,average=avg)
        recall=recall_score(actual,pred,average=avg)
        return acc,f1,precission,recall


    def log_into_mlflow(self):
        train_data = pd.read_csv(self.config.train_data_path)
        test_data = pd.read_csv(self.config.test_data_path)
        
        model = joblib.load(self.config.model_path)


        train_x = train_data.drop([self.config.target_column], axis=1)
        train_y = train_data[[self.config.target_column]]
        test_x = test_data.drop([self.config.target_column], axis=1)
        test_y = test_data[[self.config.target_column]]


        mlflow.set_registry_uri(self.config.mlflow_uri)
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
        
        # mlflow.set_tracking_uri("https://dagshub.com/SunilKumar-ugra/ObesityRisk-ML-End2End.mlflow")
        # mlflow.set_experiment("ObesityRisk")

        with mlflow.start_run(run_name='Random54'):

            y_train_pred=model.predict(train_x)
            y_test_pred=model.predict(test_x)

             #training   performance
            (trn_acc,trn_f1,trn_precission,trn_recall)=self.classification_performace_matric(train_y,y_train_pred,'weighted')
            
            #testing   performance
            (tst_acc,tst_f1,tst_precission,tst_recall)=self.classification_performace_matric(test_y,y_test_pred,'weighted')
            
            # Saving metrics as local
            # scores = {"rmse": rmse, "mae": mae, "r2": r2}
            scores = {"Accuracy":tst_acc,"F1 Score":tst_f1,"Precission":tst_precission,"Recall":tst_recall,}
            save_json(path=Path(self.config.metric_file_name), data=scores)

            
            #train performance log
            
            mlflow.log_metric("Train Accuracy", trn_acc)
            mlflow.log_metric("Train F1 Score",trn_f1)
            mlflow.log_metric("Train Precision",trn_precission)
            mlflow.log_metric("Train Recall",trn_recall)
            
            # test performance log
            
            mlflow.log_metric("Test Accuracy", tst_acc)
            mlflow.log_metric("Test F1 Score",tst_f1)
            mlflow.log_metric("Test Precision",tst_precission)
            mlflow.log_metric("Test Recall",tst_recall)
            
            # Model log
            # mlflow.sklearn.log_model(model, str(model_name)+"_model")


            # Model registry does not work with file store
            if tracking_url_type_store != "file":

                # Register the model
                # There are other ways to use the Model Registry, which depends on the use case,
                # please refer to the doc for more information:
                # https://mlflow.org/docs/latest/model-registry.html#api-workflow
                mlflow.sklearn.log_model(model, "model", registered_model_name="random")
            else:
                mlflow.sklearn.log_model(model, "model")

    
