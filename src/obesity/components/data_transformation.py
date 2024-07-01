from obesity.entity import DataTransformationConfig
import os
import pandas as pd
from obesity import logger
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from obesity.utils import save_bin,save_json
import json
from pathlib import Path
class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config

    
    ## Note: You can add different data transformation techniques such as Scaler, PCA and all
    #You can perform all kinds of EDA in ML cycle here before passing this data to the model

    def replace_nan_num(self,dataset):
        numerical_with_nan=[feature for feature in dataset.columns if dataset[feature].isnull().sum()>0 and dataset[feature].dtypes!='O']

        for feature in numerical_with_nan:
            ## We will replace by using median since there are outliers
            median_value=dataset[feature].median()
            
            ## create a new feature to capture nan values
            #dataset[feature+'nan']=np.where(dataset[feature].isnull(),1,0)
            dataset[feature].fillna(median_value,inplace=True)
        
        logger.info("Replaceed missing dataset with median")
        return dataset
    
    def correlation(self,dataset, threshold):
        col_corr = set()  # Set of all the names of correlated columns
        corr_matrix = dataset.corr()
        for i in range(len(corr_matrix.columns)):
            for j in range(i):
                if abs(corr_matrix.iloc[i, j]) > threshold: # we are interested in absolute coeff value
                    colname = corr_matrix.columns[i]  # getting the name of column
                    col_corr.add(colname)
        return col_corr

    def scaling(self,dataset):
        '''Scaling Feature'''
        
        scaling_feature=[feature for feature in dataset.columns if feature not in ['NObeyesdad'] ]
        scaler=MinMaxScaler()
        scaler.fit(dataset[scaling_feature])
        data = pd.concat([dataset[['NObeyesdad']].reset_index(drop=True),
                    pd.DataFrame(scaler.transform(dataset[scaling_feature]), columns=scaling_feature)],
                    axis=1)
        logger.info("Completed scaling dataset")
        return(data)
        
    def train_test_spliting(self,data):
        #data = pd.read_csv(self.config.data_path)
        
        # Split the data into training and test sets. (0.75, 0.25) split.
        train, test = train_test_split(data)

        train.to_csv(os.path.join(self.config.root_dir, "train.csv"),index = False)
        test.to_csv(os.path.join(self.config.root_dir, "test.csv"),index = False)

        logger.info("Splited data into training and test sets")
        logger.info(train.shape)
        logger.info(test.shape)

        print(train.shape)
        print(test.shape)
    
    
    def categorical_labelEncoding(self,dataset):
        categorical_features=[feature for feature in dataset.columns if dataset[feature].dtypes=='O']
        le_cat_name=dict()
        cat_le = LabelEncoder()
        for i in categorical_features:
            dataset[i]=cat_le.fit_transform(dataset[i])
            le_cat_name[i] = dict(zip(cat_le.classes_, cat_le.transform(cat_le.classes_)))
        
        logger.info(le_cat_name)
        #save_bin(cat_le, (os.path.join(self.config.root_dir, self.config.categorical_feature_path)))
        print(Path(os.path.join(self.config.root_dir, self.config.categorical_json_path)))
        save_json(Path(os.path.join(self.config.root_dir, self.config.categorical_json_path)),le_cat_name)  
        return dataset
    
    
    def transformation(self):
        data = pd.read_csv(self.config.data_path)
        logger.info(data.shape)
        logger.info("Converted csv data to DataFrame")
        #encode the target data
        
        # le=LabelEncoder()
        # data['cls']= le.fit_transform(data['class'])
        # logger.info("Encoded the dependent variable ")
        
        data = self.categorical_labelEncoding(data)
        
        X=data.drop(['NObeyesdad'],axis=1)
        y=data[['NObeyesdad']]
        
        
        #remove the correlated features
        # corr_features = self.correlation(X, 0.9)
        # print("corr_features"+ str(corr_features))
        # data=data.drop(corr_features,axis=1)
        # logger.info("Droped highly correlated features")
        
        #scaling the dependent variable
        data=self.scaling(data)
        self.train_test_spliting(data)


        
