import os
import urllib.request as request
import zipfile
from pathlib import Path
from obesity import logger
from obesity.utils import get_size
from obesity.entity import DataIngestionConfig

class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config


    
    def download_file(self):
        try: 
            dataset_url = self.config.source_URL
            zip_download_dir = self.config.local_data_file
            os.makedirs("artifacts/data_ingestion", exist_ok=True)
            logger.info(f"Downloading data from {dataset_url} into file {zip_download_dir}")
            
            
            if dataset_url.split('/')[3] =='datasets':
                if not os.path.isfile("artifacts/data_ingestion/data.zip"):
                    os.system("kaggle datasets download -d  "+dataset_url.split('/')[4]+"/"+dataset_url.split('/')[5])
                    
                    if os.path.isfile("archive.zip"):
                        os.system("mv archive.zip artifacts/data_ingestion/data.zip")
                    else:
                        os.rename(dataset_url.split('/')[5]+".zip" ,"artifacts/data_ingestion/data.zip")
                        
            else :
                if not os.path.isfile("artifacts/data_ingestion/data.zip"):
                    os.system("kaggle datasets download -d  "+dataset_url.split('/')[4])
                    os.system("mv "+dataset_url.split('/')[4]+".zip"+" artifacts/data_ingestion")

            logger.info(f"Downloaded data from {dataset_url} into file {zip_download_dir}")
        except Exception as e:
            raise e


    def extract_zip_file(self):
        """
        zip_file_path: str
        Extracts the zip file into the data directory
        Function returns None
        """
        unzip_path = self.config.unzip_dir
        os.makedirs(unzip_path, exist_ok=True)
        with zipfile.ZipFile(self.config.local_data_file, 'r') as zip_ref:
            zip_ref.extractall(unzip_path)
  
