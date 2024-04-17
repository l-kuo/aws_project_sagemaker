import json
import boto3
import os
import pandas as pd
import numpy as np
import argparse
 

# from sklearn.preprocessing import MinMaxScaler, LabelEncoder
# from sklearn.model_selection import train_test_split

import warnings
warnings.filterwarnings('ignore')



def lambda_handler(event, context):
    s3 = boto3.client('s3')
    
    # response = s3.list_buckets()

    # for bucket in response['Buckets']:
    #     print(f'Bucket Name: {bucket["Name"]}')

    # paginator = s3.get_paginator('list_objects_v2')
    # for page in paginator.paginate(Bucket='sagemaker-us-east-1-637423267513'):
    #     for obj in page.get('Contents', []):
    #         print(f'Key: {obj["Key"]}')


    # Get bucket name and key from the Lambda event
    bucket = event['Records'][0]['s3']['bucket']['name']
    key = event['Records'][0]['s3']['object']['key']

    # Download the file from S3
    download_path = '/tmp/{}'.format(os.path.basename(key))
    s3.download_file(bucket, key, download_path)
    
    # Process the file
    # processed_data = process_data(download_path)  # Implement this function
    X_train , X_test , y_train , y_test = preprocess(download_path)
    
    # Combine X_train and y_train
    train_data = np.column_stack((X_train, y_train))

    # Combine X_test and y_test
    test_data = np.column_stack((X_test, y_test))
    
    train_filename = os.path.join(os.path.dirname(download_path), 'train_data.csv')
    test_filename = os.path.join(os.path.dirname(download_path), 'test_data.csv')

    # Save train_data to a CSV file
    pd.DataFrame(train_data).to_csv(train_filename, header=False, index=False)
    
    # Save test_data to a CSV file
    pd.DataFrame(test_data).to_csv(test_filename, header=False, index=False)

    try:
        os.remove(os.path.join(os.path.dirname(download_path), 'raw_data_all.csv'))
        print(f"Successfully deleted raw_data_all.csv")
    except Exception as e:
        print(f"Error deleting iris.csv: {e}")

    # Save the processed file back to S3
    output_key = event['Records'][0]['s3']['object']['key']
    print(output_key)
    s3.upload_file(train_filename, bucket, 'preprocess/data/train_data.csv')
    s3.upload_file(test_filename, bucket, 'preprocess/data/test_data.csv')

    # Continue with the rest of your Lambda function
    return {
        'statusCode': 200,
        'body': 'File processed and deleted successfully'
    }

def preprocess(file_string):
    
    df = pd.read_csv(file_string)
    df.columns = ["longitude", "latitude", "housingMedianAge", "totalRooms", "totalBedrooms", "population", "households", "medianIncome", "medianHouseValue"]
    
    scaled_features = ["longitude", "latitude", "housingMedianAge", "totalRooms", "totalBedrooms", "population", "households", 'medianIncome']
    X_norm = my_standard_scaler(df[scaled_features])
    y = MyLabelEncoder().fit_transform(df['medianHouseValue'])
    X_train , X_test , y_train , y_test = train_test_split(X_norm,y,test_size=.2, shuffle=True)
    return X_train , X_test , y_train , y_test


class MyLabelEncoder:
    def __init__(self):
        self.class_mapping = {}

    def fit(self, data):
        unique_classes = np.unique(data)
        self.class_mapping = {label: idx for idx, label in enumerate(unique_classes)}
    
    def transform(self, data):
        return np.vectorize(lambda x: self.class_mapping[x])(data)
    
    def fit_transform(self, data):
        self.fit(data)
        return self.transform(data)
        
def my_standard_scaler(data):
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    return (data - mean) / std
    

def train_test_split(X, y, test_size=0.25, random_state=None, shuffle=True):
    # Set random seed for reproducibility
    if random_state is not None:
        np.random.seed(random_state)
    
    # Generate array of indices
    indices = np.arange(X.shape[0])
    
    if shuffle:
        np.random.shuffle(indices)
    
    # Calculate the number of test samples
    test_size = int(test_size * X.shape[0]) if isinstance(test_size, float) else test_size
    
    # Split indices
    train_indices = indices[:-test_size]
    test_indices = indices[-test_size:]
    
    # Split DataFrames and Series or NumPy arrays
    X_train = X.iloc[train_indices] if hasattr(X, 'iloc') else X[train_indices]
    X_test = X.iloc[test_indices] if hasattr(X, 'iloc') else X[test_indices]
    y_train = y.iloc[train_indices] if hasattr(y, 'iloc') else y[train_indices]
    y_test = y.iloc[test_indices] if hasattr(y, 'iloc') else y[test_indices]
    
    return X_train, X_test, y_train, y_test
