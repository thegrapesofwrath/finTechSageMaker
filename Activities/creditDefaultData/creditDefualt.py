#%%
# import libraries
import boto3, re, sys, math, json, os, sagemaker, urllib.request 
from sagemaker import get_execution_role
import numpy as np                                
import pandas as pd                               
import matplotlib.pyplot as plt                   
from IPython.display import Image                 
from IPython.display import display               
from time import gmtime, strftime                 
from sagemaker.predictor import csv_serializer   
#%%
role = get_execution_role()
awsBucketName = 'fintech-example1'
fileName = 'creditDefaultTest.csv'
bucketPath = f"s3://{awsBucketName}/{fileName}"

pd.read_csv(bucketPath)
