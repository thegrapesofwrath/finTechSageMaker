from sagemaker import get_execution_role                            
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

role = get_execution_role()
awsBucketName = ''
fileName = ''
bucketPath = f"s3://{awsBucketName}/{fileName}"

creditDefault = pd.read_csv(bucketPath)

#TODO 
# Finish the analysis and finish lesson 
# with demonstrating handling missing values (Imputation)
# then train a random forest classifier on 
# the creditDefault dataset.
#

