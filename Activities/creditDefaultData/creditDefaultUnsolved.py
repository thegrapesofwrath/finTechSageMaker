from sagemaker import get_execution_role                            
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

role = get_execution_role()
awsBucketName = ''
fileName = ''
bucketPath = f"s3://{awsBucketName}/{fileName}"

yourVariableName = pd.read_csv(bucketPath)