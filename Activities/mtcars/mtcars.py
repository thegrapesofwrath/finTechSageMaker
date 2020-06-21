# import libraries
from sagemaker import get_execution_role                            
import pandas as pd   

role = get_execution_role()
awsBucketName = 'studentname-bucket'
fileName = 'mtcars.csv'
bucketPath = f"s3://{awsBucketName}/{fileName}"

pd.read_csv(bucketPath)