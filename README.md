# 1.1 Lesson Plan: Predicting Credit Defaults with Amazon SageMaker

## Overview

Today's Class will train a Random Forest on data provided by [Home Credit](http://www.homecredit.net/ "Home Credit") to determine if a borrower is likely to default on a loan. The exploratory data analysis and model training will be conducted on a Jupyter Notebook on [Amazon SageMaker](https://aws.amazon.com/sagemaker/ "Amazon SageMaker provides a cloud service for machine learning."). The module files can be found at [this](https://github.com/thegrapesofwrath/finTechSageMaker "Module Github Repository") github repository. 

### Learning Objectives

At the end of the session, learners will be able to:
 
* Upload dataSets to an Amazon S3 bucket

* Retrieve data from a S3 bucket into a Jupyter notebook on SageMaker

* Understand how an exploratory data analysis is beneficial to train a model

* Understand how imputation is used to estimate missing values

* Understand how to create a random forest model

### Instructor Notes
Students will be expected to clone the data from the class repository and follow along to upload the files into their own [S3](https://aws.amazon.com/s3/ "Amazon file storage service") bucket. Once a notebook has been created proceed to introduce the topics on exploratory data analysis using the [slides](https://docs.google.com/presentation/d/1rT89zpfIj99OE-EiuRBF1ius1b4hkzULGQjJnuEGJQs/edit?usp=sharing "This week's slideshow."). 

Demonstrate an example exploratory analysis using the [mtcars](Activities/mtcars/mtcars.csv "mtcars Dataset") dataset. Split students into breakout rooms where they will try an exploratory analysis on the [creditDefaultTrain](Activities/creditDefaultData/creditDefaultTrain.csv "The credit default training data.") dataset. Upon returning from the breakout rooms have students share their findings. Proceed to finish the class by training the random forest model on creditDefaultTrain and then answer any questions. After class, students can test their model on the [creditDefaultTest](Activities/creditDefaultData/creditDefaultTest.csv) dataset and think of ways they can improve their accuracy.


##### Module Student Dependencies
* AWS account
* github account
* python proficiency
  

##### Potential Student Environment Issues
* Students maybe confused in their scripts between using the object URL and using an s3:// url with the bucket name. If they try to use the object URL the file will not be accessible by SageMaker until the security settings are changed to make it public. Best practice would be to use the s3:// url and bucket name as the file does not have to be made public. 

>**Tip**: If time permits and a student is having environment difficulties you may ask them to share their screen while you help them resolve it. Other students will benefit who are having the same issue.

![](Images/objectURL.png)

s3:// urls are thus created:
``` python
s3URL = "s3://fintech-example1/creditDefaultTest.csv" #s3://bucket-name/fileName
```

##### Git LFS
If a repository needs to be configured for large file storage after it has been created it maybe done so in the following way:
``` bash
git lfs install
git lfs track "*.csv" #type of file to track
git add .gitattributes

git add .
git commit -m "Added lfs"
git push origin master
```

##### Additional Reading
* [An Introduction to Statistical Learning](AdditionalReading/ISLRSeventhPrinting.pdf "This is a good reference and place to review some concepts for advanced students.") - Examples are in R and the theory is explained concisely.

- - -

### 0. Class Do: Interview Question Warm-Up
##### (5 mins before class - first 2 mins of class) 

Open the [slides](https://docs.google.com/presentation/d/1rT89zpfIj99OE-EiuRBF1ius1b4hkzULGQjJnuEGJQs/edit?usp=sharing "This week's slideshow.") for today's class and begin the weekly presentation with the first slide. 

>**This week's question:** What are some advantages and disadvantages for using a cloud service to train machine learning models?

Allow the question to be on the screen 5 mins prior to the start of class as students join the session. Ask the class to answer the question as they complete the pre-lecture temperature check.

**Possible answers to this week's question:**

**Advantages**
* Maybe cheaper than setting up hardware at the organization.
* Quicker to set up and use
* Easy to scale if more resources are needed

**Disadvantages**
* Businesses may not allow external services to access sensitive data
* New authentication methods could be difficult to integrate into an enterprise's existing user authentication system.

---

### 0. Instructor Do: Temperature Check 
##### (5 mins before class - first 2 mins of class)
Using the [Zoom Polling](https://support.zoom.us/hc/en-us/articles/213756303-Polling-for-Meetings  "How to setup a zoom poll.") feature launch a poll for the class to identify where the class as a group is comfortable with the material. Do this while people are joining and during this time the TA may take attendance. 

**Poll Text:**

*Select all of the topics that you feel prepared to apply outside of the class from this week's lesson:*

* Accessing data from an Amazon S3 bucket : **A**
* Creating notebooks in SageMaker : **A**
* [Pearson's correlation coefficient](https://en.wikipedia.org/wiki/Pearson_correlation_coefficient "A statistic that measures linear correlation between two variables X and Y")  : **B**
* [Label Encoding](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelEncoder.html "Scikit LabelEncoder")  : **B**
* [One-hot Encoding](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.get_dummies.html "Pandas get_dummies")  : **B**
* [Pandas dtypes](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.dtypes.html "dtypes")  : **B**
* [Imputation](https://en.wikipedia.org/wiki/Imputation_(statistics) "Replacing missing data with substituted values")  : **C**
* [Random Forest](https://towardsdatascience.com/understanding-random-forest-58381e0602d2 "Random Forest Article")  : **C**

---

## A. S3 data upload and SageMaker notebook creation
### Everyone Do: Connect S3 and SageMaker
##### (15 minutes)

Everyone one must have an account created. Students will need to create two buckets, one for the mtcars dataset and one for the creditDefault dataset. They will then create a sageMaker notebook instance for their notebooks. Everyone will verify that it is set up correctly by pulling the mtcars dataset from the S3 bucket into their notebook.

>**Tip:** The creditDefault dataset is larger than the mtcars dataset. Students may have a slower internet speed so uploading at the beginning of class may save some time.

### Create an S3 bucket



Navigate to the [aws console](https://aws.amazon.com/console/) and click on the link for S3. 
![](Images/openS3.gif)

Click on the create bucket button and give the bucket an Amazon approved name.
>**Note:** The bucket name may only contain lowercase letters and hyphens.

![](Images/createBucket.gif)
![](Images/uploadFileS3.gif)
![](Images/navigateSageMaker.gif)
![](Images/openNotebookInstances.gif)
![](Images/createNotebook1.gif)
![](Images/createNotebook2.gif)
Wait for the notebook status to change from Pending to InService
Open the new notebook instance.

![](Images/openNoteBook.gif)
![](Images/createJupyterNotebook.gif)

Rename the newly created notebook to mtcarsAnalysis.

Verify that you can import the csv file in the S3 bucket into the Jupyter notebook using the following code:
``` python
# import libraries
from sagemaker import get_execution_role                            
import pandas as pd     

role = get_execution_role()     #Gets the current SageMaker instance execution role
awsBucketName = ''              #populate with the bucket name you created
fileName = 'mtcars.csv'
bucketPath = f"s3://{awsBucketName}/{fileName}"

pd.read_csv(bucketPath)
```
You should see the mtcars dataset populated. 

>**Transition Tip:** Before moving on to the slides ask if any students are not able to access the mtcars data in their notebook. If students are having issues and time permits have them share their screen. Once everyone is able to access the dataset through their notebook open the [slides](https://docs.google.com/presentation/d/1rT89zpfIj99OE-EiuRBF1ius1b4hkzULGQjJnuEGJQs/edit?usp=sharing "This week's slideshow.") for exploratory data analysis.
## B. Exploratory Data Analysis

## C. Imputation and Model Training

