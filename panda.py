import numpy as np
import pandas as pd
from sklearn import preprocessing


train = pd.read_csv("/Users/samarth/Downloads/datafiles19cdaf8/train.csv")
test =  pd.read_csv("/Users/samarth/Downloads/datafiles19cdaf8/test.csv")

train.info()

print ("The train data has",train.shape)
print ("The test data has",test.shape)

train.head()

nans = train.shape[0] - train.dropna().shape[0]
print ("%d rows have missing values in the train data" %nans)

nand = test.shape[0] - test.dropna().shape[0]
print ("%d rows have missing values in the test data" %nand)

data = train.isnull().sum()
print data

cat = train.select_dtypes(include=['O'])
cat.apply(pd.Series.nunique)

print cat

#Education
train.workclass.value_counts(sort=True)
train.workclass.fillna('Private',inplace=True)


#Occupation
train.occupation.value_counts(sort=True)
train.occupation.fillna('Prof-specialty',inplace=True)


#Native Country
train['native.country'].value_counts(sort=True)
train['native.country'].fillna('United-States',inplace=True)

updated_train = train.isnull().sum()
print updated_train

count = train.shape[0]
print count

c = train.target.value_counts()
c_1 = c/count
print c_1

cross_table =pd.crosstab(train.education, train.target,margins=True)/train.shape[0]
print cross_table

for x in train.columns:
    if train[x].dtype == 'object':
        lbl = preprocessing.LabelEncoder()
        lbl.fit(list(train[x].values))
        train[x] = lbl.transform(list(train[x].values))

print train.head()

print train.target.value_counts()
