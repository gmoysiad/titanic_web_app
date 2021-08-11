import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.svm import SVC

train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')

imputer = KNNImputer()

model = SVC()
