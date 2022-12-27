import numpy as np
import pandas as pd

df=pd.read_csv('he.csv')

x = df.drop(['condition'],axis='columns').values
y = df.condition.values

from sklearn.ensemble import RandomForestClassifier

RF = RandomForestClassifier()

RF.fit(x,y)

import pickle

pickle.dump(RF,open('model.pkl','wb'))

model = pickle.load(open('model.pkl','rb'))