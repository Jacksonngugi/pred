#For trial 

import pandas as pd 
import numpy as np
from sklearn import preprocessing 
from sklearn.metrics import accuracy_score 
from sklearn.model_selection import GridSearchCV 
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
import glob
import warnings

all_df = []
for file in (glob.glob("/Users/Vin/numbered/*")):
    

    data=pd.read_csv(file)
    
    all_df.append(data)
df = pd.concat(all_df,ignore_index = True)
#df = pd.read_csv('/Users/Vin/fbrefdata/premierleague2022')
df = df.dropna(subset=['Time'])

grp = df.groupby("Team")

def rolling_average(grp,cols,new_cols):
    grp = grp.sort_values("Date")
    rolling_stats=grp[cols].rolling(5,closed='left').mean()
    grp[new_cols] = rolling_stats
    grp =grp.dropna(subset=new_cols)
    return grp

cols=['Poss','PK','PKatt','Sh','SoT','Dist']
new_cols = [f"{c}_rolling" for c in cols]
matches = df.groupby("Team").apply(lambda x: rolling_average(x,cols,new_cols))
df = matches.droplevel("Team")
df.index = range(matches.shape[0])


df['venue_code'] = df['Venue'].astype("category").cat.codes
df['opp_code'] = df['Opponent'].astype("category").cat.codes
df['hour'] = df['Time'].str.replace(':.+',"",regex =True).astype("int")
df['Result'] = df['Result'].astype("category").cat.codes
df['team'] = df['Team'].astype("category").cat.codes
df['Comp'] = df['Comp'].astype("category").cat.codes
df['Formation'] = df['Formation'].astype("category").cat.codes
df['Round'] = df['Round'].astype("category").cat.codes

from pprint import pprint
#Uncomment the classifier you want to using.
clf=XGBClassifier()
#clf = MLPClassifier(random_state=12,hidden_layer_sizes=20,activation='relu',solver='adam',learning_rate='adaptive',learning_rate_init = 0.0001)
#clf = RandomForestClassifier(n_estimators=50,min_samples_split=10,random_state=1)

#pprint(clf.get_params())
warnings.filterwarnings('ignore')

train = df[df["Date"] < "2023-02-20"]
test=df[df["Date"] > "2023-02-20"]

train = train[['Date','Comp','Round','Sh','SoT','PKatt','Sh_rolling','SoT_rolling','Dist_rolling','PK_rolling','PKatt_rolling','Poss_rolling','Season','hour', 'opp_code', 'venue_code', 'team','Result','Round']]
test = test[['Date','Comp','Round','Sh','SoT','PKatt','Sh_rolling','SoT_rolling','Dist_rolling','PK_rolling','PKatt_rolling','Poss_rolling','Season','hour', 'opp_code', 'venue_code', 'team','Result','Round']]

train = train.dropna()
test = test.dropna()

#x = ['hour','opp_code','venue_code','xG','xGA','team']
x = ['hour','opp_code','Comp','venue_code','SoT_rolling','Sh_rolling','team','PKatt_rolling','Poss_rolling','PK_rolling','Round']
y = ['Result']
#params = {
    #'hidden_layer_sizes':[10],[(20,20)],
    #'activation':['relu'],
    #'solver':['adam'],
    #'learning_rate':['constant','invscaling','adaptive']


#}

#gridsearch = GridSearchCV(MLPClassifier(),params,verbose=2)
#gridsearch.fit(train[x],train[y].values.ravel())
#print('best params:',gridsearch.best_params_)
clf.fit(train[x],train[y].values.ravel())

pred = clf.predict(test[x])
print(pred)
print(test[y])
acc = accuracy_score(test[y],pred)

print(acc)
