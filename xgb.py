import pandas as pd 
from sklearn.preprocessing import LabelEncoder
import glob
from xgboost import XGBClassifier

all_df = []
for file in (glob.glob("/Users/Vin/data/*.csv")):
    

    data=pd.read_csv(file)
    
    all_df.append(data)
df = pd.concat(all_df,ignore_index = True)


del df["Time"]

clf=XGBClassifier(seed=10)

class MissingDict(dict):
    __missing__ =lambda self,key:key

map_values = {
    "E0":"EPL",
    "E1":"Championship",
    "E2":"Ligue 1",
    "EC":"Conference",
    "SC0":"SPL",
    "D1":"Bundesliga",
    "I1":"Serie A",
    "SP1":"LaLiga"
}
mapping = MissingDict(**map_values)

df['new_Div']=df['Div'].map(mapping)
df['Date'] = pd.to_datetime(df['Date'],dayfirst =True)

print(df)

categorical_features = ['new_Div','HomeTeam','AwayTeam','FTR']
encoders = dict()
for cat in categorical_features:
    encoders[cat] = LabelEncoder()
    df[f'{cat}_n'] = encoders[cat].fit_transform(df[cat])

print(df[df['Date'] >= "2022-05-28"])

train = df[df['Date'] < "2022-05-28"]
test = df[df['Date'] >= "2022-05-28 "]

x = ['new_Div_n','HomeTeam_n','AwayTeam_n','1','x','2']
y = ['FTR_n']

clf.fit(train[x],train[y].values.ravel())

df = pd.DataFrame({
    'hometeam':[""],
    'awayteam':['Ath Madrid'],
    'new_div':['LaLiga'],
    '1':[3.80],
    'x':[3.40],
    '2':[2.05]

})
HT=encoders['HomeTeam']
AT=encoders['AwayTeam']
Div=encoders['new_Div']

df['HomeTeam_n']=HT.transform(df['hometeam'])
df['AwayTeam_n']=AT.transform(df['awayteam'])
df['new_Div_n']=Div.transform(df['new_div'])

ddata = pd.DataFrame({
    'new_Div_n':df['new_Div_n'],
    'HomeTeam_n':df['HomeTeam_n'],
    'AwayTeam_n':df['AwayTeam_n'],
    '1':df['1'],
    'x':df['x'],
    '2':df['2'],

})


print(ddata)

pred = clf.predict(ddata)

print(pred)

