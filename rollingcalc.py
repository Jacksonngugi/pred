import pandas as pd 

df = pd.read_csv('\\Users\\Vin\\numbered\\EPL')

#print(df['Team'][240:])
df = df[df['Team'] == 'BrightonandHoveAlbion']



df = df[['Date', 'Time', 'Comp', 'Round', 'Day', 'Venue', 'Result','GF', 'GA', 'Opponent', 'xG', 'xGA', 'Poss', 'Attendance', 'Captain','Formation', 'Referee', 'Match Report', 'Notes', 'Sh', 'SoT', 'Dist','PK', 'PKatt', 'Season', 'Team']]

df.loc[len(df.index)]= ['2023-3-19',1,'Copa del Rey',1,1,'Home',1,1,1,'Real Madrid',1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,'Barcelona']

def rolling_average(grp,cols,new_cols):
    #grp = grp.sort_values("Date")
    rolling_stats=grp[cols].rolling(5,closed='left').mean()
    grp[new_cols] = rolling_stats
    #grp =grp.dropna(subset=new_cols)
    return grp

cols=['Poss','PK','PKatt','Sh','SoT','Dist']
new_cols = [f"{c}_rolling" for c in cols]
matches = rolling_average(df,cols,new_cols)

print(matches[['Poss_rolling','PK_rolling','PKatt_rolling','Sh_rolling','SoT_rolling']])


