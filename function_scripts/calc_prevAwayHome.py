# LOAD GAME LOGS CSVS AND ADD COLUMNS WITH WHETHER EACH TEAM'S LAST GAME WAS AT HOME OR AWAY

import sys

sys.path.append('../input/gamelogs')


import pandas as pd
import numpy as np
from datetime import datetime, timedelta

yeari = 2017
statsDF = pd.read_csv('../input/gamelogs/gamelogs' + str(yeari) + '.csv', sep=',')

statsDF['Date_dt'] = statsDF['Date'].apply(lambda x: datetime.strptime(x, '%d-%m-%Y'))

ALastGame = []
HLastGame = []
for i in statsDF.index:
    dayBefore = statsDF.loc[i]['Date_dt'] - timedelta(1)
    curATeam = statsDF.loc[i]['AwayTeam']
    curHTeam = statsDF.loc[i]['HomeTeam']
    
    APlayedBefore = statsDF[((statsDF['AwayTeam'] == curATeam) | (statsDF['HomeTeam'] == curATeam)) & (statsDF['Date_dt'] == dayBefore)].reset_index(drop=True)
    if len(APlayedBefore) > 0: 
        if APlayedBefore['AwayTeam'][0] == curATeam: ALastGame.append('Away')
        else: ALastGame.append('Home')
    else: ALastGame.append('NoGame')
    HPlayedBefore = statsDF[((statsDF['AwayTeam'] == curHTeam) | (statsDF['HomeTeam'] == curHTeam)) & (statsDF['Date_dt'] == dayBefore)].reset_index(drop=True)
    if len(HPlayedBefore) > 0: 
        if HPlayedBefore['AwayTeam'][0] == curHTeam: HLastGame.append('Away')
        else: HLastGame.append('Home')
    else: HLastGame.append('NoGame')

statsDF['ALastGame'] = ALastGame
statsDF['HLastGame'] = HLastGame

statsDF.to_csv('../input/gamelogs/gamelogs' + str(yeari) + '.csv', index=False)