# LOAD GAME LOGS CSVS AND ADD COLUMNS WITH NUMBER OF DAYS OFF BEFORE CURRENT GAME FOR HOME AND AWAY TEAM

import sys

sys.path.append('../input/gamelogs')


import pandas as pd
import numpy as np
from datetime import datetime, timedelta

yeari = 2015
statsDF = pd.read_csv('../input/gamelogs/gamelogs' + str(yeari) + '_orig.csv', sep=',')

statsDF['Date_dt'] = statsDF['Date'].apply(lambda x: datetime.strptime(x, '%d-%m-%Y'))

APlayedYest = []
HPlayedYest = []
for i in statsDF.index:
    dayBefore = statsDF.loc[i]['Date_dt'] - timedelta(1)
    curATeam = statsDF.loc[i]['AwayTeam']
    curHTeam = statsDF.loc[i]['HomeTeam']
    
    APlayedBefore = statsDF[((statsDF['AwayTeam'] == curATeam) | (statsDF['HomeTeam'] == curATeam)) & (statsDF['Date_dt'] == dayBefore)]
    if len(APlayedBefore) > 0: APlayedYest.append('1')
    else: APlayedYest.append('0')
    HPlayedBefore = statsDF[((statsDF['AwayTeam'] == curHTeam) | (statsDF['HomeTeam'] == curHTeam)) & (statsDF['Date_dt'] == dayBefore)]
    if len(HPlayedBefore) > 0: HPlayedYest.append('1')
    else: HPlayedYest.append('0')
    
statsDF['APlayedYest'] = APlayedYest
statsDF['HPlayedYest'] = HPlayedYest


statsDF.to_csv('../input/gamelogs/gamelogs' + str(yeari) + '.csv', index=False)