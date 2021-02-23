import sys
sys.path.append('./data_scraping')
sys.path.append('./function_scripts')

import pandas as pd
import numpy as np



def combine_df(year):
    lineups = pd.read_csv('input/gamelogs/gamelogs' + str(year) + '_lineups.csv', sep=',')
    
    winpct = pd.read_csv('input/gamelogs/gamelogs' + str(year) + '_winpct.csv', sep=',')
    curDF = pd.merge(lineups, winpct,  how='left', left_on=['Date','AwayTeam','HomeTeam'], right_on = ['Date','AwayTeam','HomeTeam'])
    
    recwOBA = pd.read_csv('input/gamelogs/gamelogs' + str(year) + '_recwOBA.csv', sep=',')
    curDF = pd.merge(curDF, recwOBA,  how='left', left_on=['Date','AwayTeam','HomeTeam'], right_on = ['Date','AwayTeam','HomeTeam'])
    
    recFIP = pd.read_csv('input/gamelogs/gamelogs' + str(year) + '_recFIP.csv', sep=',')
    curDF = pd.merge(curDF, recFIP,  how='left', left_on=['Date','AwayTeam','HomeTeam'], right_on = ['Date','AwayTeam','HomeTeam'])
    
    weather = pd.read_csv('input/gamelogs/gamelogs' + str(year) + '_weather.csv', sep=',')
    curDF = pd.merge(curDF, weather,  how='left', left_on=['Date','AwayTeam','HomeTeam'], right_on = ['Date','AwayTeam','HomeTeam'])
    
    
    curDF['year'] = curDF['Date'].apply(lambda x: int(x[-4:]))
    curDF['month'] = curDF['Date'].apply(lambda x: int(x[3:5]))
    # Remove playoff games
    curDF = curDF[curDF['month'] != 10]
    return curDF