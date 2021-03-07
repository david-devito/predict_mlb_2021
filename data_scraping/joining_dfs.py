import sys
sys.path.append('./data_scraping')
sys.path.append('./function_scripts')

import pandas as pd
import numpy as np



def combine_df(year):
    curDF = pd.read_csv('input/gamelogs/gamelogs' + str(year) + '_lineups.csv', sep=',')
    
    #winpct = pd.read_csv('input/gamelogs/gamelogs' + str(year) + '_winpct.csv', sep=',')
    #curDF = pd.merge(curDF, winpct,  how='left', left_on=['Date','AwayTeam','HomeTeam'], right_on = ['Date','AwayTeam','HomeTeam'])
    
    #recwOBA = pd.read_csv('input/gamelogs/gamelogs' + str(year) + '_recwOBA.csv', sep=',')
    #curDF = pd.merge(curDF, recwOBA,  how='left', left_on=['Date','AwayTeam','HomeTeam'], right_on = ['Date','AwayTeam','HomeTeam'])
    
    #recFIP = pd.read_csv('input/gamelogs/gamelogs' + str(year) + '_recFIP.csv', sep=',')
    #curDF = pd.merge(curDF, recFIP,  how='left', left_on=['Date','AwayTeam','HomeTeam'], right_on = ['Date','AwayTeam','HomeTeam'])
    
    #weather = pd.read_csv('input/gamelogs/gamelogs' + str(year) + '_weather.csv', sep=',')
    #curDF = pd.merge(curDF, weather,  how='left', left_on=['Date','AwayTeam','HomeTeam'], right_on = ['Date','AwayTeam','HomeTeam'])
    
    
    curDF['year'] = curDF['Date'].apply(lambda x: int(x[-4:]))
    curDF['month'] = curDF['Date'].apply(lambda x: int(x[3:5]))
    # Remove playoff games
    curDF = curDF[curDF['month'] != 10]
    return curDF

def combine_df_hitterdkpts(year):
    curDF = pd.read_csv('input/gamelogs/gamelogs' + str(year) + '_lineups.csv', sep=',')
    
    curDF['year'] = curDF['Date'].apply(lambda x: int(x[-4:]))
    curDF['month'] = curDF['Date'].apply(lambda x: int(x[3:5]))
    
    dkpts = pd.read_csv('input/gamelogs/gamelogs' + str(year) + '_hitters_dkpts.csv', sep=',')
    curDF = pd.merge(curDF, dkpts,  how='left', left_on=['Date','AwayTeam','HomeTeam'], right_on = ['Date','AwayTeam','HomeTeam'])
    
    parkFactors = pd.read_csv('input/parkFactors/parkFactors.csv', sep=',')
    curDF = pd.merge(curDF, parkFactors,  how='left', left_on=['year','HomeTeam'], right_on = ['ParkYear','Park'])
    
    winpct = pd.read_csv('input/gamelogs/gamelogs' + str(year) + '_winpct.csv', sep=',')
    curDF = pd.merge(curDF, winpct,  how='left', left_on=['Date','AwayTeam','HomeTeam'], right_on = ['Date','AwayTeam','HomeTeam'])
    
    
    recwOBA = pd.read_csv('input/gamelogs/gamelogs' + str(year) + '_recwOBA.csv', sep=',')
    curDF = pd.merge(curDF, recwOBA,  how='left', left_on=['Date','AwayTeam','HomeTeam'], right_on = ['Date','AwayTeam','HomeTeam'])
    
    recFIP = pd.read_csv('input/gamelogs/gamelogs' + str(year) + '_recFIP.csv', sep=',')
    curDF = pd.merge(curDF, recFIP,  how='left', left_on=['Date','AwayTeam','HomeTeam'], right_on = ['Date','AwayTeam','HomeTeam'])
    
    weather = pd.read_csv('input/gamelogs/gamelogs' + str(year) + '_weather.csv', sep=',')
    curDF = pd.merge(curDF, weather,  how='left', left_on=['Date','AwayTeam','HomeTeam'], right_on = ['Date','AwayTeam','HomeTeam'])
    
    vegas = pd.read_csv('input/vegasOdds/vegasOdds_' + str(year) + '.csv', sep=',')
    curDF = pd.merge(curDF, vegas,  how='left', left_on=['Date','AwayTeam','HomeTeam'], right_on = ['Date','AwayTeam','HomeTeam'])
    

    # Remove playoff games
    curDF = curDF[curDF['month'] != 10]
    return curDF