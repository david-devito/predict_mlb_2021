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
    
    #Define opposing Pitcher for Later Joins
    curDF['OppoPitcher'] = curDF.apply(lambda x: x['AwaySP'] if x['HomeOrAway'] == 'Home' else x['HomeSP'], axis=1)
    
    
    parkFactors = pd.read_csv('input/parkFactors/parkFactorsByHand_' + str(year) + '.csv', sep=',')
    curDF = pd.merge(curDF, parkFactors,  how='left', left_on=['HomeTeam'], right_on = ['Team'])
    
    winpct = pd.read_csv('input/gamelogs/gamelogs' + str(year) + '_winpct.csv', sep=',')
    curDF = pd.merge(curDF, winpct[['Date','AwayTeam','HomeTeam','A_SeaWinPct','H_SeaWinPct']],  how='left', left_on=['Date','AwayTeam','HomeTeam'], right_on = ['Date','AwayTeam','HomeTeam'])
    
    
    recwOBA = pd.read_csv('input/gamelogs/gamelogs' + str(year) + '_recwOBA.csv', sep=',')
    curDF = pd.merge(curDF, recwOBA,  how='left', left_on=['Date','AwayTeam','HomeTeam'], right_on = ['Date','AwayTeam','HomeTeam'])
    
    recFIP = pd.read_csv('input/gamelogs/gamelogs' + str(year) + '_recFIP.csv', sep=',')
    curDF = pd.merge(curDF, recFIP,  how='left', left_on=['Date','AwayTeam','HomeTeam'], right_on = ['Date','AwayTeam','HomeTeam'])
    
    weather = pd.read_csv('input/gamelogs/gamelogs' + str(year) + '_weather.csv', sep=',')
    curDF = pd.merge(curDF, weather[['Date','AwayTeam','HomeTeam','temperature']],  how='left', left_on=['Date','AwayTeam','HomeTeam'], right_on = ['Date','AwayTeam','HomeTeam'])
    
    vegas = pd.read_csv('input/vegasOdds/vegasOdds_' + str(year) + '.csv', sep=',')
    curDF = pd.merge(curDF, vegas,  how='left', left_on=['Date','AwayTeam','HomeTeam'], right_on = ['Date','AwayTeam','HomeTeam'])
    
    batterHand = pd.read_csv('input/gamelogs/gamelogs' + str(year) + '_hitters_hand_dkpts.csv', sep=',')
    curDF = pd.merge(curDF, batterHand,  how='left', left_on=['Date','AwayTeam','HomeTeam','Batter'], right_on = ['Date','AwayTeam','HomeTeam','Batter'])
    
    pitcherHand = pd.read_csv('input/gamelogs/gamelogs' + str(year) + '_pitchers_hand_dkpts.csv', sep=',')
    curDF = pd.merge(curDF, pitcherHand,  how='left', left_on=['Date','AwayTeam','HomeTeam'], right_on = ['Date','AwayTeam','HomeTeam'])
    
    zips_h = pd.read_csv('input/projections/zips/zips_hitters_' + str(year) + '.csv', sep=',')
    curDF = pd.merge(curDF, zips_h,  how='left', left_on=['Batter'], right_on = ['Player'])
    
    zips_p = pd.read_csv('input/projections/zips/zips_pitchers_' + str(year) + '.csv', sep=',')
    curDF = pd.merge(curDF, zips_p,  how='left', left_on=['OppoPitcher'], right_on = ['Player'])
    
    #Define opposing Team for Later Joins
    curDF['OppoTeam'] = curDF.apply(lambda x: x['AwayTeam'] if x['HomeOrAway'] == 'Home' else x['HomeTeam'], axis=1)
    
    # Load last year's bullpen stats for opposing team
    bullpenStats = pd.read_csv('input/bullpen/bullpenByHand_' + str(year-1) + '.csv', sep=',')
    curDF = pd.merge(curDF, bullpenStats,  how='left', left_on=['OppoTeam'], right_on = ['Team'])

    
    # Remove playoff games
    curDF = curDF[curDF['month'] != 10]
    return curDF