## DATA CLEANING AND FEATURE CREATION

# Add paths of additional scripts
import sys
sys.path.append('./data_scraping')
sys.path.append('./function_scripts')

## IMPORT
# Python packages
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import pickle
import requests
from bs4 import BeautifulSoup, Comment
import re
# sklearn functions
from sklearn.preprocessing import MinMaxScaler, KBinsDiscretizer
from sklearn.model_selection import train_test_split
# My functions
import assorted_funcs
from joining_dfs import combine_df_hitterdkpts

BSheaders = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/39.0.2171.95 Safari/537.36'}


curStandingsYear = 2020



# Load Model
loaded_model = pickle.load(open('finalized_model.sav', 'rb'))
# Load Scaler
loaded_scaler = pickle.load(open('scaler.pkl', 'rb'))
# Load KBins Discretizer
loaded_kbins = pickle.load(open('kbins.pkl', 'rb'))

## INITIAL LOADING AND CLEANING
# Load game data
temp_statsDF = pd.DataFrame()
for yeari in range(2020,2021):
    curYear_DF = combine_df_hitterdkpts(yeari)
    temp_statsDF = pd.concat([temp_statsDF, curYear_DF], ignore_index=True)

# Cut down to a single day to test on
temp_statsDF = temp_statsDF[temp_statsDF['Date'] == '01-09-2020'].reset_index(drop=True).copy()



statsDF = pd.DataFrame()
statsDF['Batter'] = temp_statsDF.loc[0:17]['Batter'].copy()
statsDF['BattingOrder'] = temp_statsDF.loc[0:17]['BattingOrder'].copy()
statsDF['HomeOrAway'] = temp_statsDF.loc[0:17]['HomeOrAway'].copy()
statsDF['Date'] = '04-04-2021'
statsDF['AwayTeam'] = 'New York Yankees'
statsDF['HomeTeam'] = 'Toronto Blue Jays'

## BATTER HANDEDNESS
statsDF['BatterHand'] = 'R'

## PITCHER HANDEDNESS
statsDF['A_SP_Hand'] = 'R'
statsDF['H_SP_Hand'] = 'L'

## WINNING PERCENTAGE - CHANGE STANDINGS SITE TO 2021 WHEN SEASON STARTS
r = requests.get("https://www.baseball-reference.com/leagues/MLB/" + str(curStandingsYear) + "-standings.shtml", headers=BSheaders)
soup = BeautifulSoup(r.content, "lxml")
teams_winpct = [x.text for x in soup.find_all("th", {"data-stat": "team_ID"}) if x.text != 'Tm']
winpctVals = [float(x.text) for x in soup.find_all("td", {"data-stat": "win_loss_perc"})]
winpctDict = {teams_winpct[i]: winpctVals[i] for i in range(len(teams_winpct))}

statsDF['A_SeaWinPct'] = statsDF['AwayTeam'].apply(lambda x: winpctDict[x])
statsDF['H_SeaWinPct'] = statsDF['HomeTeam'].apply(lambda x: winpctDict[x])
statsDF['TE_SeaWinPct'] = statsDF.apply(lambda x: x['H_SeaWinPct'] if x['HomeOrAway'] == 'Home' else x['A_SeaWinPct'], axis=1)
statsDF['OP_SeaWinPct'] = statsDF.apply(lambda x: x['H_SeaWinPct'] if x['HomeOrAway'] == 'Away' else x['A_SeaWinPct'], axis=1)


# RECENT PITCHER FIP
statsDF['A_SP_recFIP'] = 0.5
statsDF['H_SP_recFIP'] = 0.5
statsDF['OP_SP_recFIP'] = 0.5
statsDF['TE_SP_recFIP'] = 0.5

## RECENT WOBA FOR BATTER AND THOSE HITTING AROUND HIM IN ORDER
# Get hitters batting before and after current batter in lineup
oneBefore = {1:9,2:1,3:2,4:3,5:4,6:5,7:6,8:7,9:8}
oneAfter = {1:2,2:3,3:4,4:5,5:6,6:7,7:8,8:9,9:1}
statsDF['Batter-1'] = statsDF.apply(lambda x: statsDF[(statsDF['HomeTeam'] == x['HomeTeam']) & (statsDF['HomeOrAway'] == x['HomeOrAway']) & (statsDF['BattingOrder'] == oneBefore[x['BattingOrder']])]['Batter'].values[0], axis=1)
statsDF['Batter+1'] = statsDF.apply(lambda x: statsDF[(statsDF['HomeTeam'] == x['HomeTeam']) & (statsDF['HomeOrAway'] == x['HomeOrAway']) & (statsDF['BattingOrder'] == oneAfter[x['BattingOrder']])]['Batter'].values[0], axis=1)

statsDF['Batter_recwOBA'] = 0.5
statsDF['Batter-1_recwOBA'] = 0.5
statsDF['Batter+1_recwOBA'] = 0.5

## ZIPS PROJECTIONS
statsDF['BABIP_zips'] = 0.5
statsDF['ISO_zips'] = 0.5
statsDF['OP_BABIP_zips'] = 0.5
statsDF['OP_ERA-_zips'] = 0.5


## BULLPEN STATS
#Define opposing Team for Later Joins
statsDF['OppoTeam'] = statsDF.apply(lambda x: x['AwayTeam'] if x['HomeOrAway'] == 'Home' else x['HomeTeam'], axis=1)

# Load last year's bullpen stats for opposing team
bullpenStats = pd.read_csv('input/bullpen/bullpenByHand_2020.csv', sep=',')
statsDF = pd.merge(statsDF, bullpenStats,  how='left', left_on=['OppoTeam'], right_on = ['Team'])



## TEMPERATURE
statsDF['temperature'] = 0.5

## VEGAS ODDS
statsDF['HomeOdds'] = 0.5
statsDF['OverUnder'] = 0.5

## PARK FACTORS
parkFactors = pd.read_csv('input/parkFactors/parkFactorsByHand_2021.csv', sep=',')
statsDF = pd.merge(statsDF, parkFactors,  how='left', left_on=['HomeTeam'], right_on = ['Team'])


# Separate Season win pct into hitter's team winpct and opposing team winpct
statsDF['TE_SeaWinPct'] = statsDF.apply(lambda x: x['H_SeaWinPct'] if x['HomeOrAway'] == 'Home' else x['A_SeaWinPct'], axis=1)
statsDF['OP_SeaWinPct'] = statsDF.apply(lambda x: x['H_SeaWinPct'] if x['HomeOrAway'] == 'Away' else x['A_SeaWinPct'], axis=1)

# Separate recent FIP of pitchers into hitter's team pitcher and opposing pitcher
statsDF['OP_SP_recFIP'] = statsDF.apply(lambda x: x['A_SP_recFIP'] if x['HomeOrAway'] == 'Home' else x['H_SP_recFIP'], axis=1)
statsDF['TE_SP_recFIP'] = statsDF.apply(lambda x: x['H_SP_recFIP'] if x['HomeOrAway'] == 'Home' else x['A_SP_recFIP'], axis=1)

# Get Handedness Matchup between Hitter and Opposing Pitcher
statsDF = assorted_funcs.handednessFeatures(statsDF)

# Create Park Factors Columns adjusted for Batter's Actual Handedness
# Correct switch hitters
statsDF['BatterHand'] = statsDF.apply(lambda x: 'L' if ((x['BatterHand'] == 'B') & (x['O_SP_Hand'] == 'R')) else ('R' if ((x['BatterHand'] == 'B') & (x['O_SP_Hand'] == 'L')) else x['BatterHand']), axis=1)
# Get Park Factors based on Batter's Handedness
for curStat in ['1B','2B','3B','HR']:
    statsDF['ParkAdj_' + curStat] = statsDF.apply(lambda x: x['Park_' + curStat + '_L'] if x['BatterHand'] == 'L' else x['Park_' + curStat + '_R'], axis=1)

statsDF.reset_index(drop=True, inplace=True)

# SET LABEL COLUMN AND ADD IT TO THE STATS DATAFRAME
labelCol = 'DKPts'

# Classify all numeric data into bins
nonStatsColumns = [x for x in statsDF.columns if 'prevY' not in x]
for coli in nonStatsColumns:
    if coli == labelCol: pass
    elif ((statsDF[coli].dtypes == 'int64') or (statsDF[coli].dtypes == 'float64')) and coli not in ['year']:
        #statsDF[coli] = pd.qcut(statsDF[coli], 10, labels=False, duplicates='drop')
        #statsDF[coli] = statsDF[coli].astype(str)
        statsDF[coli].fillna(statsDF[coli].mean(), inplace=True)

# CREATE DATAFRAME WITH FEATURES THAT WILL BE INPUTTED INTO MODEL
useful_features = ['BattingOrder']
useful_features.extend([x for x in statsDF.columns if 'ParkAdj' in x])
useful_features.extend(['temperature'])
useful_features.extend(['HomeOdds','OverUnder'])
useful_features.extend(['Batter_recwOBA','Batter-1_recwOBA','Batter+1_recwOBA'])
useful_features.extend(['BABIP_zips','ISO_zips'])
useful_features.extend(['OP_BABIP_zips','OP_ERA-_zips'])
useful_features.extend(['OP_SP_recFIP','TE_SP_recFIP'])
useful_features.extend(['TE_SeaWinPct','OP_SeaWinPct'])
useful_features.extend(['BP_AVG'])


# Bin only certain columns
binnedCols = ['OverUnder','HomeOdds','temperature','BABIP_zips','ISO_zips','OP_BABIP_zips','OP_ERA-_zips']
binnedCols.extend([x for x in statsDF.columns if 'ParkAdj' in x])
binnedCols.extend(['Batter_recwOBA','Batter-1_recwOBA','Batter+1_recwOBA'])
binnedCols.extend(['OP_SP_recFIP','TE_SP_recFIP'])
binnedCols.extend(['TE_SeaWinPct','OP_SeaWinPct'])
binnedCols.extend(['BP_AVG'])
nonBinnedCols = [x for x in useful_features if x not in binnedCols]

b_featuresDF = loaded_kbins.transform(statsDF[binnedCols])
b_featuresDF = pd.DataFrame(data=b_featuresDF,columns=statsDF[binnedCols].columns)
featuresDF = pd.concat([b_featuresDF, statsDF[nonBinnedCols]], axis=1, sort=False)

## DELETE AFTER FIXING KBINS ABOVE
#featuresDF = pd.concat([statsDF[binnedCols], statsDF[nonBinnedCols]], axis=1, sort=False)



featuresDF = pd.get_dummies(featuresDF)
features_list = list(featuresDF.columns)

test_features = np.array(featuresDF)

# Scale Testing Features
test_features = loaded_scaler.transform(test_features)

                    
                    
# Make predictions
predictions = loaded_model.predict(test_features)
predictions = [round(x,2) for x in predictions]