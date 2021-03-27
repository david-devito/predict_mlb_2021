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
# sklearn functions
from sklearn.preprocessing import MinMaxScaler, KBinsDiscretizer
from sklearn.model_selection import train_test_split
# My functions
import assorted_funcs
from joining_dfs import combine_df_hitterdkpts


# Load Model
loaded_model = pickle.load(open('finalized_model.sav', 'rb'))
# Load Scaler
loaded_scaler = pickle.load(open('scaler.pkl', 'rb'))

## INITIAL LOADING AND CLEANING
# Load game data
temp_statsDF = pd.DataFrame()
for yeari in range(2020,2021):
    curYear_DF = combine_df_hitterdkpts(yeari)
    temp_statsDF = pd.concat([temp_statsDF, curYear_DF], ignore_index=True)

# Cut down to a single day to test on
temp_statsDF = temp_statsDF[temp_statsDF['Date'] == '01-09-2020'].reset_index(drop=True).copy()



statsDF = pd.DataFrame()
statsDF['Batter'] = temp_statsDF.loc[0:20]['Batter'].copy()
statsDF['BattingOrder'] = temp_statsDF.loc[0:20]['BattingOrder'].copy()
statsDF['Date'] = '04-04-2021'
statsDF['AwayTeam'] = 'New York Yankees'
statsDF['HomeTeam'] = 'Toronto Blue Jays'

statsDF['A_SeaWinPct'] = 0.5
statsDF['H_SeaWinPct'] = 0.5

statsDF['A_SP_recFIP'] = 0.5
statsDF['H_SP_recFIP'] = 0.5












sys.exit()



# Separate Season win pct into hitter's team winpct and opposing team winpct
statsDF['TE_SeaWinPct'] = statsDF.apply(lambda x: x['H_SeaWinPct'] if x['HomeOrAway'] == 'Home' else x['A_SeaWinPct'], axis=1)
statsDF['OP_SeaWinPct'] = statsDF.apply(lambda x: x['H_SeaWinPct'] if x['HomeOrAway'] == 'Away' else x['A_SeaWinPct'], axis=1)

# Separate recent FIP of pitchers into hitter's team pitcher and opposing pitcher
statsDF['OP_SP_recFIP'] = statsDF.apply(lambda x: x['A_SP_recFIP'] if x['HomeOrAway'] == 'Home' else x['H_SP_recFIP'], axis=1)
statsDF['TE_SP_recFIP'] = statsDF.apply(lambda x: x['H_SP_recFIP'] if x['HomeOrAway'] == 'Home' else x['A_SP_recFIP'], axis=1)

# Add Columns defining hitters before and after current hitter in the batting order
statsDF = assorted_funcs.battingOrderVars(statsDF)

# Add Columns defining recwOBA of current hitter and hitter before and after current hitter in the batting order
statsDF = assorted_funcs.getrecwOBA(statsDF)

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

kbins = KBinsDiscretizer(n_bins=8,encode='ordinal',strategy='uniform')
b_featuresDF = kbins.fit_transform(statsDF[binnedCols])
b_featuresDF = pd.DataFrame(data=b_featuresDF,columns=statsDF[binnedCols].columns)
featuresDF = pd.concat([b_featuresDF, statsDF[nonBinnedCols]], axis=1, sort=False)



featuresDF = pd.get_dummies(featuresDF)
features_list = list(featuresDF.columns)

test_features = np.array(featuresDF)

# Scale Testing Features
test_features = loaded_scaler.transform(test_features)

                    
                    
# Make predictions
predictions = loaded_model.predict(test_features)
predictions = [round(x,2) for x in predictions]