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
# sklearn functions
from sklearn.preprocessing import MinMaxScaler, KBinsDiscretizer
from sklearn.linear_model import LinearRegression, Ridge, PoissonRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from imblearn.over_sampling import SMOTE,SMOTENC

from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingRegressor
# My functions
import get_mlb_playerstats
import assorted_funcs
import pickle
from create_kfolds import create_kfolds
from joining_dfs import combine_df_hitterdkpts
import relevant_statLists

## INITIAL LOADING AND CLEANING
# Load game data
statsDF = pd.DataFrame()
for yeari in range(2017,2021):
    curYear_DF = combine_df_hitterdkpts(yeari)
    statsDF = pd.concat([statsDF, curYear_DF], ignore_index=True)

# Remove lines in DF with NaN in Batter column - usually playoff games
statsDF = statsDF[~statsDF['Batter'].isna()].copy()
# Convert month variable to string because relationship between months and pts might not be linear
statsDF['month'] = statsDF['month'].astype(str)
# Convert certain numerical columns to strings when you want them to be treated as categorical
#statsDF['PlayedYest'] = statsDF.apply(lambda x: 0 if x['APlayedYest'] == x['HPlayedYest'] else -1 if x['APlayedYest'] > x['HPlayedYest'] else 1, axis=1)

#statsDF = statsDF[statsDF['BattingOrder'] <= 6].copy()
statsDF = statsDF[statsDF['DKPts'] <= 30].copy()
#Only one game above 105 degrees
statsDF = statsDF[statsDF['temperature'] <= 105].copy()
#statsDF = statsDF.dropna(axis=0,how='any').copy()

# Separate Season win pct into hitter's team winpct and opposing team winpct
statsDF['TE_SeaWinPct'] = statsDF.apply(lambda x: x['H_SeaWinPct'] if x['HomeOrAway'] == 'Home' else x['A_SeaWinPct'], axis=1)
statsDF['OP_SeaWinPct'] = statsDF.apply(lambda x: x['H_SeaWinPct'] if x['HomeOrAway'] == 'Away' else x['A_SeaWinPct'], axis=1)
statsDF = statsDF[(statsDF['TE_SeaWinPct'] < 70) & (statsDF['TE_SeaWinPct'] > 25)].copy()
statsDF = statsDF[(statsDF['OP_SeaWinPct'] < 70) & (statsDF['OP_SeaWinPct'] > 25)].copy()



# Separate recent FIP of pitchers into hitter's team pitcher and opposing pitcher
statsDF['OP_SP_recFIP'] = statsDF.apply(lambda x: x['A_SP_recFIP'] if x['HomeOrAway'] == 'Home' else x['H_SP_recFIP'], axis=1)
statsDF['TE_SP_recFIP'] = statsDF.apply(lambda x: x['H_SP_recFIP'] if x['HomeOrAway'] == 'Home' else x['A_SP_recFIP'], axis=1)
statsDF = statsDF[(statsDF['OP_SP_recFIP'] < 5) & (statsDF['OP_SP_recFIP'] > -2)].copy()
statsDF = statsDF[(statsDF['TE_SP_recFIP'] < 5) & (statsDF['TE_SP_recFIP'] > -2)].copy()


# Add Columns defining hitters before and after current hitter in the batting order
statsDF = assorted_funcs.battingOrderVars(statsDF)

# Add Columns defining recwOBA of current hitter and hitter before and after current hitter in the batting order
statsDF = assorted_funcs.getrecwOBA(statsDF)
statsDF = statsDF[statsDF['Batter_recwOBA']<0.8]
statsDF = statsDF[statsDF['Batter-1_recwOBA']<0.8]
statsDF = statsDF[statsDF['Batter+1_recwOBA']<0.8]

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

# Transform categorical label into binary
#statsDF[labelCol] = pd.Categorical(pd.factorize(statsDF[labelCol])[0])

# Get the label values as a numpy array
labels = np.array(statsDF[labelCol])

# Classify all numeric data into bins
nonStatsColumns = [x for x in statsDF.columns if 'prevY' not in x]
for coli in nonStatsColumns:
    if coli == labelCol: pass
    elif ((statsDF[coli].dtypes == 'int64') or (statsDF[coli].dtypes == 'float64')) and coli not in ['year']:
        #statsDF[coli] = pd.qcut(statsDF[coli], 10, labels=False, duplicates='drop')
        #statsDF[coli] = statsDF[coli].astype(str)
        statsDF[coli].fillna(statsDF[coli].mean(), inplace=True)

# CREATE DATAFRAME WITH FEATURES THAT WILL BE INPUTTED INTO MODEL
useful_features = []
#useful_features = [x for x in statsDF.columns if 'prevY' in x]
#useful_features.extend(['BattingOrder'])
useful_features = ['BattingOrder']
useful_features.extend([x for x in statsDF.columns if 'ParkAdj' in x])
useful_features.extend(['temperature'])
#useful_features.extend(['HomeOrAway'])
useful_features.extend(['HomeOdds','OverUnder'])
useful_features.extend(['Batter_recwOBA','Batter-1_recwOBA','Batter+1_recwOBA'])
#useful_features.extend(['BatterHand','HandMatchup'])
useful_features.extend(['BABIP_zips','ISO_zips'])
useful_features.extend(['OP_BABIP_zips','OP_ERA-_zips'])
useful_features.extend(['OP_SP_recFIP','TE_SP_recFIP'])
useful_features.extend(['TE_SeaWinPct','OP_SeaWinPct'])
useful_features.extend(['BP_AVG'])
#useful_features.extend([x for x in statsDF.columns if 'GB*WS' in x])


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
# save the KBinsDiscretizer to disk
pickle.dump(kbins, open('kbins.pkl', 'wb'))


featuresDF = pd.get_dummies(featuresDF)
features_list = list(featuresDF.columns)



# Look through k-folds, each time holding out one fold for testing
print('Modelling...')
train_features, test_features, train_labels, test_labels = train_test_split(np.array(featuresDF), labels, test_size = 0.1)

# Scale Training Features
scaler = MinMaxScaler()
scaler.fit(train_features)
train_features = scaler.transform(train_features)
test_features = scaler.transform(test_features)

# Fit the model
# Train on all data
rf = assorted_funcs.random_forest_reg(train_features, train_labels)
#rf = assorted_funcs.gbtregressor(train_features, train_labels)
#rf = LinearRegression().fit(train_features, train_labels)
#rf = Ridge(alpha=0.5).fit(train_features, train_labels)
#rf = PoissonRegressor(alpha=0.001).fit(train_features, train_labels)
#rf = assorted_funcs.histregressor(train_features, train_labels)
                    
                    
# Make predictions
predictions = rf.predict(test_features)


# Evaluate correlation between predictions and true values
#resCor = np.corrcoef(predictions,test_labels)



# Feature Importances
importances = rf.feature_importances_
std = np.std([tree.feature_importances_ for tree in rf.estimators_],
             axis=0)
indices = np.argsort(importances)[::-1]

# Print the feature ranking
print("Feature ranking:")

for f in range(train_features.shape[1]):
    #print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))
    print(features_list[indices[f]], ' ', round(importances[indices[f]],2))

    


m, b = np.polyfit(test_labels, predictions, 1)
plt.plot(test_labels, predictions, 'o')
plt.plot(test_labels, m*test_labels + b)




# Plot average prediction by test_value
pred_DF = pd.DataFrame()
pred_DF['predictions'] = predictions
pred_DF['test_labels'] = test_labels
pred_DF.groupby('test_labels').mean().rolling(5).median().plot.line()
plt.show()

meanErr = round(np.mean(abs(test_labels-predictions)),1)
print('Mean Error:' + str(meanErr))


# save the model to disk
pickle.dump(rf, open('finalized_model.sav', 'wb'))
# save the scaler for use in testing
pickle.dump(scaler, open('scaler.pkl', 'wb'))


