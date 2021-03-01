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
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from imblearn.over_sampling import SMOTE,SMOTENC
# My functions
import get_mlb_playerstats
import assorted_funcs
import pickle
from create_kfolds import create_kfolds
from joining_dfs import combine_df
import relevant_statLists


#for curBatStati in list(range(6,45)):

## INITIAL LOADING AND CLEANING
# Load game data
statsDF = pd.DataFrame()
for yeari in range(2010,2021):
    curYear_DF = combine_df(yeari)
    #loadGames = pd.read_csv('input/gamelogs/gamelogs' + str(yeari) + '.csv', sep=',')
    statsDF = pd.concat([statsDF, curYear_DF], ignore_index=True)

# Calculate difference in WinPct
#statsDF['SeaWinPct_Diff'] = statsDF.apply(lambda x: x['H_SeaWinPct'] - x['A_SeaWinPct'], axis=1)
#statsDF['last3WinPct_Diff'] = statsDF.apply(lambda x: x['H_last3WinPct'] - x['A_last3WinPct'], axis=1)
#statsDF['last5WinPct_Diff'] = statsDF.apply(lambda x: x['H_last5WinPct'] - x['A_last5WinPct'], axis=1)
#statsDF['last10WinPct_Diff'] = statsDF.apply(lambda x: x['H_last10WinPct'] - x['A_last10WinPct'], axis=1)

# Convert certain numerical columns to strings when you want them to be treated as categorical
#statsDF['PlayedYest'] = statsDF.apply(lambda x: 0 if x['APlayedYest'] == x['HPlayedYest'] else -1 if x['APlayedYest'] > x['HPlayedYest'] else 1, axis=1)

# Create target variable
statsDF['Winner'] = statsDF.apply(lambda x: 'Away' if x['AwayScore'] > x['HomeScore'] else 'Home', axis=1)
# Split dataset into stratified k-folds based on target variable
statsDF = create_kfolds(statsDF,5,'Winner')


## LOAD STATISTICS
# Load Batting Stats
#print('Loading Batting Stats...')
batting = dict()
batStatsCols = [5]
batStatsCols.extend([53,62,66])
#batStatsCols.extend(list(range(290,312)))
for yeari in range(2009,2020):
    batting[yeari] = get_mlb_playerstats.load_hitting_data(yeari,batStatsCols)
    

#print('Loading Pitching Stats...')
pitching = dict()
pitchStatsCols = [13]
pitchStatsCols.extend([62,109,217,221,240,284])#start at 326
for yeari in range(2009,2019):
    pitching[yeari] = get_mlb_playerstats.load_pitching_data(yeari,pitchStatsCols)

'''
# Create dictionaries to store Modes, Means, and SDs for testing data
modeTestingList = dict()
lowOutlierTestingList = dict()
highOutlierTestingList = dict()
'''
# BATTING STATS
# List of batters that you'd like included in the analysis
#batterList = ['A_1','A_2','A_3','A_4','A_5','A_6','A_7','A_8','A_9','H_1','H_2','H_3','H_4','H_5','H_6','H_7','H_8','H_9']
batterList = ['A_1','A_2','A_3']
#batterList = ['A_1','A_2','A_3','A_4','A_5','H_1','H_2','H_3','H_4','H_5']
# Compile list of statistics by removing irrelevant column names from column list
battingStatsColumns = [ elem for elem in list(batting[list(batting.keys())[0]].columns) if elem not in ['Season','Team']]


# Create columns in stats DataFrame that include each corresponding players stats from current and past years
relevantBatStats = relevant_statLists.batterStatList()
# Loop through each year, batter and statistic
for yeari in ['prevY']:
    for bati in batterList:
        print(bati)
        for stati in battingStatsColumns:
            if stati + '_' + bati + '_' + yeari in relevantBatStats:
                print(stati)
                # Create a column that contains the statistical value associated with each corresponding hitter
                statsDF[stati + '_' + bati + '_' + yeari] = statsDF.apply(lambda x: assorted_funcs.populatePlayerStats(batting, x, bati, stati, yeari),axis=1)
                # Replace any outliers with the mode from that column
                curMean = np.mean(statsDF[stati + '_' + bati + '_' + yeari])
                curSTD = np.std(statsDF[stati + '_' + bati + '_' + yeari])
                lowOutlier = curMean - (3*curSTD)
                highOutlier = curMean + (3*curSTD)
                statsDF.at[statsDF[stati + '_' + bati + '_' + yeari] < lowOutlier, stati + '_' + bati + '_' + yeari] = statsDF[stati + '_' + bati + '_' + yeari].mode()[0]
                statsDF.at[statsDF[stati + '_' + bati + '_' + yeari] > highOutlier, stati + '_' + bati + '_' + yeari] = statsDF[stati + '_' + bati + '_' + yeari].mode()[0]
                # Fill any NaN values with the mode from that column
                statsDF[stati + '_' + bati + '_' + yeari].fillna(statsDF[stati + '_' + bati + '_' + yeari].mode()[0], inplace=True)
                # Save Mode for Future Testing
                #modeTestingList[stati + '_' + bati + '_' + yeari] = statsDF[stati + '_' + bati + '_' + yeari].mode()[0]
                # Save Low and High Outlier Values for Future Testing
                #lowOutlierTestingList[stati + '_' + bati + '_' + yeari] = lowOutlier
                #highOutlierTestingList[stati + '_' + bati + '_' + yeari] = highOutlier

# Combine stats across hitters
#X = statsDF.columns[['prevY' in x for x in statsDF.columns]]
#for yeari in ['prevY']:
#    for stati in battingStatsColumns:
#        Acol = X[[stati + '_A' in x for x in X]]
#        statsDF[stati + '_A_avg_' + yeari] = statsDF[Acol].mean(axis=1)
#        Hcol = X[[stati + '_H' in x for x in X]]
#        statsDF[stati + '_H_avg_' + yeari] = statsDF[Hcol].mean(axis=1)
'''

# Calculate FB - GB Pitcher Difference and Add Effect of Wind Speed
#statsDF['H_FB-GB*WS_H'] = (statsDF['FB%_H_avg_prevY'] - statsDF['GB%_H_avg_prevY']) * statsDF['windspeed']
#statsDF['H_FB-GB*WS_A'] = (statsDF['FB%_A_avg_prevY'] - statsDF['GB%_A_avg_prevY']) * statsDF['windspeed']
#statsDF = statsDF.drop(['GB%_H_avg_prevY','GB%_A_avg_prevY'],axis=1)



# Recent wOBA Stats
statsDF['H_recwOBA_1-3'] = statsDF.apply(lambda x: (x['H_1_recwOBA'] + x['H_2_recwOBA'] + x['H_3_recwOBA'])/3, axis=1)
statsDF['H_recwOBA_4-6'] = statsDF.apply(lambda x: (x['H_4_recwOBA'] + x['H_5_recwOBA'] + x['H_6_recwOBA'])/3, axis=1)
statsDF['H_recwOBA_7-9'] = statsDF.apply(lambda x: (x['H_7_recwOBA'] + x['H_8_recwOBA'] + x['H_9_recwOBA'])/3, axis=1)
statsDF['A_recwOBA_1-3'] = statsDF.apply(lambda x: (x['A_1_recwOBA'] + x['A_2_recwOBA'] + x['A_3_recwOBA'])/3, axis=1)
statsDF['A_recwOBA_4-6'] = statsDF.apply(lambda x: (x['A_4_recwOBA'] + x['A_5_recwOBA'] + x['A_6_recwOBA'])/3, axis=1)
statsDF['A_recwOBA_7-9'] = statsDF.apply(lambda x: (x['A_7_recwOBA'] + x['A_8_recwOBA'] + x['A_9_recwOBA'])/3, axis=1)



'''
# PITCHING STATS
# List of pitchers that you'd like included in the analysis
pitcherList = ['AwaySP','HomeSP']
#pitcherList = ['HomeSP']
# Compile list of statistics by removing irrelevant column names from column list
pitchingStatsColumns = [ elem for elem in list(pitching[list(pitching.keys())[0]].columns) if elem not in ['Season','Team']]


# Create columns in stats DataFrame that include each corresponding players stats from current and past years
print(pitchingStatsColumns)
# Loop through each year, batter and statistic
for yeari in ['prevY']:
    for pitchi in pitcherList:
        print(pitchi)
        for stati in pitchingStatsColumns:
            print(stati)
            # Create a column that contains the statistical value associated with each corresponding hitter
            statsDF[stati + '_' + pitchi + '_' + yeari] = statsDF.apply(lambda x: assorted_funcs.populatePlayerStats(pitching, x, pitchi, stati, yeari),axis=1)
            # Replace any outliers with the mode from that column
            curMean = np.mean(statsDF[stati + '_' + pitchi + '_' + yeari])
            curSTD = np.std(statsDF[stati + '_' + pitchi + '_' + yeari])
            lowOutlier = curMean - (3*curSTD)
            highOutlier = curMean + (3*curSTD)
            statsDF.at[statsDF[stati + '_' + pitchi + '_' + yeari] < lowOutlier, stati + '_' + pitchi + '_' + yeari] = statsDF[stati + '_' + pitchi + '_' + yeari].mode()[0]
            statsDF.at[statsDF[stati + '_' + pitchi + '_' + yeari] > highOutlier, stati + '_' + pitchi + '_' + yeari] = statsDF[stati + '_' + pitchi + '_' + yeari].mode()[0]
            # Fill any NaN values with the mode from that column
            statsDF[stati + '_' + pitchi + '_' + yeari].fillna(statsDF[stati + '_' + pitchi + '_' + yeari].mode()[0], inplace=True)
            # Save Mode for Future Testing
            #modeTestingList[stati + '_' + pitchi + '_' + yeari] = statsDF[stati + '_' + pitchi + '_' + yeari].mode()[0]
            # Save Low and High Outlier Values for Future Testing
            #lowOutlierTestingList[stati + '_' + pitchi + '_' + yeari] = lowOutlier
            #highOutlierTestingList[stati + '_' + pitchi + '_' + yeari] = highOutlier

'''
# Calculate FB - GB Pitcher Difference and Add Effect of Wind Speed
statsDF['P_FB-GB*WS_H'] = (statsDF['FB%_AwaySP_prevY'] - statsDF['GB%_AwaySP_prevY']) * statsDF['windspeed']
statsDF['P_FB-GB*WS_A'] = (statsDF['FB%_HomeSP_prevY'] - statsDF['FB%_HomeSP_prevY']) * statsDF['windspeed']
statsDF = statsDF.drop(['FB%_AwaySP_prevY','GB%_AwaySP_prevY','FB%_HomeSP_prevY','GB%_HomeSP_prevY'],axis=1)
'''

#statsDF['recFIP_Diff'] = statsDF.apply(lambda x: x['H_SP_recFIP'] - x['A_SP_recFIP'], axis=1)


# WEATHER
'''
statsDF['windSpeed'] = statsDF['windSpeed'].apply(lambda x: '10+' if x >= 10 else ('0' if x == 0 else '>0 + <10'))
statsDF['windDirection'] = statsDF['windDirection'].fillna('NaN')
statsDF['windDirection'] = statsDF['windDirection'].apply(lambda x: 'NoWind' if x == 'NoWind' else ('unknown' if 'unknown' in x else ('in' if 'in' in x else ('out' if 'out' in x else ('crosswind' if 'from' in x else x)))))
statsDF['windSpeedAndDir'] = statsDF.apply(lambda x: x['windSpeed'] + ' ' + x['windDirection'], axis=1)

#statsDF['windDirection'] = statsDF['windDirection'].apply(lambda x: 'unknown' if 'unknown' in x else ('in' if 'in' in x else ('out' if 'out' in x else ('NoWind' if 'NoWind' in x else 'crosswind'))))
statsDF['precipitation'] = statsDF['precipitation'].fillna('NaN')
statsDF['precipitation'] = statsDF['precipitation'].apply(lambda x: 'Rain' if 'Drizzle' in x else x)
'''
# Save Modes, Means, and SDs for testing data
#pickle.dump(modeTestingList, open('modes.pkl', 'wb'))
#pickle.dump(lowOutlierTestingList, open('lowOutliers.pkl', 'wb'))
#pickle.dump(highOutlierTestingList, open('highOutliers.pkl', 'wb'))

# REMOVE COLUMNS WITH NA VALUES
#statsDF = statsDF.dropna(axis=0, how='any').copy()

# SET LABEL COLUMN AND ADD IT TO THE STATS DATAFRAME
labelCol = 'Winner'

# Transform categorical label into binary
statsDF[labelCol] = pd.Categorical(pd.factorize(statsDF[labelCol])[0])

# Get the label values as a numpy array
labels = np.array(statsDF[labelCol])

# Get kfolds as a numpy array
kfolds = np.array(statsDF['kfold'])

# Classify all numeric data into bins
nonStatsColumns = [x for x in statsDF.columns if 'prevY' not in x]
for coli in nonStatsColumns:
    if ((statsDF[coli].dtypes == 'int64') or (statsDF[coli].dtypes == 'float64')) and coli not in ['year','kfold']:
        #statsDF[coli] = pd.qcut(statsDF[coli], 10, labels=False, duplicates='drop')
        #statsDF[coli] = statsDF[coli].astype(str)
        statsDF[coli].fillna(statsDF[coli].mode()[0], inplace=True)


# CREATE DATAFRAME WITH FEATURES THAT WILL BE INPUTTED INTO MODEL
useful_features = []
useful_features = [x for x in statsDF.columns if 'prevY' in x]
#useful_features.extend([x for x in statsDF.columns if ('WinPct_Diff' in x)])
#useful_features.extend([x for x in statsDF.columns if ('recwOBA_' in x)])
useful_features.extend([x for x in statsDF.columns if ('recFIP' in x)])
#useful_features.extend(['temperature','windSpeedAndDir','precipitation'])
#useful_features.extend([x for x in statsDF.columns if 'GB*WS' in x])

# Create Full Features DF
featuresDF_orig = pd.DataFrame()
featuresDF_orig = pd.concat([featuresDF_orig, statsDF[useful_features]], axis=1, sort=False)
# Discretize Numeric Features into Bins
#kbins = KBinsDiscretizer(n_bins=20,encode='ordinal',strategy='uniform')
#featuresDF = kbins.fit_transform(featuresDF_orig)
#featuresDF = pd.DataFrame(data=featuresDF,columns=featuresDF_orig.columns)
featuresDF = pd.get_dummies(featuresDF_orig)

features_list = list(featuresDF.columns)

## CHECKING WHICH FEATURES BEST SEPARATE THE TARGET VARIABLE
df_away = statsDF[statsDF['Winner'] == 0].copy()
df_home = statsDF[statsDF['Winner'] == 1].copy()
A_useful_features = [x for x in useful_features if 'A' in x]
for coli in A_useful_features:
    x = assorted_funcs.doesVarSeparateGroups(df_away,df_home,coli)
    print(coli + ' - ' + str(x))



pd.set_option('display.max_columns', 500)
#sys.exit()

# Look through k-folds, each time holding out one fold for testing
print('Modelling...')
accArray = []
accCutOff = 0.6
cutOffAccArray = []
cat_col_index = []
for curFold in [int(x) for x in np.unique(statsDF['kfold'])]:

    # Transform features into numpy array
    train_features = np.array(featuresDF[kfolds != curFold])
    test_features = np.array(featuresDF[kfolds == curFold])
    train_labels = labels[kfolds != curFold]
    test_labels = labels[kfolds == curFold]
    # Scale Training Features
    scaler = MinMaxScaler()
    scaler.fit(train_features)
    train_features = scaler.transform(train_features)
    test_features = scaler.transform(test_features)
    
    # SMOTENC to balance classes
    sm = SMOTE()#categorical_features=cat_col_index)
    train_features, train_labels = sm.fit_resample(train_features, train_labels)
    
    # Fit the model
    # Train on all data
    rf = assorted_funcs.random_forest(train_features, train_labels)
    #rf = assorted_funcs.gbtclassifier(train_features, train_labels)
    
    # Make predictions
    predictions = rf.predict_proba(test_features)
    
    # Transform predictions into binary
    predictions_binary = np.array([0 if x[0] >= 0.5 else 1 for x in predictions])
    
    curACC = round(sum(predictions_binary == test_labels)/len(test_labels),2)
    accArray.append(curACC)
    #print('Accuracy: ', curACC)

    # Using Accuracy Cutoff to get best predictions
    # Transform predictions into binary
    predictions_binary_cutoff = np.array([0 if x[0] > accCutOff else (1 if x[0] < (1-accCutOff) else np.nan) for x in predictions])
    ind = np.where(~np.isnan(predictions_binary_cutoff))[0]
    curACC = round(sum(predictions_binary_cutoff[ind] == test_labels[ind])/len(test_labels[ind]),2)
    cutOffAccArray.append(curACC)

print('MEAN ACCURACY: ', np.mean(accArray))
print('MEAN ACCURACY WITH CUTOFF: ', np.mean(cutOffAccArray))



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


# save the model to disk
#pickle.dump(rf, open('finalized_model.sav', 'wb'))
# save the scaler for use in testing
#pickle.dump(scaler, open('scaler.pkl', 'wb'))
    
conf_mat = confusion_matrix(y_true=test_labels[ind], y_pred=predictions_binary[ind])
print('Confusion matrix:\n', conf_mat)

labels = ['Class 0', 'Class 1']
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(conf_mat, cmap=plt.cm.Blues)
fig.colorbar(cax)
ax.set_xticklabels([''] + labels)
ax.set_yticklabels([''] + labels)
plt.xlabel('Predicted')
plt.ylabel('Expected')
plt.show()