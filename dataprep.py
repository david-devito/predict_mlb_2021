## DATA CLEANING AND FEATURE CREATION

# Add paths of additional scripts
import sys
sys.path.append('./data_scraping')
sys.path.append('./function_scripts')

## IMPORT
# Python packages
import pandas as pd
import numpy as np
# sklearn functions
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
# My functions
import get_mlb_playerstats
import assorted_funcs
import pickle
from create_kfolds import create_kfolds



#for curBatStati in list(range(6,45)):

## INITIAL LOADING AND CLEANING
# Load game data
statsDF = pd.DataFrame()
for yeari in range(2019,2020):
    loadGames = pd.read_csv('input/gamelogs/gamelogs' + str(yeari) + '.csv', sep=',')
    statsDF = pd.concat([statsDF, loadGames], ignore_index=True)
# Create column containing year each game was played
statsDF['year'] = statsDF['Date'].apply(lambda x: int(x[-4:]))
# Calculate difference in WinPct
statsDF['WinPct_Diff'] = statsDF.apply(lambda x: x['H_WinPct'] - x['A_WinPct'], axis=1)
# Convert certain numerical columns to strings when you want them to be treated as categorical
statsDF['PlayedYest'] = statsDF.apply(lambda x: 0 if x['APlayedYest'] == x['HPlayedYest'] else -1 if x['APlayedYest'] > x['HPlayedYest'] else 1, axis=1)
statsDF['precipitation'] = statsDF['precipitation'].apply(lambda x: x.replace(' ','').replace('.',''))

# Create target variable
statsDF['Winner'] = statsDF.apply(lambda x: 'Away' if x['AwayScore'] > x['HomeScore'] else 'Home', axis=1)
# Split dataset into stratified k-folds based on target variable
statsDF = create_kfolds(statsDF,5,'Winner')

## LOAD STATISTICS
# Load Batting Stats
#print('Loading Batting Stats...')
batting = dict()
#batStatsCols = [5,curBatStati]
batStatsCols = [5]
batStatsCols.extend([43,48,56,58,68,71,72,76,78,79,80,93,104,105])
for yeari in range(2018,2020):
    batting[yeari] = get_mlb_playerstats.load_hitting_data(yeari,batStatsCols)


#relStatsList = ['+WPA_A_1_prevY','+WPA_A_1_prevY']
#relStatsList.append(list(batting[list(batting.keys())[0]].columns)[-1] + '_A_2_prevY')
#relStatsList.append(list(batting[list(batting.keys())[0]].columns)[-1] + '_H_2_prevY')

# Create dictionaries to store Modes, Means, and SDs for testing data
modeTestingList = dict()
lowOutlierTestingList = dict()
highOutlierTestingList = dict()

# BATTING STATS
# List of batters that you'd like included in the analysis
#batterList = ['A_1','A_2','A_3','A_4','A_5','A_6','A_7','A_8','A_9','H_1','H_2','H_3','H_4','H_5','H_6','H_7','H_8','H_9']
#batterList = ['A_1','A_2','A_3','A_4','A_5']
batterList = ['A_1','A_2','A_3','A_4','A_5','H_1','H_2','H_3','H_4','H_5']
# Compile list of statistics by removing irrelevant column names from column list
battingStatsColumns = [ elem for elem in list(batting[list(batting.keys())[0]].columns) if elem not in ['Season','Team']]


# Create columns in stats DataFrame that include each corresponding players stats from current and past years
print(battingStatsColumns)
# Loop through each year, batter and statistic
for yeari in ['prevY']:
    for bati in batterList:
        print(bati)
        for stati in battingStatsColumns:
            #if stati + '_' + bati + '_' + yeari in relStatsList:
            #curStat = stati + '_' + bati
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
X = statsDF.columns[['prevY' in x for x in statsDF.columns]]
for yeari in ['prevY']:
    for stati in battingStatsColumns:
        Acol = X[[stati + '_A' in x for x in X]]
        statsDF[stati + '_A_avg_' + yeari] = statsDF[Acol].mean(axis=1)
        Hcol = X[[stati + '_H' in x for x in X]]
        statsDF[stati + '_H_avg_' + yeari] = statsDF[Hcol].mean(axis=1)


# Save Modes, Means, and SDs for testing data
#pickle.dump(modeTestingList, open('modes.pkl', 'wb'))
#pickle.dump(lowOutlierTestingList, open('lowOutliers.pkl', 'wb'))
#pickle.dump(highOutlierTestingList, open('highOutliers.pkl', 'wb'))

# SET LABEL COLUMN AND ADD IT TO THE STATS DATAFRAME
labelCol = 'Winner'

# Transform categorical label into binary
statsDF[labelCol] = pd.Categorical(pd.factorize(statsDF[labelCol])[0])

# Get the label values as a numpy array
labels = np.array(statsDF[labelCol])

# Get kfolds as a numpy array
kfolds = np.array(statsDF['kfold'])

# CREATE DATAFRAME WITH FEATURES THAT WILL BE INPUTTED INTO MODEL
useful_features = [x for x in statsDF.columns if 'avg_prevY' in x]
useful_features.extend(['WinPct_Diff','PlayedYest'])
useful_features.extend(['temperature','windspeed','precipitation'])
featuresDF = pd.DataFrame()
featuresDF = pd.concat([featuresDF, statsDF[useful_features]], axis=1, sort=False)
featuresDF = pd.get_dummies(featuresDF)

features_list = list(featuresDF.columns)

#sys.exit()

# Look through k-folds, each time holding out one fold for testing
print('Modelling...')
accArray = []
for curFold in np.unique(statsDF['kfold']):

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
    
    # Fit the model
    # Train on all data
    rf = assorted_funcs.random_forest(train_features, train_labels)
    
    # Make predictions
    predictions = rf.predict_proba(test_features)
    
    # Transform predictions into binary
    predictions_binary = np.array([0 if x[0] >= 0.5 else 1 for x in predictions])
    
    curACC = round(sum(predictions_binary == test_labels)/len(test_labels),2)
    accArray.append(curACC)
    #print('Accuracy: ', curACC)


print('MEAN ACCURACY: ', np.mean(accArray))


sys.exit()

# Split Data into Training and Testing Set
train_features, test_features, train_labels, test_labels = train_test_split(features_toModel, labels, test_size = 0.10)


print('Modelling...')
# Fit the model
#rf = assorted_functions.random_forest(train_features, train_labels)
# Train on all data
rf = assorted_funcs.random_forest(features_toModel, labels)

# Make predictions
predictions = rf.predict_proba(test_features)

# Transform predictions into binary
predictions_binary = np.array([0 if x[0] >= 0.5 else 1 for x in predictions])

print('Accuracy: ', round(sum(predictions_binary == test_labels)/len(test_labels),2))



# Feature Importances
importances = rf.feature_importances_
std = np.std([tree.feature_importances_ for tree in rf.estimators_],
             axis=0)
indices = np.argsort(importances)[::-1]

# Print the feature ranking
print("Feature ranking:")

for f in range(features_toModel.shape[1]):
    #print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))
    print(features_list[indices[f]], ' ', round(importances[indices[f]],2))


# save the model to disk
#pickle.dump(rf, open('finalized_model.sav', 'wb'))
# save the scaler for use in testing
#pickle.dump(scaler, open('scaler.pkl', 'wb'))