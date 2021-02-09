# Import general libraries
import pandas as pd
import numpy as np
import sys
import math
import random
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

import pandas as pd
import numpy as np
import get_mlb_playerstats
import assorted_funcs

# Load Model
loaded_model = pickle.load(open('finalized_model.sav', 'rb'))
# Load Scaler
loaded_scaler = pickle.load(open('scaler.pkl', 'rb'))
# Load Modes, Means, and SDs for testing data
loaded_modes = pickle.load(open('modes.pkl', 'rb'))
loaded_lowOutliers = pickle.load(open('lowOutliers.pkl', 'rb'))
loaded_highOutliers = pickle.load(open('highOutliers.pkl', 'rb'))


# Load Batting Stats
print('Loading Batting Stats...')
batting = dict()
for yeari in range(2018,2019):
    batting[yeari] = get_mlb_playerstats.load_hitting_data(yeari)



# Load game data
inputDF = pd.DataFrame()
for yeari in range(2019,2021):
    loadGames = pd.read_csv('Input/' + str(yeari) + '_withodds.csv', sep='\t')
    inputDF = pd.concat([inputDF, loadGames], ignore_index=True)
# Create column containing year each game was played
inputDF['year'] = inputDF['date'].apply(lambda x: int(x[-4:]))

## USED WHEN TRYING TO PREDICT WINNERS
inputDF['Winner'] = inputDF.apply(lambda x: 'Away' if x['A_Score'] > x['H_Score'] else 'Home', axis=1)

# Restrict data to day you're interested in testing
#inputDF = inputDF[(inputDF['day'] == 12) & (inputDF['month'] == 9) & (inputDF['year'] == 2020)].copy()
inputDF = inputDF[inputDF['year'] == 2019].copy()


# Create a blank dataframe that will house all statistics for the model
statsDF = pd.DataFrame()






# BATTING STATS
# List of batters that you'd like included in the analysis
batterList = ['A_1','A_2','A_3','A_4','A_5','A_6','A_7','A_8','A_9','H_1','H_2','H_3','H_4','H_5','H_6','H_7','H_8','H_9']
# Compile list of statistics by removing irrelevant column names from column list
battingStatsColumns = [ elem for elem in list(batting[2020].columns) if elem not in ['Season','Team']]


# Create columns in stats DataFrame that include each corresponding players stats from current and past years
relevantBatStats = relStatsLists.batterStatList()
# Loop through each year, batter and statistic
for yeari in ['prevY','twoPrevY']:
    for bati in batterList:
        for stati in battingStatsColumns:
            curStat = stati + '_' + bati
            if curStat in ["_".join(x.split("_", 3)[:-1]) for x in relevantBatStats]:
                print(curStat)
                # Create a column that contains the statistical value associated with each corresponding pitcher
                statsDF[stati + '_' + bati + '_' + yeari] = inputDF.apply(lambda x: assorted_functions.populatePlayerStats(batting, x, bati, stati, yeari),axis=1)
                # Replace any outliers with the mode from that column
                lowOutlier = loaded_lowOutliers[stati + '_' + bati + '_' + yeari]
                highOutlier = loaded_highOutliers[stati + '_' + bati + '_' + yeari]
                statsDF.at[statsDF[stati + '_' + bati + '_' + yeari] < lowOutlier, stati + '_' + bati + '_' + yeari] = loaded_modes[stati + '_' + bati + '_' + yeari]
                statsDF.at[statsDF[stati + '_' + bati + '_' + yeari] > highOutlier, stati + '_' + bati + '_' + yeari] = loaded_modes[stati + '_' + bati + '_' + yeari]
                # Fill any NaN values with the mode from that column
                statsDF[stati + '_' + bati + '_' + yeari].fillna(loaded_modes[stati + '_' + bati + '_' + yeari], inplace=True)


# Create columns in stats DataFrame that include each corresponding players stats from current and past years
#relevantBatStats = relStatsLists.batterStatList()
# Loop through each year, batter and statistic
for yeari in ['prevY']:
    for bati in batterList:
        for stati in battingStatsColumns:
            #curStat = stati + '_' + bati:
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







# SET LABEL COLUMN AND ADD IT TO THE STATS DATAFRAME
labelCol = 'Winner'

# Transform categorical label into binary
statsDF[labelCol] = inputDF[labelCol].copy()

# Get the label values as a numpy array
labels = np.array(statsDF[labelCol])

# CREATE DATAFRAME WITH FEATURES THAT WILL BE INPUTTED INTO MODEL
featuresDF = pd.DataFrame()
#featuresDF['Season'] = statsDF['Season'].copy()
featuresDF = pd.concat([featuresDF, statsDF[relevantBatStats]], axis=1, sort=False)
featuresDF = pd.concat([featuresDF, statsDF[relevantPitchStats]], axis=1, sort=False)


features_list = list(featuresDF.columns)


# Transform features into numpy array
features_toModel = np.array(featuresDF)
# Scale Features
scaler = loaded_scaler
scaler.fit(features_toModel)
features_toModel = scaler.transform(features_toModel)

print('Modelling...')
# Fit the model
rf = loaded_model


# Make predictions
predictions = rf.predict_proba(features_toModel)

# Transform predictions into binary
predictions_binary = np.array([0 if x[0] >= 0.5 else 1 for x in predictions])

print('Accuracy: ', round(sum(predictions_binary == labels)/len(labels),2))






predDF = pd.DataFrame()
predDF['predA'] = [x[0] for x in predictions]
predDF['Winner'] = inputDF['Winner'].values
predDF = predDF.sort_values(by=['predA'],ascending=False)
predDF
