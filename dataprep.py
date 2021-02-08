## DATA CLEANING AND FEATURE CREATION


import pandas as pd
import numpy as np
import get_mlb_playerstats
import assorted_funcs
import pickle

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# Load data
statsDF = pd.read_csv('input/gamelogs/output.csv', sep=',')
statsDF['year'] = statsDF['Date'].apply(lambda x: int(x[-4:]))


## USED WHEN TRYING TO PREDICT WINNERS
statsDF['Winner'] = statsDF.apply(lambda x: 'Away' if x['AwayScore'] > x['HomeScore'] else 'Home', axis=1)




# Load Batting Stats
print('Loading Batting Stats...')
batting = dict()
for yeari in range(2018,2019):
    batting[yeari] = get_mlb_playerstats.load_hitting_data(yeari)

# Create dictionaries to store Modes, Means, and SDs for testing data
modeTestingList = dict()
lowOutlierTestingList = dict()
highOutlierTestingList = dict()


# BATTING STATS
# List of batters that you'd like included in the analysis
batterList = ['A_1','A_2','A_3','A_4','A_5','A_6','A_7','A_8','A_9','H_1','H_2','H_3','H_4','H_5','H_6','H_7','H_8','H_9']
#batterList = ['A_1']
# Compile list of statistics by removing irrelevant column names from column list
battingStatsColumns = [ elem for elem in list(batting[list(batting.keys())[0]].columns) if elem not in ['Season','Team']]


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

# CREATE DATAFRAME WITH FEATURES THAT WILL BE INPUTTED INTO MODEL
useful_features = [x for x in statsDF.columns if 'prevY' in x]
featuresDF = pd.DataFrame()
featuresDF = pd.concat([featuresDF, statsDF[useful_features]], axis=1, sort=False)

features_list = list(featuresDF.columns)


# Transform features into numpy array
features_toModel = np.array(featuresDF)
# Scale Features
scaler = MinMaxScaler()
scaler.fit(features_toModel)
features_toModel = scaler.transform(features_toModel)

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