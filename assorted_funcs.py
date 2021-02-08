# Assorted functions that don't fit in a clear group


import numpy as np
from scipy.stats import ks_2samp
from sklearn.ensemble import RandomForestClassifier

# Function to check if player name is part of dataframe index
def isPlayerInDF(df, player):
    try:
        df.loc[player]
        return True
    except KeyError:
        return False
    

# Function used to populate column of dataframe with specific statistic of a position for a given year
def populatePlayerStats(df, gamei, position, stat, year):
    if year == 'prevY':
        statYear = gamei['year'] - 1
    else:
        statYear = gamei['year'] - 2
    
    try:
        # If statement covers situations where player name appears multiple times
        if len(df[statYear].loc[gamei[position]][stat]) > 1:
            return np.NaN
    except:
        try:
            return df[statYear].loc[gamei[position]][stat]
        except:
            return np.NaN
    
    
    
    
def combineStatsAcrossPlayers(curStat, curDF, labelArray, homeOrAway):
    tempStat = np.zeros(np.shape(curDF)[0])
    for starti in range(1,10):
        if starti > 1: tempStat = np.zeros(np.shape(curDF)[0])
        for endi in range(starti,10):
            for combi in range(1,endi+1):
                tempStat = tempStat + curDF[curStat + '_' + homeOrAway + '_' + str(combi)].values
            
            tempStat = tempStat/endi
            
            correlation_matrix = np.corrcoef(tempStat, labelArray)
            correlation_xy = correlation_matrix[0,1]
            r_squared = correlation_xy**2
            print(str(round(r_squared,3)) + '\t' + str(starti) + '\t' + str(endi))
            


# Function used to calculate if a feature successfully differentiates between groups
def doesVarSeparateGroups(df1,df2,curFeat):
    x = ks_2samp(df1[curFeat], df2[curFeat]).statistic
    #x = ks_2samp(df1[curFeat], df2[curFeat]).pvalue
    #print(x)
    x = round(x,4)
    return x

def random_forest(
    X_train, 
    y_train,
    n_estimators=1000,
    min_samples_split=2, 
    max_leaf_nodes=None, 
    max_features='auto', 
    max_depth=10, 
    bootstrap=True
    ):
    # making the RandomForestRegressor paramteres changable for hyperparameter optimization
    regr = RandomForestClassifier(
        n_estimators = n_estimators, 
        min_samples_split=min_samples_split, 
        max_leaf_nodes=max_leaf_nodes,
        max_features=max_features,
        max_depth=max_depth,
        bootstrap=bootstrap
        )

    regr.fit(X_train, y_train)
    return regr