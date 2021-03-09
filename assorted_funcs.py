# Assorted functions that don't fit in a clear group


import numpy as np
from scipy.stats import ks_2samp
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, RandomForestRegressor, GradientBoostingRegressor

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
    elif year == 'twoPrevY':
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
    n_estimators=750,
    min_samples_split=2, 
    max_leaf_nodes=None, 
    max_features='auto', 
    max_depth=8, 
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

def gbtclassifier(
    X_train, 
    y_train,
    n_estimators=750,
    validation_fraction=0.2, 
    n_iter_no_change=5, 
    tol=0.01
    ):
    # making the RandomForestRegressor paramteres changable for hyperparameter optimization
    regr = GradientBoostingClassifier(
        n_estimators = n_estimators, 
        validation_fraction=validation_fraction, 
        n_iter_no_change=n_iter_no_change,
        tol=tol
        )

    regr.fit(X_train, y_train)
    return regr


def random_forest_reg(
    X_train, 
    y_train,
    n_estimators=100,
    min_samples_split=6, 
    max_leaf_nodes=None, 
    max_features='auto', 
    max_depth=20, 
    bootstrap=True
    ):
    # making the RandomForestRegressor paramteres changable for hyperparameter optimization
    regr = RandomForestRegressor(
        n_estimators = n_estimators, 
        min_samples_split=min_samples_split, 
        max_leaf_nodes=max_leaf_nodes,
        max_features=max_features,
        max_depth=max_depth,
        bootstrap=bootstrap
        )

    regr.fit(X_train, y_train)
    return regr

def gbtregressor(
    X_train, 
    y_train,
    n_estimators=100,
    min_samples_split=2, 
    max_leaf_nodes=None, 
    max_features='auto', 
    max_depth=4
    ):
    # making the RandomForestRegressor paramteres changable for hyperparameter optimization
    regr = GradientBoostingRegressor(
        n_estimators = n_estimators, 
        min_samples_split=min_samples_split, 
        max_leaf_nodes=max_leaf_nodes,
        max_features=max_features,
        max_depth=max_depth
        )

    regr.fit(X_train, y_train)
    return regr




def battingOrderVars(df):
    oneBefore = {1:9,2:1,3:2,4:3,5:4,6:5,7:6,8:7,9:8}
    df['Batter-1'] = df.apply(lambda x: x[x['HomeOrAway'][0] + '_' + str(int(oneBefore[x['BattingOrder']]))],axis=1)
    oneAfter = {1:2,2:3,3:4,4:5,5:6,6:7,7:8,8:9,9:1}
    df['Batter+1'] = df.apply(lambda x: x[x['HomeOrAway'][0] + '_' + str(int(oneAfter[x['BattingOrder']]))],axis=1)
    
    
    return df
    
def getrecwOBA(df):
    
    df['Batter_recwOBA'] = df.apply(lambda x: x[x['HomeOrAway'][0] + '_' + str(int(x['BattingOrder'])) + '_recwOBA'], axis=1)
    
    oneBefore = {1:9,2:1,3:2,4:3,5:4,6:5,7:6,8:7,9:8}
    df['Batter-1_recwOBA'] = df.apply(lambda x: x[x['HomeOrAway'][0] + '_' + str(int(oneBefore[x['BattingOrder']])) + '_recwOBA'], axis=1)
    
    oneAfter = {1:2,2:3,3:4,4:5,5:6,6:7,7:8,8:9,9:1}
    df['Batter+1_recwOBA'] = df.apply(lambda x: x[x['HomeOrAway'][0] + '_' + str(int(oneAfter[x['BattingOrder']])) + '_recwOBA'], axis=1)
    
    
    return df
    
    





