## FUNCTION TO SPLIT DATASET INTO STRATIFIED K-FOLDS BASED ON TARGET VARIABLE

from sklearn.model_selection import StratifiedKFold

def create_kfolds(df,numFolds,targetVar):
    
    # Create column that will be populated with K-Folds
    df['kfold'] = -1
    
    # Randomize the dataframe rows
    df = df.sample(frac=1).reset_index(drop=True)
    
    y = df[targetVar].values
    
    kf = StratifiedKFold(n_splits=numFolds)
    
    # Assign K-Folds
    for f, (t_, v_) in enumerate(kf.split(X=df, y=y)):
        df.loc[v_, 'kfold'] = f
    
    return df