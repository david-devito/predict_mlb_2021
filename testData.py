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
from math import modf
import csv
import xlsxwriter
# sklearn functions
from sklearn.preprocessing import MinMaxScaler, KBinsDiscretizer
from sklearn.model_selection import train_test_split
# My functions
import assorted_funcs
from joining_dfs import combine_df_hitterdkpts

BSheaders = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/39.0.2171.95 Safari/537.36'}

curDate = '04-04-2021'
curStandingsYear = 2020

runModel = 1

# Load Model
loaded_model = pickle.load(open('finalized_model_hitter_dkpts.sav', 'rb'))
# Load Scaler
loaded_scaler = pickle.load(open('scaler_hitter_dkpts.pkl', 'rb'))
# Load KBins Discretizer
loaded_kbins = pickle.load(open('kbins_hitter_dkpts.pkl', 'rb'))
# Load means to fill nan values
loaded_fillna_means = pickle.load(open('fillna_means_hitter_dkpts.pkl', 'rb'))

print('SCRAPING ROTOGRINDERS')
statsDF = pd.DataFrame()

## DAILY LINEUPS INFO
# Team Names
r = requests.get("https://www.rotogrinders.com/lineups/mlb", headers=BSheaders)
soup = BeautifulSoup(r.content, "lxml")
teamNames = soup.find_all("div", {"class": "teams"})
awayTeamNames = [x.find_all("span", {"class": "lng"})[0].text + " " + x.find_all("span", {"class": "mascot"})[0].text for x in teamNames]
homeTeamNames = [x.find_all("span", {"class": "lng"})[1].text + " " + x.find_all("span", {"class": "mascot"})[1].text for x in teamNames]
# Starting Pitchers
startingPitchers = soup.find_all("div", {"class": "pitcher players"})
startingPitchers = [x.select('a')[0].text for x in startingPitchers]
# Starting Lineups
startingBatters = soup.find_all("span", {"class": "pname"})
startingBatters = [x.select('a')[0].text for x in startingBatters]


statsDF['Batter'] = startingBatters
statsDF['BattingOrder'] = list(range(1,10))*(int(len(startingBatters)/9))
statsDF['HomeOrAway'] = (['Away']*9+['Home']*9)*(int(len(startingBatters)/18))

oddIX = [x for x in range(len(startingPitchers)) if x % 2 == 1]
evenIX = [x for x in range(len(startingPitchers)) if x % 2 == 0]
awayStartingPitchers = [startingPitchers[x] for x in evenIX]
homeStartingPitchers = [startingPitchers[x] for x in oddIX]

def popDF(curList):
    temp = []
    for x in curList:
        temp.extend([x]*18)
    return temp

statsDF['AwayTeam'] = popDF(awayTeamNames)
statsDF['HomeTeam'] = popDF(homeTeamNames)
statsDF['AwaySP'] = popDF(awayStartingPitchers)
statsDF['HomeSP'] = popDF(homeStartingPitchers)

statsDF['Date'] = curDate


# Print CSV Used for Inputting Temperatures and Vegas Odds
if runModel == 0:
    #Open Excel Workbork
    curRow = 1
    workbook = xlsxwriter.Workbook('input/input_temperature_and_vegas.xlsx')
    worksheet = workbook.add_worksheet('tempAndVegas')
    # Write Headers
    worksheet.write(0, 1, 'temperature')
    worksheet.write(0, 2, 'HomeOdds')
    worksheet.write(0, 3, 'OverUnder')
    
    for i in awayTeamNames:
        worksheet.write(curRow, 0, i)
        curRow = curRow + 1
    # Close the workbook
    workbook.close()
    print('Temperature and Vegas Odds Sheet Generated - Enter Data')
    sys.exit()
else:
    # Load the sheet where temperature and vegas odds have been inputted
    tempAndVegas = pd.read_excel('input/input_temperature_and_vegas.xlsx', index_col=0)
    statsDF = pd.merge(statsDF, tempAndVegas,  how='left', left_on=['AwayTeam'], right_on = tempAndVegas.index)

## BATTER HANDEDNESS
batterHand_DF = pd.read_csv('input/2021_hitterhand_database.csv', sep=',')
statsDF = pd.merge(statsDF, batterHand_DF,  how='left', left_on=['Batter'], right_on = ['Batter'])
print('PLAYERS MISSING BATTER HANDEDNESS BELOW:')
print(statsDF[statsDF['BatterHand'].isna()]['Batter'].values)

## PITCHER HANDEDNESS
pitcherHand_DF = pd.read_csv('input/2021_pitcherhand_database.csv', sep=','); pitcherHand_DF.set_index('Pitcher',inplace=True)
statsDF['A_SP_Hand'] = statsDF['AwaySP'].apply(lambda x: pitcherHand_DF.loc[x])
statsDF['H_SP_Hand'] = statsDF['HomeSP'].apply(lambda x: pitcherHand_DF.loc[x])

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


print('PITCHER RECENT FIP')
# RECENT PITCHER FIP
pitcherLink_DF = pd.read_csv('input/2021_pitcher_BR_link_database.csv', sep=','); pitcherLink_DF.set_index('Pitcher',inplace=True)


def getrecwFIP(curPlayer,curStandingsYear):
    recent_FIP = np.nan
    try:
        url = "https://www.baseball-reference.com/players/gl.fcgi?id=" + pitcherLink_DF.loc[curPlayer]['Link'] + "&t=p&year=2021"
        
        r = requests.get(url, headers=BSheaders)
        soup = BeautifulSoup(r.content, "lxml")
        
        statDict = dict()
        def recentStat(statDict,curStat):
            X = [x.text for x in soup.find_all("td", {"data-stat": curStat})[:-1]]
            X = ['0' if x == '' else x for x in X]
            if len(X) < 3:
                statDict[curStat] = np.nan
                if curStat == 'HR': print(curPlayer + ' - NO RECENT FIP')
            else:
                statDict[curStat] = np.sum([float(x) for x in X[-3:]])
            return statDict
        
        for curStat in ['HR','BB','HBP','SO','IP']:
            statDict = recentStat(statDict,curStat)
        statDict['IP'] = (modf(float(statDict['IP']))[0] * 3) + statDict['IP']
        if statDict['IP'] == 0: recent_FIP = np.nan
        else: recent_FIP = round(((13 * statDict['HR']) + (3*(statDict['BB'] + statDict['HBP'])) - (2*statDict['SO'])) /  
                                (statDict['IP'] + 3.214),3)
    except:
        print(curPlayer + ' - NO BASEBALL REFERENCE LINK')
    return recent_FIP

recFIP_dict = dict()
for x in startingPitchers:
    recFIP_dict[x] = getrecwFIP(x,curStandingsYear)
    
statsDF['A_SP_recFIP'] = statsDF['AwaySP'].apply(lambda x: recFIP_dict[x])
statsDF['H_SP_recFIP'] = statsDF['HomeSP'].apply(lambda x: recFIP_dict[x])
statsDF['OP_SP_recFIP'] = statsDF.apply(lambda x: x['A_SP_recFIP'] if x['HomeOrAway'] == 'Home' else x['H_SP_recFIP'], axis=1)
statsDF['TE_SP_recFIP'] = statsDF.apply(lambda x: x['H_SP_recFIP'] if x['HomeOrAway'] == 'Home' else x['A_SP_recFIP'], axis=1)


print('BATTER RECENT WOBA')
## RECENT WOBA FOR BATTER AND THOSE HITTING AROUND HIM IN ORDER
# Get hitters batting before and after current batter in lineup
oneBefore = {1:9,2:1,3:2,4:3,5:4,6:5,7:6,8:7,9:8}
oneAfter = {1:2,2:3,3:4,4:5,5:6,6:7,7:8,8:9,9:1}
statsDF['Batter-1'] = statsDF.apply(lambda x: statsDF[(statsDF['HomeTeam'] == x['HomeTeam']) & (statsDF['HomeOrAway'] == x['HomeOrAway']) & (statsDF['BattingOrder'] == oneBefore[x['BattingOrder']])]['Batter'].values[0], axis=1)
statsDF['Batter+1'] = statsDF.apply(lambda x: statsDF[(statsDF['HomeTeam'] == x['HomeTeam']) & (statsDF['HomeOrAway'] == x['HomeOrAway']) & (statsDF['BattingOrder'] == oneAfter[x['BattingOrder']])]['Batter'].values[0], axis=1)

batterLink_DF = pd.read_csv('input/2021_hitter_BR_link_database.csv', sep=','); batterLink_DF.set_index('Batter',inplace=True)


def getrecwOBA(curPlayer,curStandingsYear):
    recent_wOBA = np.nan
    try:
        url = "https://www.baseball-reference.com/players/gl.fcgi?id=" + batterLink_DF.loc[curPlayer]['Link'] + "&t=b&year=2021"
        
        r = requests.get(url, headers=BSheaders)
        soup = BeautifulSoup(r.content, "lxml")
        
        statDict = dict()
        def recentStat(statDict,curStat):
            X = [x.text for x in soup.find_all("td", {"data-stat": curStat})[:-1]]
            X = ['0' if x == '' else x for x in X]
            if len(X) < 3:
                statDict[curStat] = np.nan
                if curStat == 'H': print(curPlayer + ' - NO RECENT wOBA')
            else:
                statDict[curStat] = np.sum([float(x) for x in X[-3:]])
            return statDict
        
        for curStat in ['H','BB','HBP','2B','3B','HR','IBB','SF','AB']:
            statDict = recentStat(statDict,curStat)
        statDict['1B'] = statDict['H'] - statDict['2B'] - statDict['3B'] - statDict['HR']
        if statDict['AB'] == 0: recent_wOBA = np.nan
        else: recent_wOBA = round((((0.69*statDict['BB']) + (0.719*statDict['HBP']) + (0.87*statDict['1B']) + (1.217*statDict['2B']) + 
                              (1.529*statDict['3B']) + (1.94*statDict['HR'])) / 
                             (statDict['AB'] + statDict['BB'] - statDict['IBB'] + statDict['SF'] + statDict['HBP'])),3)
    except:
        print(curPlayer + ' - NO BASEBALL REFERENCE LINK')
    return recent_wOBA

recwOBA_dict = dict()
for x in startingBatters:
    recwOBA_dict[x] = getrecwOBA(x,curStandingsYear)

statsDF['Batter_recwOBA'] = statsDF['Batter'].apply(lambda x: recwOBA_dict[x])
statsDF['Batter-1_recwOBA'] = statsDF['Batter-1'].apply(lambda x: recwOBA_dict[x])
statsDF['Batter+1_recwOBA'] = statsDF['Batter+1'].apply(lambda x: recwOBA_dict[x])


print('ZIPS PROJECTIONS')
## ZIPS PROJECTIONS
zips_h = pd.read_csv('input/projections/zips/zips_hitters_2021.csv', sep=',')
statsDF = pd.merge(statsDF, zips_h,  how='left', left_on=['Batter'], right_on = ['Player'])

statsDF['OP_BABIP_zips'] = 0.5
statsDF['OP_ERA-_zips'] = 0.5


## BULLPEN STATS
#Define opposing Team for Later Joins
statsDF['OppoTeam'] = statsDF.apply(lambda x: x['AwayTeam'] if x['HomeOrAway'] == 'Home' else x['HomeTeam'], axis=1)

# Load last year's bullpen stats for opposing team
bullpenStats = pd.read_csv('input/bullpen/bullpenByHand_2020.csv', sep=',')
statsDF = pd.merge(statsDF, bullpenStats,  how='left', left_on=['OppoTeam'], right_on = ['Team'])

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

zeroedColumns = ['Batter_recwOBA','Batter-1_recwOBA','Batter+1_recwOBA','BABIP_zips','ISO_zips','OP_BABIP_zips','OP_ERA-_zips','OP_SP_recFIP','TE_SP_recFIP']
for coli in statsDF.columns:
    if coli == labelCol: pass
    elif coli in zeroedColumns:
        statsDF[coli].fillna(0, inplace=True)
    elif ((statsDF[coli].dtypes == 'int64') or (statsDF[coli].dtypes == 'float64')) and coli not in ['year']:
        statsDF[coli].fillna(loaded_fillna_means[coli], inplace=True)

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

featuresDF = pd.get_dummies(featuresDF)
features_list = list(featuresDF.columns)

test_features = np.array(featuresDF)

# Scale Testing Features
test_features = loaded_scaler.transform(test_features)

                    
                    
# Make predictions
predictions = loaded_model.predict(test_features)
predictions = [round(x,2) for x in predictions]