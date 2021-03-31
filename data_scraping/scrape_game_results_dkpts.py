## SCRAPE DATA FROM MLB GAME LOGS - BASEBALL-REFERENCE.COM



import requests
from bs4 import BeautifulSoup, Comment
import re
import sys
import pandas as pd
import numpy as np
import csv
from time import sleep
import datetime
from math import modf

BSheaders = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/39.0.2171.95 Safari/537.36'}

# Dictionary used for replacing special characters in names
spcharReplace = {'í':'i',
                 'Á':'A',
                 'é':'e',
                 'á':'a'}

section = 'hitters_homers' #hitters, hitters_hand, hitters_homers, pitchers, pitchers_hand

year = '2017'
if year == '2020': monthsWithGames = ['06','07','08','09','10']
else: monthsWithGames = ['03','04','05','06','07','08','09','10']
homeTeams = ['ANA','ARI','ATL','BAL','BOS','CHA','CHN','CIN','CLE','COL',
             'DET','HOU','KCA','LAN','MIA','MIL','MIN','NYA','NYN','OAK',
             'PHI','PIT','SDN','SEA','SFN','SLN','TBA','TEX','TOR','WAS']

# WRITE HEADERS TO OUTPUTFILE
outputHeaders = ['Date','AwayTeam','HomeTeam']
if section == 'hitters':
    outputHeaders.extend(['Batter','DKPts','BattingOrder','HomeOrAway'])
elif section == 'hitters_hand':
    outputHeaders.extend(['Batter','BatterHand'])
elif section == 'hitters_homers':
    outputHeaders.extend(['Batter','HomeRun','BattingOrder','HomeOrAway'])
elif section == 'pitchers':pass
elif section == 'pitchers_hand':
    outputHeaders.extend(['A_SP_Hand','H_SP_Hand'])
with open('/Users/daviddevito/Desktop/predict_mlb_2021/input/gamelogs/gamelogs' + year + '_' + section + '_dkpts.csv', 'w', newline='') as csvfile:
    statswriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
    statswriter.writerow(outputHeaders)

for hometeami in homeTeams:
    print(hometeami)
    # Loop through each day within every month of regular season
    for monthi in monthsWithGames:
        print(monthi)
        for dayi in list(range(0,32)):
            # Try statement will fail if no game exists for that particular day
            try:
                # Add a 0 before day if it's a single digit
                if dayi < 10: url = "https://www.baseball-reference.com/boxes/" + hometeami + "/" + hometeami + year + monthi + '0' + str(dayi) + "0.shtml"
                else: url = "https://www.baseball-reference.com/boxes/" + hometeami + "/" + hometeami + year + monthi + str(dayi) + "0.shtml"
                
                # Scrape url data
                r = requests.get(url, headers=BSheaders)
                soup = BeautifulSoup(r.content, "lxml")
                
                ## DATE OF GAME
                date = soup.find_all("div", {"class": "scorebox_meta"})[0].select('div')[0].text
                date = date[date.index(' ')+1:]
                # Convert date to useable format
                date = datetime.datetime.strptime(date, '%B %d, %Y').strftime('%d-%m-%Y')
                
                ## TEAM NAMES
                teamNames = [x.text for x in soup.find_all("a", {"itemprop": "name"})]
                awayTeam, homeTeam = [teamNames[0], teamNames[1]]
                
                
                if (section == 'hitters') | (section == 'hitters_hand') | (section == 'hitters_homers') | (section == 'pitchers'):
                    ## STARTING LINEUPS
                    def getStarters(num):
                        starters = soup.find(text=lambda n: isinstance(n, Comment) and 'id="' + num + '"' in n)
                        starters = BeautifulSoup(starters, "lxml")
                        starters = [x.text for x in starters.select('#' + num)[0].select('a')]
                        return starters
                    
                    # Starting Hitters
                    awayStarters = getStarters("lineups_1")
                    homeStarters = getStarters("lineups_2")
                    # Starting Pitchers
                    awaySP = getStarters("div_" + awayTeam.replace(' ','').replace('.','') + "pitching")[0]
                    homeSP = getStarters("div_" + homeTeam.replace(' ','').replace('.','') + "pitching")[0]
                
                    # Replace Special Characters in Names
                    for repi in spcharReplace.keys():
                        awayStarters = [x.replace(repi,spcharReplace[repi]) for x in awayStarters]
                        homeStarters = [x.replace(repi,spcharReplace[repi]) for x in homeStarters]
                        awaySP = awaySP.replace(repi,spcharReplace[repi])
                        homeSP = homeSP.replace(repi,spcharReplace[repi])
                
                if (section == 'hitters') | (section == 'hitters_hand') | (section == 'hitters_homers'):
                    ## GET STARTING PLAYER STATS FROM PREVIOUS N NUMBER OF GAMES
                    numGames = 3
                    awayStarterLinks = BeautifulSoup(soup.find(text=lambda n: isinstance(n, Comment) and 'id="lineups_1"' in n),"lxml").select('#lineups_1')[0].select('a')
                    awayStarterLinks = ['.'.join(x['href'].split('/')[-1].split('.')[:-1]) for x in awayStarterLinks]
                    homeStarterLinks = BeautifulSoup(soup.find(text=lambda n: isinstance(n, Comment) and 'id="lineups_2"' in n),"lxml").select('#lineups_2')[0].select('a')
                    homeStarterLinks = ['.'.join(x['href'].split('/')[-1].split('.')[:-1]) for x in homeStarterLinks]
                    
                    if section == 'hitters':
                        hitter_dkpts = []
                        for curPlayer in awayStarterLinks[0:9] + homeStarterLinks[0:9]:
                            url = "https://www.baseball-reference.com/players/gl.fcgi?id=" + str(curPlayer) + "&t=b&year=" + year
                            r = requests.get(url, headers=BSheaders)
                            soup = BeautifulSoup(r.content, "lxml")
                            dates = [x.text for x in soup.find_all("td", {"data-stat": "date_game"})]
                            curDate = datetime.datetime.strptime(date, '%d-%m-%Y').strftime('%b %d').lstrip("0").replace(" 0", " ")
                            
                            statDict = dict()
                            def recentStat(statDict,curStat):
                                X = [x.text for x in soup.find_all("td", {"data-stat": curStat})]
                                X = ['0' if x == '' else x for x in X]
                                statDict[curStat] = float(X[dates.index(curDate)])
                                return statDict
                            
                            for curStat in ['H','2B','3B','HR','RBI','R','IBB','BB','HBP','SB','AB']: 
                                statDict = recentStat(statDict,curStat)
                            statDict['1B'] = statDict['H'] - statDict['2B'] - statDict['3B'] - statDict['HR']
                            hitter_dkpts.append((statDict['1B']*3) + (statDict['2B']*5) + (statDict['3B']*8) + (statDict['HR']*10) + (statDict['RBI']*2) + (statDict['R']*2) + (statDict['IBB']*2) + (statDict['BB']*2) + (statDict['HBP']*2) + (statDict['SB']*5))
                    elif section == 'hitters_hand':
                        hitter_hand = []
                        for curPlayer in awayStarterLinks[0:9] + homeStarterLinks[0:9]:
                            try:
                                url = "https://www.baseball-reference.com/players/gl.fcgi?id=" + str(curPlayer) + "&t=b&year=" + year
                                r = requests.get(url, headers=BSheaders)
                                soup = BeautifulSoup(r.content, "lxml")
                                curHitterHand = soup.find_all("div", {"itemtype": "https://schema.org/Person"})[0].find_all("p")[1].text.split(':')[1][1]
                            except:
                                curHitterHand = np.nan
                            hitter_hand.append(curHitterHand)
                    elif section == 'hitters_homers':
                        hitter_homers = []
                        for curPlayer in awayStarterLinks[0:9] + homeStarterLinks[0:9]:
                            url = "https://www.baseball-reference.com/players/gl.fcgi?id=" + str(curPlayer) + "&t=b&year=" + year
                            r = requests.get(url, headers=BSheaders)
                            soup = BeautifulSoup(r.content, "lxml")
                            dates = [x.text for x in soup.find_all("td", {"data-stat": "date_game"})]
                            curDate = datetime.datetime.strptime(date, '%d-%m-%Y').strftime('%b %d').lstrip("0").replace(" 0", " ")
                            
                            statDict = dict()
                            def recentStat(statDict,curStat):
                                X = [x.text for x in soup.find_all("td", {"data-stat": curStat})]
                                X = ['0' if x == '' else x for x in X]
                                statDict[curStat] = float(X[dates.index(curDate)])
                                return statDict
                            
                            for curStat in ['HR','AB']: 
                                statDict = recentStat(statDict,curStat)
                            hitter_homers.append(statDict['HR'])
    
                            
                if (section == 'pitchers') | (section == 'pitchers_hand'):
                    ## GET STARTING PITCHER STATS FROM PREVIOUS N NUMBER OF GAMES
                    numGames = 3
                    awaySP_link = soup.find(text=lambda n: isinstance(n, Comment) and 'id="div_' + awayTeam.replace(' ','').replace('.','') + 'pitching"' in n)
                    awaySP_link = BeautifulSoup(awaySP_link, "lxml")
                    awaySP_link = awaySP_link.select('#div_' + awayTeam.replace(' ','').replace('.','') + 'pitching')[0].select('a')[0]['href'].split('/')[3].split('.')[0]
                    homeSP_link = soup.find(text=lambda n: isinstance(n, Comment) and 'id="div_' + homeTeam.replace(' ','').replace('.','') + 'pitching"' in n)
                    homeSP_link = BeautifulSoup(homeSP_link, "lxml")
                    homeSP_link = homeSP_link.select('#div_' + homeTeam.replace(' ','').replace('.','') + 'pitching')[0].select('a')[0]['href'].split('/')[3].split('.')[0]
                    
                    if section == 'pitchers':pass
                    elif section == 'pitchers_hand':
                        pitcher_hand = []
                        for curPlayer in [awaySP_link,homeSP_link]:
                            try:
                                url = "https://www.baseball-reference.com/players/gl.fcgi?id=" + str(curPlayer) + "&t=b&year=" + year
                                r = requests.get(url, headers=BSheaders)
                                soup = BeautifulSoup(r.content, "lxml")
                                curPitcherHand = soup.find_all("div", {"itemtype": "https://schema.org/Person"})[0].find_all("p")[1].text.split(':')[2][1]
                            except:
                                curPitcherHand = np.nan
                            pitcher_hand.append(curPitcherHand)
                
                
                ## WRITE TO CSV
                homeOrAway = ['Away']*9 + ['Home']*9
                battingOrder = list(range(1,10))*2
                if (section == 'hitters') | (section == 'hitters_hand') | (section == 'hitters_homers'):
                    for ix, curPlayer in enumerate(awayStarters[0:9] + homeStarters[0:9]):
                        if section == 'hitters':
                            dataToWrite = [date,awayTeam,homeTeam,curPlayer,hitter_dkpts[ix],battingOrder[ix],homeOrAway[ix]]
                        elif section == 'hitters_hand':
                            dataToWrite = [date,awayTeam,homeTeam,curPlayer,hitter_hand[ix]]
                        elif section == 'hitters_homers':
                            dataToWrite = [date,awayTeam,homeTeam,curPlayer,hitter_homers[ix],battingOrder[ix],homeOrAway[ix]]
                        
                        with open('/Users/daviddevito/Desktop/predict_mlb_2021/input/gamelogs/gamelogs' + year + '_' + section + '_dkpts.csv', 'a', newline='') as csvfile:
                            statswriter = csv.writer(csvfile, delimiter=',',
                                                     quotechar='|', quoting=csv.QUOTE_MINIMAL)
                            statswriter.writerow(dataToWrite)
                elif (section == 'pitchers') | (section == 'pitchers_hand'):
                    if section == 'pitchers_hand':
                        dataToWrite = [date,awayTeam,homeTeam,pitcher_hand[0],pitcher_hand[1]]
                    
                    with open('/Users/daviddevito/Desktop/predict_mlb_2021/input/gamelogs/gamelogs' + year + '_' + section + '_dkpts.csv', 'a', newline='') as csvfile:
                        statswriter = csv.writer(csvfile, delimiter=',',
                                                 quotechar='|', quoting=csv.QUOTE_MINIMAL)
                        statswriter.writerow(dataToWrite)

            except:pass