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

BSheaders = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/39.0.2171.95 Safari/537.36'}

# Dictionary used for replacing special characters in names
spcharReplace = {'í':'i',
                 'Á':'A',
                 'é':'e',
                 'á':'a'}


year = '2013'
if year == '2020': monthsWithGames = ['06','07','08','09','10']
else: monthsWithGames = ['03','04','05','06','07','08','09','10']
homeTeams = ['ANA','ARI','ATL','BAL','BOS','CHA','CHN','CIN','CLE','COL',
             'DET','HOU','KCA','LAN','MIA','MIL','MIN','NYA','NYN','OAK',
             'PHI','PIT','SDN','SEA','SFN','SLN','TBA','TEX','TOR','WAS']

if year == '2020': weatherTable = "div_3390179539"
else: weatherTable = "div_2016723098"


# WRITE HEADERS TO OUTPUTFILE
outputHeaders = ['Date','AwayTeam','HomeTeam','AwayScore','HomeScore','AwaySP','HomeSP']
outputHeaders.extend(['A_' + str(x) for x in list(range(1,10))])
outputHeaders.extend(['H_' + str(x) for x in list(range(1,10))])
outputHeaders.extend(['A_WinPct','H_WinPct'])
outputHeaders.extend(['temperature','windspeed','precipitation'])
with open('/Users/daviddevito/Desktop/predict_mlb_2021/input/gamelogs/gamelogs' + year + '_orig.csv', 'w', newline='') as csvfile:
    statswriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
    statswriter.writerow(outputHeaders)

for hometeami in ['SEA']:#homeTeams:
    print(hometeami)
    # Loop through each day within every month of regular season
    for monthi in ['04']:#monthsWithGames:
        for dayi in [27]:#list(range(0,32)):
    
            # Try statement will fail if no game exists for that particular day
            #try:
            if dayi < 10: # Add a 0 before day if it's a single digit
                url = "https://www.baseball-reference.com/boxes/" + hometeami + "/" + hometeami + year + monthi + '0' + str(dayi) + "0.shtml"
            else:
                url = "https://www.baseball-reference.com/boxes/" + hometeami + "/" + hometeami + year + monthi + str(dayi) + "0.shtml"
            
            # Scrape url data
            r = requests.get(url, headers=BSheaders)
            soup = BeautifulSoup(r.content, "lxml")
            
            ## DATE
            date = soup.find_all("div", {"class": "scorebox_meta"})[0].select('div')[0].text
            date = date[date.index(' ')+1:]
            # Convert date to useable format
            date = datetime.datetime.strptime(date, '%B %d, %Y').strftime('%d-%m-%Y')
            
            ## TEAM NAMES
            teamNames = [x.text for x in soup.find_all("a", {"itemprop": "name"})]
            awayTeam, homeTeam = [teamNames[0], teamNames[1]]
            
            ## WEATHER
            weather = soup.find(text=lambda n: isinstance(n, Comment) and 'id="' + weatherTable + '"' in n)
            weather = BeautifulSoup(weather, "lxml")
            weather = weather.find_all("div")[-1].text.split(':')[1].split(',')
            temperature = int(''.join(filter(str.isdigit, weather[0])))
            windspeed = int(''.join(filter(str.isdigit, weather[1])))
            precipitation = weather[2]
            precipitation = precipitation.replace(' ','').replace('.','')
            
            ## RUNS SCORED
            t_runsScored = [int(x.text) for x in soup.find_all("div", {"class": "score"})]
            awayScore, homeScore = [t_runsScored[0], t_runsScored[1]]
            
            ## TEAM WIN PERCENTAGE GOING INTO THE GAME
            WP_calc = soup.find_all("div", {"class": "scorebox"})[0]
            AWP = WP_calc.find_all("div")[0].find_all("div")[4].text
            AWP = [int(x) for x in AWP.split('-')]
            # Record includes current game so subtract win or loss depending on result
            if awayScore > homeScore: AWP[0] = AWP[0] - 1
            else: AWP[1] = AWP[1] - 1
            try: AWP = round((AWP[0] / (AWP[0] + AWP[1]))*100,2)
            except: AWP = 0
            HWP = WP_calc.find_all("div")[7].find_all("div")[4].text
            HWP = [int(x) for x in HWP.split('-')]
            # Record includes current game so subtract win or loss depending on result
            if homeScore > awayScore: HWP[0] = HWP[0] - 1
            else: HWP[1] = HWP[1] - 1
            try: HWP = round((HWP[0] / (HWP[0] + HWP[1]))*100,2)
            except: HWP = 0
            
            
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
            
            ## GET STARTING PLAYER STATS FROM PREVIOUS N NUMBER OF GAMES
            numGames = 5
            
            
            starters = soup.find(text=lambda n: isinstance(n, Comment) and 'id="lineups_1"' in n)
            sys.exit()
            starters = BeautifulSoup(starters, "lxml")
            x = starters.select('a')[0]['href'].split('/')[-1].split('.')[0]
            print(x)
            
            url = "https://www.baseball-reference.com/players/gl.fcgi?id=" + str(x) + "&t=b&year=" + year
            r = requests.get(url, headers=BSheaders)
            soup = BeautifulSoup(r.content, "lxml")
            dates = []
            dates.extend([x.text for x in soup.find_all("td", {"data-stat": "date_game"})])
            curDate = datetime.datetime.strptime(date, '%d-%m-%Y').strftime('%b %d')
            
            statDict = dict()
            def recentStat(statDict,curStat):
                X = []
                X.extend([x.text for x in soup.find_all("td", {"data-stat": curStat})])
                statDict[curStat] = np.sum([float(x) for x in X[dates.index(curDate)-(numGames+1):dates.index(curDate)-1]])
            
                return statDict
            
            for curStat in ['H','BB','HBP','2B','3B','HR','IBB','SF','AB']:
                statDict = recentStat(statDict,curStat)
            statDict['1B'] = statDict['H'] - statDict['2B'] - statDict['3B'] - statDict['HR']
            
            recent_wOBA = (((0.69*statDict['BB']) + (0.719*statDict['HBP']) + (0.87*statDict['1B']) + (1.217*statDict['2B']) + (1.529*statDict['3B']) + (1.94*statDict['HR'])) / (statDict['AB'] + statDict['BB'] - statDict['IBB'] + statDict['SF'] + statDict['HBP']))
            recent_wOBA = round(recent_wOBA,3)
            
            print(recent_wOBA)
            
            #recentH = np.sum([float(x) for x in H[dates.index(curDate)-(numGames+1):dates.index(curDate)-1]])
            
            
            
            
            
            
            
            sys.exit()
            
            # Replace Special Characters in Names
            for repi in spcharReplace.keys():
                awayStarters = [x.replace(repi,spcharReplace[repi]) for x in awayStarters]
                homeStarters = [x.replace(repi,spcharReplace[repi]) for x in homeStarters]
                awaySP = awaySP.replace(repi,spcharReplace[repi])
                homeSP = homeSP.replace(repi,spcharReplace[repi])
            
            ## WRITE TO CSV
            dataToWrite = [date,awayTeam,homeTeam,awayScore,homeScore,awaySP,homeSP]
            dataToWrite.extend(awayStarters[0:9])
            dataToWrite.extend(homeStarters[0:9])
            dataToWrite.extend([AWP,HWP])
            dataToWrite.extend([temperature,windspeed,precipitation])
            with open('/Users/daviddevito/Desktop/predict_mlb_2021/input/gamelogs/gamelogs' + year + '_orig.csv', 'a', newline='') as csvfile:
                statswriter = csv.writer(csvfile, delimiter=',',
                                        quotechar='|', quoting=csv.QUOTE_MINIMAL)
                statswriter.writerow(dataToWrite)
            #except:pass