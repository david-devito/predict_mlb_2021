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
                 'é':'e'}



# WRITE HEADERS TO OUTPUTFILE
outputHeaders = ['Date','AwayTeam','HomeTeam','AwayScore','HomeScore','AwaySP','HomeSP']
outputHeaders.extend(['A_' + str(x) for x in list(range(1,10))])
outputHeaders.extend(['H_' + str(x) for x in list(range(1,10))])
with open('/Users/daviddevito/Desktop/predict_mlb_2021/input/gamelogs/output.csv', 'w', newline='') as csvfile:
    statswriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
    statswriter.writerow(outputHeaders)

year = '2019'
monthsWithGames = ['03','04','05','06','07','08','09','10']
hometeami = 'BAL'
# Loop through each day within every month of regular season
for monthi in monthsWithGames:
    for dayi in list(range(0,32)):

        # Try statement will fail if no game exists for that particular day
        try:
            if dayi == 0: print('Starting Month: ' + monthi)
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
            
            ## RUNS SCORED
            t_runsScored = [int(x.text) for x in soup.find_all("div", {"class": "score"})]
            awayScore, homeScore = [t_runsScored[0], t_runsScored[1]]
            
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
            awaySP = getStarters("div_" + awayTeam.replace(' ','') + "pitching")[0]
            homeSP = getStarters("div_" + homeTeam.replace(' ','') + "pitching")[0]
            
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
            with open('/Users/daviddevito/Desktop/predict_mlb_2021/input/gamelogs/output.csv', 'a', newline='') as csvfile:
                statswriter = csv.writer(csvfile, delimiter=',',
                                        quotechar='|', quoting=csv.QUOTE_MINIMAL)
                statswriter.writerow(dataToWrite)
        except:pass