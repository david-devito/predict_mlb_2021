## SCRAPE DATA FROM MLB GAME LOGS - BASEBALL-REFERENCE.COM



import requests
from bs4 import BeautifulSoup, Comment
import re
import sys
import pandas as pd
import numpy as np
import csv
from time import sleep



# WRITE HEADERS TO OUTPUTFILE
with open('/Users/daviddevito/Desktop/predict_mlb_2021/input/gamelogs/output.csv', 'w', newline='') as csvfile:
    statswriter = csv.writer(csvfile, delimiter=',',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
    statswriter.writerow(['Date','AwayTeam','HomeTeam','AwayScore','HomeScore'])

monthsWithGames = ['03','04','05']

for monthi in monthsWithGames:
    for dayi in list(range(0,32)):
        try:
            print(monthi + ' ' + str(dayi))
            url = "https://www.baseball-reference.com/boxes/NYN/NYN2019" + monthi + str(dayi) + "0.shtml"
            headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/39.0.2171.95 Safari/537.36'}
            
            r = requests.get(url, headers=headers)
            soup = BeautifulSoup(r.content, "lxml")
            
            ## DATE
            date = soup.find_all("div", {"class": "scorebox_meta"})[0].select('div')[0].text.replace(',','')
            
            
            ## TEAM NAMES
            teamNames = [x.text for x in soup.find_all("a", {"itemprop": "name"})]
            awayTeam, homeTeam = [teamNames[0], teamNames[1]]
            
            
            ## RUNS SCORED
            t_runsScored = [int(x.text) for x in soup.find_all("div", {"class": "score"})]
            awayScore, homeScore = [t_runsScored[0], t_runsScored[1]]
            
            ## STARTING LINEUPS
            def getStarters(team,num):
                starters = soup.find(text=lambda n: isinstance(n, Comment) and 'id="' + num + '"' in n)
                starters = BeautifulSoup(starters, "lxml")
                starters = [x.text for x in starters.select('#' + num)[0].select('a')]
                return starters
            
            awayStarters = getStarters("away","lineups_1")
            homeStarters = getStarters("home","lineups_2")
            
            ## WRITE TO CSV
            
            with open('/Users/daviddevito/Desktop/predict_mlb_2021/input/gamelogs/output.csv', 'a', newline='') as csvfile:
                statswriter = csv.writer(csvfile, delimiter=',',
                                        quotechar='|', quoting=csv.QUOTE_MINIMAL)
                statswriter.writerow([date,awayTeam,homeTeam,awayScore,homeScore])
        except:pass