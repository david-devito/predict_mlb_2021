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

section = 'recwOBA' #lineups, recwOBA, winpct, weather, recFIP

year = '2015'
if year == '2020': monthsWithGames = ['06','07','08','09','10']
else: monthsWithGames = ['03','04','05','06','07','08','09','10']
homeTeams = ['ANA','ARI','ATL','BAL','BOS','CHA','CHN','CIN','CLE','COL',
             'DET','HOU','KCA','LAN','MIA','MIL','MIN','NYA','NYN','OAK',
             'PHI','PIT','SDN','SEA','SFN','SLN','TBA','TEX','TOR','WAS']
homeTeams = ['OAK',
             'PHI','PIT','SDN','SEA','SFN','SLN','TBA','TEX','TOR','WAS']
if year == '2020': weatherTable = "div_3390179539"
else: weatherTable = "div_2016723098"


# WRITE HEADERS TO OUTPUTFILE
outputHeaders = ['Date','AwayTeam','HomeTeam']
if section == 'lineups':
    outputHeaders.extend(['AwayScore','HomeScore','AwaySP','HomeSP'])
    outputHeaders.extend(['A_' + str(x) for x in list(range(1,10))])
    outputHeaders.extend(['H_' + str(x) for x in list(range(1,10))])
elif section == 'recwOBA':
    outputHeaders.extend(['A_' + str(x) + '_recwOBA' for x in list(range(1,10))])
    outputHeaders.extend(['H_' + str(x) + '_recwOBA' for x in list(range(1,10))])
elif section == 'recFIP':
    outputHeaders.extend(['A_SP_recFIP','H_SP_recFIP'])
elif section == 'winpct':
    outputHeaders.extend(['A_SeaWinPct','A_last3WinPct','A_last5WinPct','A_last10WinPct','H_SeaWinPct','H_last3WinPct','H_last5WinPct','H_last10WinPct',])
elif section == 'weather':
    outputHeaders.extend(['temperature','windSpeed','windDirection','precipitation'])
#with open('/Users/daviddevito/Desktop/predict_mlb_2021/input/gamelogs/gamelogs' + year + '_' + section + '.csv', 'w', newline='') as csvfile:
#    statswriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
#    statswriter.writerow(outputHeaders)

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
                
                if section == 'weather':
                    ## WEATHER
                    weather = soup.find(text=lambda n: isinstance(n, Comment) and 'id="' + weatherTable + '"' in n)
                    weather = BeautifulSoup(weather, "lxml")
                    weather = weather.find_all("div")[-1].text.split(':')[1].split(',')
                    temperature = int(''.join(filter(str.isdigit, weather[0])))
                    windspeed = int(''.join(filter(str.isdigit, weather[1])))
                    if windspeed == 0: windDirection = 'NoWind'
                    else: windDirection = ' '.join(weather[1].split(' ')[3:])
                    if len(weather) < 3: precipitation = ''
                    else: precipitation = weather[-1].replace(' ','').replace('.','')
                
                ## RUNS SCORED BY EACH TEAM
                t_runsScored = [int(x.text) for x in soup.find_all("div", {"class": "score"})]
                awayScore, homeScore = [t_runsScored[0], t_runsScored[1]]
                
                if section == 'winpct':
                    ## TEAM WIN PERCENTAGE GOING INTO THE GAME
                    def getWinPct(soup,curTeamScore,oppTeamScore,divNum):
                        WP_calc = soup.find_all("div", {"class": "scorebox"})[0]
                        WP = WP_calc.find_all("div")[divNum].find_all("div")[4].text
                        WP = [int(x) for x in WP.split('-')]
                        # Record includes current game so subtract win or loss depending on result
                        if curTeamScore > oppTeamScore: WP[0] = WP[0] - 1
                        else: WP[1] = WP[1] - 1
                        try: WP = round((WP[0] / (WP[0] + WP[1]))*100,2)
                        except: WP = 0
                        return WP
                    
                    AWP = getWinPct(soup,awayScore,homeScore,0)
                    HWP = getWinPct(soup,homeScore,awayScore,7)
                    
                    awayTeamAbb = soup.find_all("div", {"itemprop": "performer"})[0].select('a')[2]['href'].split('/')[2]
                    homeTeamAbb = soup.find_all("div", {"itemprop": "performer"})[1].select('a')[2]['href'].split('/')[2]
                    
                    
                    def recentTeamRecord(team,year,recentGames):
                        url = "https://www.baseball-reference.com/teams/" + team + "/" + year + "-schedule-scores.shtml"
                        r = requests.get(url, headers=BSheaders)
                        soup = BeautifulSoup(r.content, "lxml")
                        dates = [x.text for x in soup.find_all("td", {"data-stat": "date_game"})]
                        curDate = datetime.datetime.strptime(date, '%d-%m-%Y').strftime('%A, %b %d').lstrip("0").replace(" 0", " ")
                        teamWinLoss = [x.text[0] for x in soup.find_all("td", {"data-stat": "win_loss_result"})]
                        record = teamWinLoss[dates.index(curDate)-recentGames+1:dates.index(curDate)-1]
                        if not record: record = np.nan
                        else: record = round(np.sum([1 if x == 'W' else 0 for x in record])/recentGames,2)
                        return record
                        
                    A_last3Record = recentTeamRecord(awayTeamAbb,year,3)
                    A_last5Record = recentTeamRecord(awayTeamAbb,year,5)
                    A_last10Record = recentTeamRecord(awayTeamAbb,year,10)
                    H_last3Record = recentTeamRecord(homeTeamAbb,year,3)
                    H_last5Record = recentTeamRecord(homeTeamAbb,year,5)
                    H_last10Record = recentTeamRecord(homeTeamAbb,year,10)
                
                
                if (section == 'lineups') | (section == 'recFIP'):
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
                
                if section == 'recwOBA':
                    ## GET STARTING PLAYER STATS FROM PREVIOUS N NUMBER OF GAMES
                    numGames = 3
                    awayStarterLinks = BeautifulSoup(soup.find(text=lambda n: isinstance(n, Comment) and 'id="lineups_1"' in n),"lxml").select('#lineups_1')[0].select('a')
                    awayStarterLinks = [x['href'].split('/')[-1].split('.')[0] for x in awayStarterLinks]
                    homeStarterLinks = BeautifulSoup(soup.find(text=lambda n: isinstance(n, Comment) and 'id="lineups_2"' in n),"lxml").select('#lineups_2')[0].select('a')
                    homeStarterLinks = [x['href'].split('/')[-1].split('.')[0] for x in homeStarterLinks]
                    
                    recent_wOBA = []
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
                            if dates.index(curDate)-(numGames+1) < 0:
                                statDict[curStat] = np.nan
                            else:
                                statDict[curStat] = np.sum([float(x) for x in X[dates.index(curDate)-(numGames):dates.index(curDate)]])
                            return statDict
                        
                        for curStat in ['H','BB','HBP','2B','3B','HR','IBB','SF','AB']:
                            statDict = recentStat(statDict,curStat)
                        statDict['1B'] = statDict['H'] - statDict['2B'] - statDict['3B'] - statDict['HR']
                        if statDict['AB'] == 0: recent_wOBA.append(np.nan)
                        else: recent_wOBA.append(round((((0.69*statDict['BB']) + (0.719*statDict['HBP']) + (0.87*statDict['1B']) + (1.217*statDict['2B']) + 
                                              (1.529*statDict['3B']) + (1.94*statDict['HR'])) / 
                                             (statDict['AB'] + statDict['BB'] - statDict['IBB'] + statDict['SF'] + statDict['HBP'])),3))
                
                if section == 'recFIP':
                    ## GET STARTING PITCHER STATS FROM PREVIOUS N NUMBER OF GAMES
                    numGames = 3
                    awaySP_link = soup.find(text=lambda n: isinstance(n, Comment) and 'id="div_' + awayTeam.replace(' ','').replace('.','') + 'pitching"' in n)
                    awaySP_link = BeautifulSoup(awaySP_link, "lxml")
                    awaySP_link = awaySP_link.select('#div_' + awayTeam.replace(' ','').replace('.','') + 'pitching')[0].select('a')[0]['href'].split('/')[3].split('.')[0]
                    homeSP_link = soup.find(text=lambda n: isinstance(n, Comment) and 'id="div_' + homeTeam.replace(' ','').replace('.','') + 'pitching"' in n)
                    homeSP_link = BeautifulSoup(homeSP_link, "lxml")
                    homeSP_link = homeSP_link.select('#div_' + homeTeam.replace(' ','').replace('.','') + 'pitching')[0].select('a')[0]['href'].split('/')[3].split('.')[0]
                    
                    recent_FIP = []
                    for curPlayer in [awaySP_link,homeSP_link]:
                        url = "https://www.baseball-reference.com/players/gl.fcgi?id=" + str(curPlayer) + "&t=p&year=" + year
                        r = requests.get(url, headers=BSheaders)
                        soup = BeautifulSoup(r.content, "lxml")
                        dates = [re.sub('\xa0', ' ', x.text).split('(')[0] for x in soup.find_all("td", {"data-stat": "date_game"})]
                        curDate = datetime.datetime.strptime(date, '%d-%m-%Y').strftime('%b %d').lstrip("0").replace(" 0", " ")
                        
                        statDict = dict()
                        def recentStat(statDict,curStat):
                            X = [x.text for x in soup.find_all("td", {"data-stat": curStat})]
                            X = ['0' if x == '' else x for x in X]
                            if dates.index(curDate)-(numGames+1) < 0:
                                statDict[curStat] = np.nan
                            else:
                                statDict[curStat] = np.sum([float(x) for x in X[dates.index(curDate)-(numGames):dates.index(curDate)]])
                            return statDict
                        
                        for curStat in ['HR','BB','HBP','SO','IP']:
                            statDict = recentStat(statDict,curStat)
                        statDict['IP'] = (modf(float(statDict['IP']))[0] * 3) + statDict['IP']
                        if statDict['IP'] == 0: recent_FIP.append(np.nan)
                        else: recent_FIP.append(((13 * statDict['HR']) + (3*(statDict['BB'] + statDict['HBP'])) - (2*statDict['SO'])) /  
                                                (statDict['IP'] + 3.214))
                
                
                ## WRITE TO CSV
                dataToWrite = [date,awayTeam,homeTeam]
                if section == 'lineups':
                    dataToWrite.extend([awayScore,homeScore,awaySP,homeSP])
                    dataToWrite.extend(awayStarters[0:9])
                    dataToWrite.extend(homeStarters[0:9])
                elif section == 'recwOBA':
                    dataToWrite.extend(recent_wOBA)
                elif section == 'recFIP':
                    dataToWrite.extend(recent_FIP)
                elif section == 'winpct':
                    dataToWrite.extend([AWP,A_last3Record,A_last5Record,A_last10Record,HWP,H_last3Record,H_last5Record,H_last10Record])
                elif section == 'weather':
                    dataToWrite.extend([temperature,windspeed,windDirection,precipitation])
                
                with open('/Users/daviddevito/Desktop/predict_mlb_2021/input/gamelogs/gamelogs' + year + '_' + section + '.csv', 'a', newline='') as csvfile:
                    statswriter = csv.writer(csvfile, delimiter=',',
                                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
                    statswriter.writerow(dataToWrite)
            except:pass