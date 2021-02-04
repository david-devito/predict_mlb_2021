## SCRAPE DATA FROM MLB GAME LOGS - BASEBALL-REFERENCE.COM



import requests
from bs4 import BeautifulSoup, Comment
import re
import sys
import pandas as pd
import numpy as np
import csv
from time import sleep
import pyperclip


url = "https://www.baseball-reference.com/boxes/NYN/NYN202007300.shtml"
headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/39.0.2171.95 Safari/537.36'}

r = requests.get(url, headers=headers)
soup = BeautifulSoup(r.content, "lxml")

## TEAM NAMES
teamNames = [x.text for x in soup.find_all("a", {"itemprop": "name"})]
awayTeam, homeTeam = [teamNames[0], teamNames[1]]


## RUNS SCORED
t_runsScored = [int(x.text) for x in soup.find_all("div", {"class": "score"})]
awayScore, homeScore = [t_runsScored[0], t_runsScored[1]]

## STARTING LINEUPS
def getStarters(team,startersDict):
    starters = soup.find(text=lambda n: isinstance(n, Comment) and 'id="' + startersDict[team] + '"' in n)
    starters = BeautifulSoup(starters, "lxml")
    starters = starters.select('#' + startersDict[team])[0]
    starters = starters.select('a')
    starters = [x.text for x in starters]
    return starters

awayStarters = getStarters("away",{"away":"lineups_1","home":"lineups_2"})
homeStarters = getStarters("home",{"away":"lineups_1","home":"lineups_2"})














