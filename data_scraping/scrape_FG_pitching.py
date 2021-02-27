# Scrapes fangraphs for pitching statistics from given year


import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
import requests
import sys

def get_soup(start_season, end_season, league, qual, ind, adjhandVar, statsCols):
    statsCols = ",".join(str(e) for e in statsCols)
    statsCols = 'c,' + statsCols
    
    url =  'http://www.fangraphs.com/leaders.aspx?pos=all&stats=pit&lg={}&qual={}&type=' + statsCols + '&season={}&month={}&season1={}&ind={}&team=&rost=&age=&filter=&players=&page=1_10000'
    url = url.format(league, qual, end_season, adjhandVar, start_season, ind)
    s=requests.get(url).content
    return BeautifulSoup(s, "lxml")

def get_table(soup, ind):
    table = soup.find('table', {'class': 'rgMasterTable'})
    data = []
    # pull heading names from the fangraphs tables
    headings = [row.text.strip() for row in table.find_all('th')[1:]]
    print(headings)
    #sys.exit()
    
    table_body = table.find('tbody')
    rows = table_body.find_all('tr')
    for row in rows:
        cols = row.find_all('td')
        cols = [ele.text.strip() for ele in cols]
        data.append([ele for ele in cols[1:]])

    data = pd.DataFrame(data=data, columns=headings)

    #replace empty strings with NaN
    data.replace(r'^\s*$', np.nan, regex=True, inplace = True)

    # convert percentage datatype to floats
    # get list of data columns that output as string with % symbol
    percentages = []
    for coli in data.columns:
        # Search through each row iteratively in case the first couple row values are NaN
        for rowi in range(0,len(data)):
            try:
                if '%' in data[coli][rowi]:
                    percentages.append(coli)
                break
            except: pass
    for col in percentages:
        # skip if column is all NA (happens for some of the more obscure stats + in older seasons)
        if not data[col].empty:
            if pd.api.types.is_string_dtype(data[col]):
                data[col] = data[col].str.strip(' %')
                data[col] = data[col].str.strip('%')
                data[col] = data[col].astype(float)/100.
        else:
            pass
        
    #convert everything except name and team to numeric
    cols_to_numeric = [col for col in data.columns if col not in ['Name', 'Team']]
    data[cols_to_numeric] = data[cols_to_numeric].astype(float)
    return data

def pitching_stats(start_season, end_season=None, league='all', qual=1, ind=1, handVar='B', statsCols=list(range(1,50))):
    if start_season is None:
        raise ValueError("You need to provide the season to collect data for.")
    if end_season is None:
        end_season = start_season
    # Adjust handVar variable based on month number used by Fantrax
    handVarDict = {'B':'0','L':'13','R':'14'}
    adjhandVar = handVarDict[handVar]
    
    soup = get_soup(start_season=start_season, end_season=end_season, league=league, qual=qual, ind=ind, adjhandVar=adjhandVar, statsCols=statsCols)
    table = get_table(soup, ind)
    
    return table
