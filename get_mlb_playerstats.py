# FUNCTIONS USED TO IMPORT PLAYER STATISTICS

import sys

sys.path.append('./data_scraping')

# Import my functions
from scrape_FG_hitting import batting_stats
from scrape_FG_pitching import pitching_stats
from scrape_FG_fielding import fielding_stats


def load_hitting_data(year):
    # Define columns of stats you'd like to load
    batStatsCols = [5,6,7]
    # Scrape stats from inputted year
    batting = batting_stats(year,statsCols=batStatsCols)
    # Restrict results to batters with over defined number of ABs
    batting = batting[batting['AB'].astype(int) >= 15]
    # Re-index Batting Stats DF by Player Name
    batting = batting.set_index('Name')
    return batting
