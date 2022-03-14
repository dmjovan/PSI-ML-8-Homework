import numpy as np
import pandas as pd

import os
from pathlib import Path
import json
from pprint import pprint
import time

from pyparsing import countedArray


##################### GLOBAL VARIABLES ##################

MISSING = np.NaN

venue_stats = {
    "json_name": [],
    "year": [],
    "month": [],
    "day": [],
    "country": [],
    "city": [],
    "band_name": [],
    "is_indie": [],
    "attendance": [],
    "invalid": []
}

fill_attendance_map = {}

unique_json_counter = 0

all_att, all_att_num = 0, 0

##################### IMPLEMENTATION ####################


def parse_year(year: str) -> str:
    if len(year) == 2:
        year = int(year)
        if year >= 60:
            year = "19" + str(year)
        else:
            if year < 10:
                year = "200" + str(year)
            else:
                year = "20" + str(year)
    return str(year)


def parse_month(month: str) -> str:
    return str(int(month))


def parse_day(day: str) -> str:
    return str(int(day))


def parse_country(country: str) -> str:

    # Replace dashs with underscore
    country = country.replace('-','_')

    # Remove prefix
    PREFIX = 'the_'
    if country[:len(PREFIX)] == PREFIX:
        country = country[len(PREFIX):]

    return country#.strip('the_')


def parse_city(city: str) -> str:

    # Replace dashs with underscore
    city = city.replace('-','_')

    return city


def parse_json(json_path: str) -> list:
    global venue_stats, MISSING, unique_json_counter, all_att, all_att_num

    att, num_att = 0, 0
    unique_json_counter += 1

    def parse_event(event: dict) -> list:
        """ Parsing event dict """
        nonlocal att, num_att, json_name, year, month, day, country, city

        # Keys of dict
        event_keys = event.keys()

        if "band_name" in event_keys:

            # Handilng 'is_indie'
            if "is_indie" not in event_keys: # set default values
                event["is_indie"] = False
            
            # Handilng 'attendance'
            if "attendance" not in event_keys:
                event["attendance"] = ('_').join([json_name, country, city, year, month, day])
            else:
                att += event["attendance"]
                num_att += 1

            # Save info
            venue_stats["band_name"].append(event["band_name"])
            venue_stats["is_indie"].append(event["is_indie"])
            venue_stats["attendance"].append(event["attendance"])
            venue_stats["invalid"].append(False)

        else:
            # Invalid event
            venue_stats["band_name"].append(MISSING)
            venue_stats["is_indie"].append(MISSING)
            venue_stats["attendance"].append(MISSING)
            venue_stats["invalid"].append(True)

        venue_stats["json_name"].append(json_name)
        venue_stats["year"].append(year)
        venue_stats["month"].append(month)
        venue_stats["day"].append(day)
        venue_stats["country"].append(country)
        venue_stats["city"].append(city)

    
    # Load json and exctract events
    try:
        with open(json_path, 'r') as f:
            venue = json.load(f)
    except:
        with open(json_path, 'r') as f:
            venue = [json.loads(line.strip('\n')) for line in f.readlines()]

    # Handling only one event without [] in json
    if isinstance(venue, dict):
        venue = [venue]

    # Get json name (for DataFrame)
    json_name = str(json_path.stem).replace('-','_')

    # # Remove prefix
    # PREFIX = 'the_'
    # if json_name[:len(PREFIX)] == PREFIX:
    #     json_name = json_name[len(PREFIX):]

    # Extract info from path
    year, month, day, country, city = pattern_matching(str(json_path))

    if len(venue) == 0:
        # Dict to store venue stats
        venue_stats["json_name"].append(json_name)
        venue_stats["year"].append(year)
        venue_stats["month"].append(month)
        venue_stats["day"].append(day)
        venue_stats["country"].append(country)
        venue_stats["city"].append(city)
        venue_stats["band_name"].append(MISSING)
        venue_stats["is_indie"].append(MISSING)
        venue_stats["attendance"].append(MISSING)
        venue_stats["invalid"].append(True)
    else:
        [parse_event(event) for event in venue]

    if num_att > 0:
        all_att += att
        all_att_num += num_att
        att //= num_att
    else:
        att = MISSING

    # Add mapper
    fill_attendance_map[('_').join([json_name, country, city, year, month, day])] = att


def pattern_matching(rel_path: str):
    global set_path

    # Split relative path
    dirs = rel_path.replace(set_path, '').split(os.path.sep)

    if len(dirs) == 7:
        # {dataset}/{year}/{month}/{day}/{country}/{city}/{venue}.json
        year = parse_year(dirs[1])
        month = parse_month(dirs[2])
        day = parse_day(dirs[3])
        country = parse_country(dirs[4])
        city = parse_city(dirs[5])
    else:
        if any(char.isdigit() for char in dirs[1]):
            # {dataset}/{year_month_day}/{country}/{city}/{venue}.json
            year = parse_year(dirs[1].split('_')[0])
            month = parse_month(dirs[1].split('_')[1])
            day = parse_day(dirs[1].split('_')[2])
            country = parse_country(dirs[2])
            city = parse_city(dirs[3])
        else:
            # {dataset}/{country}/{city}/{year_month_day}/{venue}.json
            year = parse_year(dirs[3].split('_')[0])
            month = parse_month(dirs[3].split('_')[1])
            day = parse_day(dirs[3].split('_')[2])
            country = parse_country(dirs[1])
            city = parse_city(dirs[2])

    return year, month, day, country, city


def main(set_path: str):
    global venue_stats, fill_attendance_map, unique_json_counter, all_att, all_att_num

    # Reset venue_stats
    venue_stats = {
        "json_name": [],
        "year": [],
        "month": [],
        "day": [],
        "country": [],
        "city": [],
        "band_name": [],
        "is_indie": [],
        "attendance": [],
        "invalid": []
    }

    # Reset fill_attendance_map
    fill_attendance_map = {}

    # Reset unique_json_counter
    unique_json_counter = 0

    # Reset all mean
    all_att, all_att_num = 0, 0

    # Parse json
    [parse_json(path) for path in Path(set_path).rglob('*.json')]

    # Calcuate mean of attendance for all venues
    all_att /= all_att_num
    
    # Convert dictionary to DataFrame
    stats_df = pd.DataFrame.from_dict(venue_stats)

    # Extract only valid concerts
    stats_df = stats_df[stats_df.invalid == False]


    # anoint_rat_idxs = stats_df[stats_df.band_name == "anoint_rat"].index.tolist()[:20]
    #print(stats_df[stats_df.index.isin(anoint_rat_idxs)])

    # Fill attendance by mean of attendance by venue mean
    stats_df['attendance'] = stats_df['attendance'].apply(lambda x: x if x not in fill_attendance_map.keys() else fill_attendance_map[x])

    #print(stats_df[stats_df.index.isin(anoint_rat_idxs)])
    # print(sorted(stats_df.band_name.unique().tolist()))
    # print(sorted(stats_df.json_name.unique().tolist()))
    # print(sorted(stats_df.year.unique().tolist()))
    # print(sorted(stats_df.month.unique().tolist()))
    # print(sorted(stats_df.day.unique().tolist()))
    # print(sorted(stats_df.country.unique().tolist()))
    # print(sorted(stats_df.city.unique().tolist()))

    # Fill attendance by mean of attendance by all mean # BUG
    stats_df['attendance'] = stats_df['attendance'].fillna(all_att)
    # stats_df['attendance'] = stats_df['attendance'].fillna(stats_df['attendance'].mean()) 

    #print("\nGlobal average", all_att)
    #print(stats_df[stats_df.index.isin(anoint_rat_idxs)])

    ########################### ANSWER A ##################################
    A_ans = unique_json_counter
    print(A_ans)

    ########################### ANSWER B ##################################
    
    B_ans = stats_df.country.nunique()
    print(B_ans)

    ########################### ANSWER C ##################################
    city_counts = stats_df.city.value_counts()
    city_counts_max = city_counts.max()
    C_ans = sorted(list(filter(lambda x: city_counts[x] == city_counts_max, city_counts.index.tolist())))[0]
    print(C_ans)

    ########################### ANSWER D ##################################
    # stats_df['country_city'] = stats_df['country'] + '-8-' + stats_df['city']
    # D_ans = (',').join(list(map(lambda x: x.split('-8-')[-1], stats_df[stats_df.is_indie == True].groupby('country_city').agg({"attendance": ['sum']}).sort_values(('attendance', 'sum'), ascending=False).index[:3])))
    D_ans = (',').join(stats_df[stats_df.is_indie == True].groupby('city').agg({"attendance": ['sum']}).sort_values(('attendance', 'sum'), ascending=False).index[:3])
    print(D_ans)

    ########################### ANSWER E ##################################

    # Unique venue, year, month, day combination    
    stats_df['concert_info'] = stats_df['json_name'] + '_' + stats_df['year'].astype(str) + '_' + stats_df['month'].astype(str) + '_' + stats_df['day'].astype(str) + '_' + stats_df['country'].astype(str) + '_' + stats_df['city'].astype(str)

    # Extract info about venues with indie performers that night
    concert_info_with_indie = stats_df[stats_df.is_indie == True].concert_info.unique()

    # Most popular bands based on average attendance 
    # print(stats_df[stats_df.concert_info.isin(concert_info_with_indie)].groupby('band_name').agg({"attendance": ['mean']}).sort_values(('attendance', 'mean'), ascending=False))
    E_ans = (',').join(stats_df[stats_df.concert_info.isin(concert_info_with_indie)].groupby('band_name').agg({"attendance": ['mean']}).sort_values(('attendance', 'mean'), ascending=False).index[:3])
    print(E_ans)

    return [A_ans, B_ans, C_ans, D_ans, E_ans]



# example: C:\homework-publish\rockmyworld\public\set\0
if __name__ == "__main__":

    # Get input
    # set_path = input().strip('\n') # BUG
    
    # set_path = "/home/grbic/Desktop/PSI-ML-8-Homework/TaskB/test_dataset"
    # set_path = "/home/grbic/Desktop/PSIML:8/B/public/sets/6"
    # ans = main(set_path)


    # Public test
    for i in range(10):
        print(f"\n\n#################### {i+1} ################")

        # Path to dataset
        set_path = f"/home/grbic/Desktop/PSI-ML-8-Homework/TaskB/dataset/public/sets/{i}"

        # Run script
        s = time.time()
        ans = main(set_path)
        print(f"Elapsed time: {round(time.time()-s,2)}s\n")

        # Read expected output
        with open(f"/home/grbic/Desktop/PSI-ML-8-Homework/TaskB/dataset/public/outputs/case-{i+1}.out") as f:
            out = list(map(lambda x: x.strip('\n'), f.readlines()))

        for a, o, l in zip(ans, out, ['A','B','C','D','E']):
            if str(a) != o:
                print(f"\n------- {l} -------\n")
                print("Output:", a)
                print("Expected:", o)
