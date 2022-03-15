import numpy as np
import pandas as pd

import os
from pathlib import Path
import json

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

    return country


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
                event["attendance"] = ('=').join([json_name, country, city])
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
    else:
        att = MISSING

    # Add mapper
    venue_key = ('=').join([json_name, country, city])

    if venue_key in fill_attendance_map.keys():
        if isinstance(fill_attendance_map[venue_key], list):
            fill_attendance_map[venue_key][0] += att
            fill_attendance_map[venue_key][1] += num_att
        else:
            if num_att > 0:
                fill_attendance_map[venue_key] = [att, num_att]

    else:
        if num_att > 0:
            fill_attendance_map[venue_key] = [att, num_att]
        else:
            fill_attendance_map[venue_key] = MISSING



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

    # Extract only valid
    stats_df = stats_df[stats_df.invalid == False]

    # Fill attendance by mean of attendance by venue mean
    fill_attendance_map = {k: v[0]/v[1] if isinstance(v, list) else v for k, v in fill_attendance_map.items()}

    stats_df['attendance'] = stats_df['attendance'].apply(lambda x: x if x not in fill_attendance_map.keys() else fill_attendance_map[x])

    # Fill attendance by mean of attendance by all mean # BUG
    stats_df['attendance'] = stats_df['attendance'].fillna(all_att)

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
    D_ans = (',').join(stats_df[stats_df.is_indie == True].groupby('city').agg({"attendance": ['sum']}).sort_values(('attendance', 'sum'), ascending=False).index[:3])
    print(D_ans)

    ########################### ANSWER E ##################################

    # Unique venue, year, month, day combination    
    stats_df['concert_info'] = stats_df['json_name'] + '_' + stats_df['year'].astype(str) + '_' + stats_df['month'].astype(str) + '_' + stats_df['day'].astype(str) + '_' + stats_df['country'].astype(str) + '_' + stats_df['city'].astype(str)

    # Extract info about venues with indie performers that night
    concert_info_with_indie = stats_df[stats_df.is_indie == True].concert_info.unique()

    # Most popular bands based on average attendance 
    E_ans = (',').join(stats_df[stats_df.concert_info.isin(concert_info_with_indie)].groupby('band_name').agg({"attendance": ['mean']}).sort_values(('attendance', 'mean'), ascending=False).index[:3])
    print(E_ans)

    return [A_ans, B_ans, C_ans, D_ans, E_ans]


if __name__ == "__main__":

    # Get input
    set_path = input().strip('\n') # BUG

    ans = main(set_path)