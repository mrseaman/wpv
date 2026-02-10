"""Hourly pageview parsing utilities.

Extracted from course/wiki_pageviews.py - functions needed by the
recommender preprocessing pipeline.
"""

import datetime as dt
import itertools

import pandas as pd


def chr2date(char):
    return ord(char) - 64


def date2chr(date):
    return chr(date + 64)


def parse_hourly_views(compressed_hourly_views, month, year=2019, debug=0):
    result = []
    num_of_days = [0, 31, 28, 31]
    daily_views = compressed_hourly_views.split(',')
    del daily_views[-1]

    if len(daily_views) < num_of_days[month]:
        if chr2date(daily_views[-1][0]) < num_of_days[month]:
            daily_views.append(date2chr(num_of_days[month]) + 'A0')
        for i in range(1, num_of_days[month]):
            if chr2date(daily_views[i - 1][0]) > i:
                daily_views.insert(i - 1, date2chr(i) + 'A0')
    if debug > 2:
        print(daily_views)
    for i in range(len(daily_views)):
        s = daily_views[i]
        day = chr2date(s[0])
        temp = ["".join(x) for _, x in itertools.groupby(s[1:], key=str.isdigit)]
        hours = [ord(x) - 65 for x in temp[::2]]
        hourly_views = [int(x) for x in temp[1::2]]
        if debug > 1:
            print(hours, daily_views[i])
        if len(hours) < 24:
            if hours[-1] < 23:
                hours.append(23)
                hourly_views.append(0)
            for j in range(24):
                if hours[j] > j:
                    hours.insert(j, j)
                    hourly_views.insert(j, 0)
        timestamp = []
        for j in range(len(hours)):
            timestamp.append(dt.datetime(year, month, day, hour=hours[j]))
        result.append(pd.Series(hourly_views, timestamp))
    return pd.concat(result)


def get_all_hourly_views(entry, debug=0):
    hourly_views = []
    for i in range(1, 4):
        hourly_views.append(
            parse_hourly_views(entry['hourly_views_{:02d}'.format(i)], month=i, debug=debug)
        )
    return pd.concat(hourly_views), entry.name
