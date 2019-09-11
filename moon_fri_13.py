import unittest
import datetime
from skyfield import api
from skyfield.api import earth, moon, JulianDate
from skyfield import almanac
import matplotlib.pyplot as plt
import pickle
from scipy.signal import argrelextrema
import numpy as np
from pytz import timezone
from pprint import pprint
from tqdm import tqdm
from tqdm import tqdm

ts = api.load.timescale()
now = ts.now().utc_datetime().replace(microsecond = 0)
e = api.load('de430t.bsp')

UTC = timezone('UTC')
eastern = timezone('US/Eastern')
central = timezone('US/Central')
mountain = timezone('US/Mountain')
pacific = timezone('US/Pacific')
alaska = timezone('US/Alaska')
hawaii = timezone('US/Hawaii')
dom = timezone('America/Dominica')




def apsis(year = 2019, body='moon'):
    '''
    find the points of in the orbit of the given planet (Moon by default) that are nearest and furthers
    :param year: year to calculate over
    :param body: the body to measure distance from the Earth against.  Must be in the ephemris preloaded ephemeris
    :return: dictionaries of distances in km (body center to body center) of apogees and perigees indexed by datetime (UTC)
    '''
    picklefile = 'data/apsis.p'
    apogees = None
    perigees = None
    try:
        data = pickle.load(open(picklefile, "rb"))
    except:
        data = {} # boostrapin
    if body in data.keys():
        if year in data[body]:
            apogees = data[body][year]['apogees']
            perigees = data[body][year]['perigees']
    else:
        data[body]= dict()

    if apogees is None or perigees is None:
        print(f"calculating apsis for {year}")
        apogees = dict()
        perigees = dict()
        planets = e
        earth, moon = planets['earth'], planets[body]

        t = ts.utc(year, 1, range(1,367))  # start with introduction of Gregorian calendar
        dt = t.utc_datetime()#_and_leap_second()

        astrometric = earth.at(t).observe(moon)
        _, _, distance = astrometric.radec()

        #find each perigee, at day precision
        localmaxes = argrelextrema(distance.km, np.less)[0]
        for i in localmaxes:
            # get minute precision
            t2 = ts.utc(dt[i].year, dt[i].month, dt[i].day-1, 0, range(2881))
            dt2 = t2.utc_datetime()  # _and_leap_second()
            astrometric2 = earth.at(t2).observe(moon)
            _, _, distance2 = astrometric2.radec()
            m = min(distance2.km)
            daindex = list(distance2.km).index(m)
            perigees[dt2[daindex]] = m

        #find each apogee, at day precision
        localmins = argrelextrema(distance.km, np.greater)[0]
        for i in localmins:
            # get minute precision
            t2 = ts.utc(dt[i].year, dt[i].month, dt[i].day-1, 0, range(2881))
            dt2 = t2.utc_datetime()  # _and_leap_second()
            astrometric2 = earth.at(t2).observe(moon)
            _, _, distance2 = astrometric2.radec()
            m = max(distance2.km)
            daindex = list(distance2.km).index(m)
            apogees[dt2[daindex]] = m

        data[body][year] = {'apogees': apogees, 'perigees': perigees}
        pickle.dump(data, open(picklefile, "wb"))

    return apogees, perigees

def main(startyear = 1582, years = 1000, month=10, day=15):
    '''
    Generate tables of number of, next and previous occurances of :
       Full moon falling on Friday the 13th
       Mini moon falling on Friday the 13th
       Super moon falling on Friday the 13th
    Across American timezones and anywhere in the world
    :param startyear: year to start
    :param years: number of years to consider
    :param month: month to start (defaults to first month of Gregorian calendar)
    :param day: day to start (defaults to first day of Gregorian calendar)
    :return: nothing
    '''
    endyear = startyear + years
    picklefile = f"data/fullmoons_{startyear}-{endyear}.p"
    try:
        # try cache first
        (t,y) = pickle.load(open(picklefile, "rb"))
    except:
        # first day =  Friday, 15 October 1582
        t0 = ts.utc(startyear, month, day)  # start with introduction of Gregorian calendar
        t1 = ts.utc(endyear, month, day)
        t, y = almanac.find_discrete(t0, t1, almanac.moon_phases(e))
        pickle.dump((t,y), open(picklefile, "wb"))

    picklefile2 = f"data/coincidences_{startyear}-{endyear}.p"
    try:
        (fri13, fri13apogee, fri13perigee, total) = pickle.load(open(picklefile2, "rb"))
    except:
        fri13, fri13apogee, fri13perigee, total = find_coincidences(t, y)
        pickle.dump((fri13, fri13apogee, fri13perigee, total), open(picklefile2, "wb"))

    dump_html(day, fri13, fri13apogee, fri13perigee, month, startyear, total, years)

def find_coincidences(t, y):
    '''
    Finds full moons, mini moons, super mons that coincide with Friday the 13th
    :param t: numpy array of timescales
    :param y: numpy array of  Skyfield moon phases (2 is full, the one we are looking for here)
    :return: dictionary of timezones of lists of friday 13 full, mini and super moons
    '''
    fri13 = {'UTC': [], 'Eastern': [], 'Central': [], 'Mountain': [], 'Pacific': [], 'Alaska': [], 'Hawaii': [], 'Dominican': [], }
    fri13perigee = {'UTC': [], 'Eastern': [], 'Central': [], 'Mountain': [], 'Pacific': [], 'Alaska': [], 'Hawaii': [], 'Dominican': [], }
    fri13apogee = {'UTC': [], 'Eastern': [], 'Central': [], 'Mountain': [], 'Pacific': [], 'Alaska': [], 'Hawaii': [], 'Dominican': [], }
    lastdt = None
    total = 0
    for x, a in tqdm(zip(t, y), unit='fullmoon'):
        if a == 2:  # full moon
            hit = None
            dt = x.utc_datetime().replace(microsecond=0)
            if dt == lastdt:
                # skyfield provides down to the microsecond precision, we dont need that
                continue
            lastdt = dt
            total += 1

            datebyzone = {'UTC': x.astimezone(UTC),
                          'Eastern': x.astimezone(eastern),
                          'Central': x.astimezone(central),
                          'Mountain': x.astimezone(mountain),
                          'Pacific': x.astimezone(pacific),
                          'Alaska': x.astimezone(alaska),
                          'Hawaii': x.astimezone(hawaii),
                          'Dominican': x.astimezone(dom),
                          }

            for tz in datebyzone.keys():
                if (datebyzone[tz].strftime("%a %d") == 'Fri 13')  or \
                    (tz == 'UTC' and \
                     (datebyzone['UTC'].strftime("%a %d") == 'Sat 14' and int(
                      datebyzone['UTC'].strftime("%H")) <= 12) or \
                     (datebyzone['UTC'].strftime("%a %d") == 'Thu 12' and int(
                      datebyzone['UTC'].strftime("%H")) >= 12)):

                    apogees, perigees = apsis(year=datebyzone[tz].year)
                    supermini = apsisconincidence(apogees, perigees, datebyzone[tz])

                    fri13[tz].append(datebyzone[tz])   # Fri Full Moon

                    if supermini['within 24 hours']['perigee']:  # Fri 13 super moon
                        fri13perigee[tz].append(datebyzone[tz])

                    if supermini['within 24 hours']['apogee']:   # Fri 13 mini moon
                        fri13apogee[tz].append(datebyzone[tz])

    return fri13, fri13apogee, fri13perigee, total

def dump_html(day, fri13, fri13apogee, fri13perigee, month, startyear, total, years):
    '''
    outputs the data in html files for easy cut/paste into emails
    :param fri13:  list of friday the 13th full moons
    :param fri13apogee:  list of friday the 13th mini moons
    :param fri13perigee: list of friday the 13th super moon
    :param month: starting month
    :param day: starting day of the month
    :param startyear: starting year
    :param total: total full moons in the time period
    :param years: years in the time period
    :return:  nothing
    '''
    for timezone in list(fri13.keys())[1:]:
        filename = f"results/FRI13{timezone}.html"
        htmlstr = f"Looking at the {total:,} full moons across 1000 years of the Gregorian Calendar (beginning {month}/{day}/{startyear})"
        # htmlstr += "<font size=\"10pt\" face=\"Arial\">\n"
        htmlstr += "<table border=1>\n"
        htmlstr += "<tr><th rowspan=2>timezone</th>"
        htmlstr += "<th colspan=3>Friday the 13th Full Moons</th>"
        htmlstr += "<th colspan=3>Friday the 13th Mini Moons</th>"
        htmlstr += "<th colspan=3>Friday the 13th Super Moons</th>"
        htmlstr += "<tr>"
        for i in range(3):
            htmlstr += " <th># (%)</th><th>most recent</br>(years since)</th><th>next</br>(years until)</th>\n"
        for tz in [timezone, 'UTC']:
            if tz == 'UTC':
                htmlstr += f"<tr><th>any timezone</th>\n"
            else:
                htmlstr += f"<tr><th>{tz}</th>\n"
            htmlstr += f"<td> {len(fri13[tz]):3} (%{100 * len(fri13[tz]) / total:.2f})</td>"
            htmlstr += mostrecentnext(fri13, tz)
            htmlstr += f"<td> {len(fri13apogee[tz]):4} (%{100 * len(fri13apogee[tz]) / total:.2f})</td>"
            htmlstr += mostrecentnext(fri13apogee, tz)
            htmlstr += f"<td> {len(fri13perigee[tz]):4} (%{100 * len(fri13perigee[tz]) / total:.2f})</td>"
            htmlstr += mostrecentnext(fri13perigee, tz)
            htmlstr += "</tr>\n"
        htmlstr += "</table>\n"
        text_file = open(filename, "w")
        text_file.write(htmlstr)
        text_file.close()

def apsisconincidence(apogees, perigees, targetdate):
    '''
    determines if a given datetime (a full or new moon) is within 24 hours of
    lunar perigee or apogee
    :param apogees:
    :param perigees:
    :param targetdate:
    :return: dictionary of boolean values for super and mini moon validity for the given date
    '''
    result = {'within 24 hours': {'apogee': False, 'perigee': False}}

    for dateapogee in apogees.keys():
        # loop through apogees
        delta = targetdate - dateapogee
        if delta.days == 0:
            result['within 24 hours']['apogee'] = True

    for dateperigee in perigees:
        # loop through perigees
        delta = targetdate - dateperigee
        if delta.days == 0:  # within 24 hours
            result['within 24 hours']['perigee'] = True
    return result

def mostrecentnext(datelist, tz):
    '''
    generate table cells containing most recent and next two events in the supplie
    list
    :param datelist:
    :param tz: timezone being considered
    :return:
    '''
    mostrecent = None
    next2 = []
    returnhtml = ""

    for dt in sorted(datelist[tz]):
        if dt < now:
            mostrecent = dt
        else:
            next2.append(dt)
    next2 = sorted(list(set(next2)))[:2]

    returnhtml += "<td>"
    if mostrecent is None:
        returnhtml += "none"
    else:
        returnhtml += mostrecent.strftime("%m/%d/%Y %H:%M</br>")
        mostrecentdeltayears = (now - mostrecent).days/365.0
        if mostrecentdeltayears < 1:
            returnhtml += f" {tz}"
        else:
            returnhtml += f" ({mostrecentdeltayears:.2f})"

    returnhtml += "</td>\n"

    # returnhtml += "<td style=\"font-family: Courier; font-size: 12;}\">"
    returnhtml += "<td>"
    if len(next2) == 0:
        returnhtml += "none"
    else:
        for dt in next2:
            returnhtml += dt.strftime("%m/%d/%Y %H:%M")
            nextdeltayears = (dt - now).days / 365.0
            if nextdeltayears < 1:
                returnhtml += f" {tz}</br>"
            else:
                returnhtml += f" ({nextdeltayears:.2f})</br>"
    returnhtml += "</td>\n"
    return returnhtml

class MyTestCases(unittest.TestCase):
    def test_something(self):
        # main(startyear=1832, month=1, day=1, years=150)
        main()

    def test_0_apsis(self):
        a, p = apsis(year=2019)
        self.assertEqual(len(a), 13)
        self.assertEqual(len(p), 13)

        a, p = apsis(year=2018)
        self.assertEqual(len(a), 13)
        self.assertEqual(len(p), 14)

    def test_1_minimoon(self):
        import pytz
        timezone = pytz.timezone("UTC")
        apogees, perigees = apsis(2019)
        dt = datetime.datetime(2019, 9, 14, 3, 32)
        dt_aware = timezone.localize(dt)
        result = apsisconincidence(apogees, perigees, dt_aware)
        self.assertTrue(result['within 24 hours']['apogee'])
        self.assertFalse(result['within 24 hours']['perigee'])

    def test_1_supermoon(self):
        import pytz
        timezone = pytz.timezone("UTC")
        apogees, perigees = apsis(2049)
        dt = datetime.datetime(2049, 8, 13, 9, 20)
        dt_aware = timezone.localize(dt)
        result = apsisconincidence(apogees, perigees, dt_aware)
        self.assertFalse(result['within 24 hours']['apogee'])
        self.assertTrue(result['within 24 hours']['perigee'])



if __name__ == '__main__':
    unittest.main()
