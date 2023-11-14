def isLeapYear(year):
    if year % 400 == 0:
        return True
    if year % 100 == 0:
        return False
    if year % 4 == 0:
        return True
    return False

def daysInMonth(year, month):
    if month == 1 or month == 3 or month == 5 or month == 7 \
       or month == 8 or month == 10 or month == 12:
        return 31
    else: 
        if month == 2:
            if isLeapYear(year):
                return 29
            else:
                return 28
        else:
            return 30

def nextDay(year, month, day):
    """ Warning: this version assumes every month has 30 days"""
    if day < daysInMonth(year, month):
        return year, month, day + 1
    else:
        if month < 12:
            return year, month + 1, 1
        else:
            return year + 1, 1, 1


def dateIsBefore(year1, month1, day1, year2, month2, day2):
    """Returns True if year1-month1-day1 is before year2-month2-day2. Otherwise, returns False."""
    if year1 < year2:
        return True
    if year1 == year2:
        if month1 < month2:
            return True
        if month1 == month2:
            return day1 < day2
    return False 


def daysBetweenDates(year1, month1, day1, year2, month2, day2):
    """Returns the number of days between year1/month1/day1
       and year2/month2/day2. Assumes inputs are valid dates
       in Gregorian calendar."""
    assert not dateIsBefore(year2, month2, day2, year1, month1, day1)

    days = 0
    while dateIsBefore(year1, month1, day1, year2, month2, day2):
        year1, month1, day1 = nextDay(year1, month1, day1)
        days += 1
    return days  


def test():
    assert daysBetweenDates(2013, 1, 1, 2013, 1, 1) == 0
    assert daysBetweenDates(2013, 1, 1, 2013, 1, 2) == 1   
    assert nextDay(2013, 1, 1) == (2013, 1, 2)
    assert nextDay(2013, 4, 30) == (2013, 5, 1)
    assert nextDay(2012, 12, 31) == (2013, 1, 1)
    assert nextDay(2013, 2, 28) == (2013, 3, 1)
    assert nextDay(2013, 9, 30) == (2013, 10, 1)
    assert nextDay(2012, 2, 28) == (2012, 2, 29)
    assert daysBetweenDates(2012, 1, 1, 2013, 1, 1) == 366  
    assert daysBetweenDates(2013, 1, 1, 2014, 1, 1) == 365
    assert daysBetweenDates(2013, 1, 24, 2013, 6, 29) == 156  
    print("Tests finished.")

test()