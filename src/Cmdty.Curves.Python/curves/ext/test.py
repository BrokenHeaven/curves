from datetime import datetime, timedelta
from spline_back import build


def time_func(period1, period2):
    return int((period2 - period1) / timedelta(hours=1))


def weighting(time_period):
    start = time_period
    end = time_period + timedelta(hours=1)
    return (end - start)/timedelta(minutes=1)


def mult_adjust_func(time_period):
    return 1.0


def add_adjust_func(time_period):
    return 0.0


def create_contract(date, price):
    start = date
    end = date + timedelta(hours=23)
    return {
        'Start': start,
        'End': end,
        'Price': price
    }

if __name__ == "__main__":
    intercept = 45.7
    dailySlope = 0.8

    dailyPrice = intercept
    contractDay = datetime(2019, 5, 11)
    contracts = []

    for i in range(14):
        contracts.append(create_contract(contractDay, dailyPrice))
        contractDay += timedelta(days=1)
        dailyPrice += dailySlope

    #weights = [(x['End'] - x['Start']).seconds/60 for x in contracts]
    #multiAdj = [1]*len(weights)
    #addAdj = [0]*len(weights)
    curve = build(contracts, weighting, mult_adjust_func,
                  add_adjust_func, time_func)


