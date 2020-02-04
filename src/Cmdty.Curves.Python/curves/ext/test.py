from datetime import datetime, timedelta
from support_funcs import (time_func, weighting, mult_adjust_func, add_adjust_func)
from spline_refactor import build


def create_contract(start, end, price):
    return {
        'Start': start,
        'End': end,
        'Price': price
    }


def test_scenario1():
    flatPrice = 84.54

    contracts = []
    contracts.append(create_contract(datetime(2019, 1, 1),
                                     datetime(2019, 1, 31, 23), flatPrice))
    contracts.append(create_contract(datetime(2019, 2, 1),
                                     datetime(2019, 2, 28, 23), flatPrice))
    contracts.append(create_contract(datetime(2019, 3, 1),
                                     datetime(2019, 3, 31, 23), flatPrice))
    contracts.append(create_contract(datetime(2019, 4, 1),
                                     datetime(2019, 6, 30, 23), flatPrice))
    contracts.append(create_contract(datetime(2019, 7, 1),
                                     datetime(2019, 9, 30, 23), flatPrice))
    contracts.append(create_contract(datetime(2019, 10, 1),
                                     datetime(2020, 3, 31, 23), flatPrice))

    curve = build(contracts, weighting, mult_adjust_func,
                  add_adjust_func, time_func)

    for price in curve.values:
        assert round(flatPrice, 3) == round(price, 3)



def test_scenario2():
    intercept = 45.7
    dailySlope = 0.8

    dailyPrice = intercept
    contractDay = datetime(2019, 5, 11)
    contracts = []

    for i in range(14):
        contracts.append(create_contract(
            contractDay, contractDay + timedelta(hours=23), dailyPrice))
        contractDay += timedelta(days=1)
        dailyPrice += dailySlope

    curve = build(contracts, weighting, mult_adjust_func,
                  add_adjust_func, time_func)

    hourlySlope = dailySlope / 24

    for pricePairHour1, pricePairHour2 in zip(curve.values, curve.values[1:]):
        hourlyChange = pricePairHour2 - pricePairHour1
        assert round(hourlySlope, 3) == round(hourlyChange, 3)


if __name__ == "__main__":
    test_scenario1()
    test_scenario2()




