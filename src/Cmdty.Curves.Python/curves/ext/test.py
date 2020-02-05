from datetime import datetime, date, timedelta
from support_funcs import (time_func, weighting, mult_adjust_func, add_adjust_func)
#from curves._common import deconstruct_contract
from spline_refactor import build


class Contract:
    def __init__(self, contract_tpl, offset='days'):
        self.Start = contract_tpl[0][0]
        self.End = contract_tpl[0][1]
        self.Price = contract_tpl[1]
        self.Offset = offset


def create_contract(start, end, price):
    return {
        'Start': start,
        'End': end,
        'Price': price
    }


def create_contract_tpl(date, price):
    return ((date[0], date[1]), price)


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


def test_scenario3():
    flatPrice = 84.54

    contracts = []
    contracts.append(create_contract_tpl((date(2019, 1, 1),
                                     date(2019, 1, 31)), flatPrice))
    contracts.append(create_contract_tpl((date(2019, 2, 1),
                                     date(2019, 2, 28)), flatPrice))
    contracts.append(create_contract_tpl((date(2019, 3, 1),
                                     date(2019, 3, 31)), flatPrice))
    contracts.append(create_contract_tpl((date(2019, 4, 1),
                                     date(2019, 6, 30)), flatPrice))
    contracts.append(create_contract_tpl((date(2019, 7, 1),
                                     date(2019, 9, 30)), flatPrice))
    contracts.append(create_contract_tpl((date(2019, 10, 1),
                                     date(2020, 3, 31)), flatPrice))

    def deconstruct_contract(contract):
        if len(contract) == 2:
            (period, price) = contract
        elif len(contract) == 3:
            (period, price) = ((contract[0], contract[1]), contract[2])
        else:
            raise ValueError("contract tuple must have either 2 or 3 items")
        return (period, price)

    decons = []
    for contract in contracts:
        decons.append(Contract(deconstruct_contract(contract)))

    curve = build(decons, weighting, mult_adjust_func,
                  add_adjust_func, time_func)

    for price in curve.values:
        assert round(flatPrice, 3) == round(price, 3)


if __name__ == "__main__":
    #test_scenario1()
    #test_scenario2()
    test_scenario3()



