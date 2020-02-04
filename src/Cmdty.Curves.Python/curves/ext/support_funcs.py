from datetime import timedelta


def time_func(period1, period2): 
    return int((period2 - period1) / timedelta(hours=1))
    

def weighting(time_period):
    start = time_period
    end = time_period + timedelta(hours=1)
    return (end - start)/timedelta(minutes=1)


def mult_adjust_func(time_period): return 1.0
    

def add_adjust_func(time_period): return 0.0


def delta_pow(time_to_start, time_to_end, power):
    return pow(time_to_end, power) - pow(time_to_start, power)


def enumerate_hours(start_date, end_date):
    for n in range(int((end_date - start_date)/timedelta(hours=1) + 1)):
        yield start_date + timedelta(hours=n)
