from datetime import timedelta


def offset(delta, step):
    if delta=='days':
        return timedelta(days=step)
    elif delta=='hours':
        return timedelta(hours=step)
    else:
        return timedelta(months=step)


def time_func(period1, period2, delta):
    return int((period2 - period1) / offset(delta, 1))
    

def weighting(time_period, delta):
    start = time_period
    end = time_period + offset(delta, 1)
    return (end - start)/timedelta(minutes=1)


def mult_adjust_func(time_period): return 1.0
    

def add_adjust_func(time_period): return 0.0


def delta_pow(time_to_start, time_to_end, power):
    return pow(time_to_end, power) - pow(time_to_start, power)


def enumerate_periods(start_date, end_date, delta):
    for n in range(int((end_date - start_date)/offset(delta, 1) + 1)):
        yield start_date + offset(delta, n)
