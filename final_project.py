import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, date, time
import datetime
from math import exp, sqrt
import math
from scipy import stats

'''
This defines the option class I use to test my function on. Consists of everything needed for both the
Monte Carlo method and Black-Scholes model: The underlying asset's price, the strike price, the 
risk free rate of return, the volitility, and the option's time left until maturity.
'''

class option_attributes:
    def __init__(self, underlying_price, strike_price, risk_free_rate, vol, time_left):
        self.underlying_price = underlying_price
        self.strike_price = strike_price
        self.risk_free_rate = risk_free_rate
        self.vol = vol
        self.time_left = time_left

'''
This function calculates the value of the option, so how much profit you make based off of the terms of the
contract itself. In this case, for a call option, you only make money when the strike price, the price you
agree to buy the asset at, is less than S, the asset's actual price, so S - strike must be positive. If
this is not the case, then the option will be worthless.
'''
def option_value(S, strike_price):
    value = max(S - strike_price, 0.0)
    return value

'''
This function utilizes nested for loops to simulate the price change of the underlying asset each day
until the date of maturity, and then values the option at that point. The outer for loop is for each
individual simulation trial, while the inner for loop is for each day until maturity. Note that for step_num
we multiply the time by 252 as opposed to 365 as there are only 252 trading days in a year, and those
are the only days that will impact our price. This function in particular also graphs one path, ten paths,
and all simulation paths in order to give a visualization of how the underlying asset's price is capable
of moving in both average and extreme scenarios. 
'''

def day_by_day_price_simulation(option, path_count):
    S_initial = option.underlying_price
    strike_price = option.strike_price
    time = option.time_left
    rate = option.risk_free_rate
    vol = option.vol
    step_num = int(time * 252)
    interval = time / step_num
    returns = []
    final_value = []
    x_ax = np.arange(0, step_num+1)
    for j in range(0, path_count):
        returns.append([S_initial])

        for i in range(1, step_num + 1):
            epsilon = np.random.standard_normal(1)
            next_price = returns[j][i-1] * np.exp((rate - 0.5 * vol ** 2) * \
                            interval + (vol * np.sqrt(interval) * epsilon))
            returns[j].append(next_price)
        plt.plot(x_ax, returns[j][:])
        final_value.append(option_value(returns[j][-1], strike_price))
    
    plt.title('Paths of All Simulations')
    plt.xlabel('Number of Days From Start')
    plt.ylabel('Underlying Asset Price')
    plt.show()
    plt.plot(x_ax, returns[4][:], color = 'black')
    plt.title('Path of Single Simulation')
    plt.xlabel('Number of Days From Start')
    plt.ylabel('Underlying Asset Price')
    plt.show()
    avg_list = []
    
    for i in range(0, step_num + 1):
        full_at_ind = []
        for j in range(0, path_count):
            full_at_ind.append(returns[j][i])

        avg_at_ind = np.mean(full_at_ind)
        avg_list.append(avg_at_ind)
    for j in range(0, 10):
        plt.plot(x_ax, returns[j][:])
    plt.title('Path of Ten Simulations')
    plt.xlabel('Number of Days From Start')
    plt.ylabel('Underlying Asset Price')
    plt.show()
    
    price = math.exp(- option.risk_free_rate * option.time_left) * (sum(final_value) / float(path_count))
    return price


'''
Same funciton as above, but removes the plotting so as to make it more convenient to run multiple times
with differing path counts.
'''

def day_by_day_price_simulation_noplot(option, path_count):
    S_initial = option.underlying_price
    strike_price = option.strike_price
    time = option.time_left
    rate = option.risk_free_rate
    vol = option.vol
    step_num = int(time * 252)
    interval = time / step_num
    returns = []
    final_value = []
    x_ax = np.arange(0, step_num+1)
    for j in range(0, path_count):
        returns.append([S_initial])

        for i in range(1, step_num + 1):
            epsilon = np.random.standard_normal(1)
            next_price = returns[j][i-1] * np.exp((rate - 0.5 * vol ** 2) * \
                            interval + (vol * np.sqrt(interval) * epsilon))
            returns[j].append(next_price)

        final_value.append(option_value(returns[j][-1], strike_price))

    price = math.exp(- option.risk_free_rate * option.time_left) * (sum(final_value) / float(path_count))
    return price


'''
This simply implements the Black-Scholes formula.
'''

def option_price_formula_bs(option):

    d1 = 1 / (option.vol * np.sqrt(option.time_left)) * \
    (np.log(option.underlying_price / option.strike_price) + \
    (option.risk_free_rate + (0.5 * option.vol ** 2)) * option.time_left)

    d2 = 1 / (option.vol * np.sqrt(option.time_left)) * \
    (np.log(option.underlying_price / option.strike_price) + \
    (option.risk_free_rate - (0.5 * option.vol ** 2)) * option.time_left)

    price = option.underlying_price * stats.norm.cdf(d1) - option.strike_price * \
    np.exp(-option.risk_free_rate * option.time_left) * stats.norm.cdf(d2)
    return price

'''
Creates the option to run our functions on. Underlying asset price is 1200, strike price is 1220,
risk free rate of return is .14%, volitility is 21.21%, and time left until maturity is .1616 years
'''
test_option = option_attributes(1200, 1220, 0.0014, 0.2121, \
(datetime.date(2014,7,3) - datetime.date(2014,5,5)).days / 365.0)

'''
Prints the output of the two methods for comparison
'''
print('Price Returned from Black-Scholes')
print(option_price_formula_bs(test_option))
print('Price Returned from Computational Simulation')
print(day_by_day_price_simulation(test_option, 1000))

'''
Below we run the simulation multiple times with varying path counts in order to determine the impact
the numbers of paths run has on the accuracy of the option's price when compared to the output from
Black-Scholes
'''
path_count_list = [100, 200, 300, \
400, 500, 600, 700, 800, 900, 1000, 1250, 1500, 1750, 2000, 2250, 2500, 2750, 3000]

opt_price_diff = []
bs_price = option_price_formula_bs(test_option)
for k in path_count_list:
    opt_price_diff.append(day_by_day_price_simulation_noplot(test_option, k) - bs_price)

plt.plot(path_count_list, opt_price_diff, color = 'black')
plt.title('Testing the Accuracy of the Monte Carlo Model Against Black-Scholes')
plt.ylabel('Difference Between Model Prices')
plt.xlabel('Number of Simulated Paths')
plt.show()




