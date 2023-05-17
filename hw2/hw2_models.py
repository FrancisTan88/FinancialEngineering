import numpy as np
from scipy.stats import norm, binom
from typing import List
from math import comb
import csv


N = norm.cdf


class BlackScholesModel:
    def __init__(self, s0, r, q, k, sigma, big_t) -> None:
        self.s0 = s0
        self.r = r
        self.q = q
        self.k = k
        self.sigma = sigma
        self.big_t = big_t

    def calculate_d1(self) -> float:
        return (np.log(self.s0/self.k) + self.big_t*(self.r - self.q + self.sigma**2/2)) / (self.sigma * self.big_t**0.5)
    
    def calculate_options(self) -> List:
        d1 = self.calculate_d1()
        d2 = d1 - self.sigma * self.big_t**0.5
        calls_price = self.s0 * np.exp(-self.q * self.big_t) * N(d1) - self.k * np.exp(-self.r * self.big_t) * N(d2)
        puts_price = self.k * np.exp(-self.r * self.big_t) * N(-d2) - self.s0 * np.exp(-self.q * self.big_t) * N(-d1)
        return calls_price, puts_price


class Monte_Carlo_Simulation(BlackScholesModel):
    def __init__(self, s0, r, q, sigma, big_t, k, sampling_times, repetitions) -> None:
        super(Monte_Carlo_Simulation, self).__init__(s0, r, q, k, sigma, big_t)
        self.sampling_times = sampling_times
        self.repetitions = repetitions

    def calculate_stock_price(self) -> float:
        mean = np.log(self.s0) + self.big_t * (self.r - self.q - (self.sigma**2)/2)
        std = self.sigma * self.big_t**0.5
        stock_price = mean + std * np.random.normal(0, 1)
        return np.exp(stock_price)

    # return price of calls, puts
    def payoff(self, stock_price):
        if stock_price > self.k:
            return stock_price - self.k, 0
        elif stock_price < self.k:
            return 0, self.k - stock_price
        else:
            return 0, 0

    def simulation(self) -> List:
        total_calls, total_puts = [], []
        for j in range(self.repetitions):
            calls_per_rep, puts_per_rep = [], []
            for i in range(self.sampling_times):
                stock_price = self.calculate_stock_price()
                calls, puts = self.payoff(stock_price)
                calls_per_rep.append(calls)
                puts_per_rep.append(puts)
            total_calls.append(np.mean(calls_per_rep) * np.exp(-self.r * self.big_t))
            total_puts.append(np.mean(puts_per_rep) * np.exp(-self.r * self.big_t))
        mean_calls = np.mean(total_calls)
        mean_puts = np.mean(total_puts)
        std_calls = np.std(total_calls)
        std_puts = np.std(total_puts)
        CI_calls = [mean_calls-2*std_calls, mean_calls+2*std_calls]
        CI_puts = [mean_puts-2*std_puts, mean_puts+2*std_puts]
        return CI_calls, CI_puts


# using 2D stocks, 2D options
class BinomialTree2D(BlackScholesModel):
    def __init__(self, s0, r, q, k, sigma, big_t, n) -> None:
        super(BinomialTree2D, self).__init__(s0, r, q, k, sigma, big_t)
        
        self.n = n
        self.dt = self.big_t / self.n
        self.u = np.exp(self.sigma * self.dt**0.5)
        self.d = 1 / self.u
        self.p = (np.exp((self.r - self.q) * self.dt) - self.d) / (self.u - self.d)

    # calculates stocks as (n+1)*(n+1) array
    def calculate_stock(self) -> List[List]:
        stocks = [[-1] * (self.n + 1) for i in range(self.n + 1)]
        for j in range(self.n + 1):
            for i in range(j+1):
                stocks[i][j] = self.s0 * self.u**(j-i) * self.d**i
        # with open("./BT2D_stocks.csv", "w+") as file:
        #     csv_writer = csv.writer(file, delimiter=",")
        #     csv_writer.writerows(stocks)
        return stocks
    
    def backward_induction(self) -> float:
        stocks = self.calculate_stock()
        european_calls = [[-1] * (self.n+1) for i in range(self.n+1)]
        european_puts = [[-1] * (self.n+1) for i in range(self.n+1)]
        american_calls = [[-1] * (self.n+1) for i in range(self.n+1)]
        american_puts = [[-1] * (self.n+1) for i in range(self.n+1)]
        df = np.exp(-self.r * self.dt)
        
        for i in range(self.n + 1):
            european_calls[i][self.n] = max(stocks[i][self.n] - self.k, 0)
            european_puts[i][self.n] = max(self.k - stocks[i][self.n], 0)
            american_calls[i][self.n] = max(stocks[i][self.n] - self.k, 0)
            american_puts[i][self.n] = max(self.k - stocks[i][self.n], 0)
        for j in range(self.n - 1, -1, -1):
            for i in range(j+1):
                european_calls[i][j] = df * (self.p * european_calls[i][j+1] + (1-self.p) * european_calls[i+1][j+1])
                european_puts[i][j] = df * (self.p * european_puts[i][j+1] + (1-self.p) * european_puts[i+1][j+1])
                
                # for American Options: compare the "option price" with the "exercise"
                american_calls[i][j] = max(stocks[i][j] - self.k, \
                                           df * (self.p * american_calls[i][j+1] + (1-self.p) * american_calls[i+1][j+1]))
                american_puts[i][j] = max(self.k - stocks[i][j], \
                                           df * (self.p * american_puts[i][j+1] + (1-self.p) * american_puts[i+1][j+1]))
        return european_calls[0][0], european_puts[0][0], american_calls[0][0], american_puts[0][0]


# using 1D stocks, 1D options
class BinomialTree1D(BinomialTree2D):
    def __init__(self, s0, r, q, k, sigma, big_t, n) -> None:
        super(BinomialTree1D, self).__init__(s0, r, q, k, sigma, big_t, n)

    def backward_induction(self) -> float:
        df = np.exp(-self.r * self.dt)
        european_calls = [-1] * (self.n + 1)
        european_puts = [-1] * (self.n + 1)
        american_calls = [-1] * (self.n + 1)
        american_puts = [-1] * (self.n + 1)
        
        for i in range(self.n + 1):
            european_calls[i] = max(self.s0 * self.u**(self.n - i) * self.d**i - self.k, 0)
            european_puts[i] = max(self.k - self.s0 * self.u**(self.n - i) * self.d**i, 0)
            american_calls[i] = max(self.s0 * self.u**(self.n - i) * self.d**i - self.k, 0)
            american_puts[i] = max(self.k - self.s0 * self.u**(self.n - i) * self.d**i, 0)
        for j in range(self.n - 1, -1, -1):
            for i in range(j+1):
                european_calls[i] = df * (self.p * european_calls[i] + (1-self.p) * european_calls[i+1])
                european_puts[i] = df * (self.p * european_puts[i] + (1-self.p) * european_puts[i+1])
                american_calls[i] = max(self.s0 * self.u**(j-i) * self.d**i - self.k, \
                                        df * (self.p * american_calls[i] + (1-self.p) * american_calls[i+1]))
                american_puts[i] = max(self.k - self.s0 * self.u**(j-i) * self.d**i, \
                                        df * (self.p * american_puts[i] + (1-self.p) * american_puts[i+1]))
        return european_calls[0], european_puts[0], american_calls[0], american_puts[0]


class Combinatorics(BinomialTree2D):
    def __init__(self, s0, r, q, k, sigma, big_t, n) -> None:
        super(Combinatorics, self).__init__(s0, r, q, k, sigma, big_t, n)
    
    def calculate(self) -> float:
        calls, puts = 0, 0
        for j in range(self.n):
            bin_dis = binom.pmf(self.n - j, self.n, self.p)
            calls += bin_dis * max(self.s0 * self.u**(self.n - j) * self.d**j - self.k, 0)
            puts += bin_dis * max(self.k - self.s0 * self.u**(self.n - j) * self.d**j, 0)
        calls = np.exp(-self.r * self.big_t) * calls
        puts = np.exp(-self.r * self.big_t) * puts
        return calls, puts
