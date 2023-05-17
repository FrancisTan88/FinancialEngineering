import numpy as np
from typing import List


class Monte_Carlo_Simulation:
    def __init__(self, s0, r, q, sigma, big_t, k, sampling_times, repetitions) -> None:
        self.s0 = s0
        self.r = r
        self.q = q
        self.sigma = sigma
        self.big_t = big_t
        self.k = k
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
