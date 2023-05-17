import numpy as np
from typing import List


class Monte_Carlo_Simulation:
    def __init__(self, s0, r, q, sigma, big_t, k1, k2, k3, k4, sampling_times, repetitions) -> None:
        self.s0 = s0
        self.r = r
        self.q = q
        self.sigma = sigma
        self.big_t = big_t
        self.k1 = k1
        self.k2 = k2
        self.k3 = k3
        self.k4 = k4
        self.sampling_times = sampling_times
        self.repetitions = repetitions

    def calculate_stock_price(self) -> float:
        mean = np.log(self.s0) + self.big_t * \
            (self.r - self.q - (self.sigma**2)/2)
        std = self.sigma * (self.big_t**0.5)
        stock_price = mean + std * np.random.normal(0, 1)
        return np.exp(stock_price)

    def payoff(self, stock_price) -> float:
        if stock_price >= self.k1 and stock_price <= self.k2:
            return max(stock_price - self.k1, 0)
        elif stock_price >= self.k2 and stock_price <= self.k3:
            return max(self.k2 - self.k1, 0)
        elif stock_price >= self.k3 and stock_price <= self.k4:
            return max(((self.k2-self.k1)/(self.k4-self.k3)) * (self.k4-stock_price), 0)
        else:
            return 0

    def simulation(self) -> List:
        arr_repetitions = []
        for j in range(self.repetitions):
            calls_price = []
            for i in range(self.sampling_times):
                stock_price = self.calculate_stock_price()
                calls_price.append(self.payoff(stock_price))
            arr_repetitions.append(np.mean(calls_price)
                                   * np.exp(-self.r * self.big_t))
        mean_repetitions = np.mean(arr_repetitions)
        std_repetitions = np.std(arr_repetitions)
        confidence_interval = [mean_repetitions-2 *
                               std_repetitions, mean_repetitions+2*std_repetitions]
        return [mean_repetitions, std_repetitions, confidence_interval]
