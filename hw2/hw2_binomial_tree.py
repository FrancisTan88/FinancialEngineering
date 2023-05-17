import numpy as np
from typing import List
from math import comb


# using 2D stocks, 2D options
class BinomialTree2D:
    def __init__(self, s0, r, q, k, sigma, big_t, n) -> None:
        self.s0 = s0
        self.r = r
        self.q = q
        self.k = k
        self.sigma = sigma
        self.big_t = big_t
        self.n = n

        self.dt = self.big_t / self.n
        self.u = np.exp(self.sigma * self.dt**0.5)
        self.d = 1 / self.u
        self.p = (np.exp((self.r - self.q) * self.dt) - self.d) / (self.u - self.d)

    def calculate_stock(self) -> List[List]:
        stocks = [[-1] * (self.n + 1) for i in range(self.n + 1)]
        for j in range(self.n + 1):
            for i in range(j+1):
                stocks[i][j] = self.s0 * self.u**(j-i) * self.d**i
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
                american_calls[i][j] = max(stocks[i][j] - self.k, \
                                           df * (self.p * american_calls[i][j+1] + (1-self.p) * american_calls[i+1][j+1]))
                american_puts[i][j] = max(self.k - stocks[i][j], \
                                           df * (self.p * american_puts[i][j+1] + (1-self.p) * american_puts[i+1][j+1]))
        return european_calls[0][0], european_puts[0][0], american_calls[0][0], american_puts[0][0]


# using 1D stocks, 1D options
class BinomialTree1D(BinomialTree2D):
    def __init__(self, s0, r, q, k, sigma, big_t, n) -> None:
        super(BinomialTree1D, self).__init__(s0, r, q, k, sigma, big_t, n)

    def calculate_stock(self) -> List[List]:
        stocks = []
        for i in range(self.n + 1):
            stocks.append(self.s0 * self.u**(self.n - i) * self.d**i)
        return stocks

    def backward_induction(self) -> float:
        stocks = self.calculate_stock()
        df = np.exp(-self.r * self.dt)
        european_calls = [-1] * (self.n + 1)
        european_puts = [-1] * (self.n + 1)
        american_calls = [-1] * (self.n + 1)
        american_puts = [-1] * (self.n + 1)
        
        for i in range(self.n + 1):
            european_calls[i] = max(stocks[i] - self.k, 0)
            european_puts[i] = max(self.k - stocks[i], 0)
            american_calls[i] = max(stocks[i] - self.k, 0)
            american_puts[i] = max(self.k - stocks[i], 0)
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
            calls += comb(self.n, j) * self.p**(self.n - j) * (1-self.p)**j * \
                max(self.s0 * self.u**(self.n - j) * self.d**j - self.k, 0)
            puts += comb(self.n, j) * self.p**(self.n - j) * (1-self.p)**j * \
                max(self.k - self.s0 * self.u**(self.n - j) * self.d**j, 0)
        calls = np.exp(-self.r * self.big_t) * calls
        puts = np.exp(-self.r * self.big_t) * puts
        return calls, puts


