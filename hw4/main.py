import numpy as np
import csv
import copy


def write_csv(path, data):
    with open(path, "wt") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerows(data)

class BinomialTree:
    def __init__(self, St, r, q, sigma, t, T, Smax_t, n, nums_sim, nums_rep) -> None:
        self.St = St
        self.r = r
        self.q = q
        self.sigma = sigma
        self.t = t
        self.T = T
        self.Smax_t = Smax_t
        self.n = n
        self.nums_sim = nums_sim
        self.nums_rep = nums_rep

        self.dt = (self.T - self.t) / self.n
        self.u = np.exp(self.sigma * (self.dt**0.5))
        self.d = 1 / self.u
        self.p = (np.exp((self.r - self.q) * self.dt) - self.d) / (self.u - self.d)

    # -> (n+1) * (n+1)
    def calculate_stock(self):
        # process the "numerical error problem" 
        def calibration(stocks):
            for j in range(len(stocks[0]) - 2):
                for i in range(j+1):
                    stocks[i+1][j+2] = stocks[i][j]
            return stocks

        stocks = [[-1] * (self.n + 1) for _ in range(self.n + 1)]
        for j in range(self.n + 1):
            for i in range(j+1):
                stocks[i][j] = self.St * self.u**(j-i) * self.d**i
        return calibration(stocks)

    # calculate the S_max for each node -> 3D array
    def forward_tracking(self, stocks):
        s_maxss = [[-1] * (self.n + 1) for _ in range(self.n + 1)]
        ini = {}
        ini[self.Smax_t] = -1
        s_maxss[0][0] = ini
        for j in range(1, self.n+1):
            for i in range(j+1):
                curr_dict = {}
                if i == 0:
                    for k, v in s_maxss[i][j-1].items():
                        curr_dict[max(k, stocks[i][j])] = -1
                elif i == j:
                    for k, v in s_maxss[i-1][j-1].items():
                        curr_dict[max(k, stocks[i][j])] = -1
                else:
                    for k, v in s_maxss[i-1][j-1].items():
                        curr_dict[max(k, stocks[i][j])] = -1
                    for k, v in s_maxss[i][j-1].items():
                        curr_dict[max(k, stocks[i][j])] = -1
                s_maxss[i][j] = curr_dict
        return s_maxss

    # For European puts
    def backward_induction_euro(self, stocks, S_maxes):
        factor = np.exp(-self.r * self.dt)
        for j in range(len(S_maxes)-1, -1, -1):
            for i in range(j+1):
                # for the last column: calculate its payoff directly
                if j == len(S_maxes)-1:
                    for k, v in S_maxes[i][j].items():
                        S_maxes[i][j][k] = k - stocks[i][j]
                # other than the last column
                else:
                    for k, v in S_maxes[i][j].items():
                        if k in S_maxes[i][j+1]:
                            S_maxes[i][j][k] = \
                                factor * (self.p*S_maxes[i][j+1][k] + (1-self.p)*S_maxes[i+1][j+1][k])
                        else:
                            S_maxes[i][j][k] = \
                                factor * (self.p*S_maxes[i][j+1][stocks[i][j+1]] + (1-self.p)*S_maxes[i+1][j+1][k])
        return S_maxes[0][0][self.Smax_t]
    
    # For American puts
    def backward_induction_usa(self, stocks, S_maxes):
        factor = np.exp(-self.r * self.dt)
        for j in range(len(S_maxes)-1, -1, -1):
            for i in range(j+1):
                # for the last column: calculate its payoff directly
                if j == len(S_maxes)-1:
                    for k, v in S_maxes[i][j].items():
                        S_maxes[i][j][k] = k - stocks[i][j]
                # other than the last column
                else:
                    for k, v in S_maxes[i][j].items():
                        if k in S_maxes[i][j+1]:
                            S_maxes[i][j][k] = \
                                max(factor * (self.p*S_maxes[i][j+1][k] + (1-self.p)*S_maxes[i+1][j+1][k]),
                                    k - stocks[i][j])
                        else:
                            S_maxes[i][j][k] = \
                                max(factor * (self.p*S_maxes[i][j+1][stocks[i][j+1]] + (1-self.p)*S_maxes[i+1][j+1][k]),
                                    k - stocks[i][j])
        return S_maxes[0][0][self.Smax_t]

    # output euro puts and american puts
    def main(self):
        stocks = self.calculate_stock()
        s_maxs_euro = self.forward_tracking(stocks)
        s_maxs_usa = copy.deepcopy(s_maxs_euro)  # prepare a clone for american puts
        puts_euro = self.backward_induction_euro(stocks, s_maxs_euro)
        puts_usa = self.backward_induction_usa(stocks, s_maxs_usa)
        return puts_euro, puts_usa


"""
Compared to the original binomial tree,
I use a faster approach to determine the "S_Max List" for the arbitrary node,
and by using this approach, we can also determine the "S_Max List" of any nodes
without considering its parents' or any other nodes' "S_Max List",
namely, we only need to know the stock prices.
""" 
class BinomialTreeQuickApproach(BinomialTree):
    def __init__(self, St, r, q, sigma, t, T, Smax_t, n, nums_sim, nums_rep) -> None:
        super(BinomialTreeQuickApproach, self).__init__(St, r, q, sigma, t, T, Smax_t, n, nums_sim, nums_rep)

    # -> (n+1) * (n+1)
    def calculate_stock_q(self):
        # process the "numerical error problem" 
        def calibration(stocks):
            for j in range(len(stocks[0]) - 2):
                for i in range(j+1):
                    stocks[i+1][j+2] = stocks[i][j]
            return stocks
        stocks = [[-1] * (self.n + 1) for _ in range(self.n + 1)]
        for j in range(self.n + 1):
            for i in range(j+1):
                stocks[i][j] = self.St * self.u**(j-i) * self.d**i
        return calibration(stocks)

    def cal_smax_quick(self, stocks):
        s_maxs = [[-1] * (self.n + 1) for _ in range(self.n + 1)]
        s_maxs[0][0] = {self.Smax_t: -1}
        nodes = 1
        for j in range(1, self.n+1):
            mid = nodes // 2
            for i in range(j+1):
                curr_dict = {}
                if i == 0:
                    curr_dict[max(self.Smax_t, stocks[i][j])] = -1
                elif i == j:
                    curr_dict[self.Smax_t] = -1
                else:
                    if i <= mid:
                        row, col = 0, j-i
                        while row <= i:
                            curr_dict[max(self.Smax_t, stocks[row][col])] = -1
                            if stocks[row][col] <= self.Smax_t:
                                break
                            row += 1
                            col += 1
                    else:
                        """
                        because of Python data structures' properties(say, a points to b,
                            then b will change if we change a),
                        we need to use deepcopy instead of assignment(or say, pointer),
                        otherwise, we would have problems in the "backward induction" part.
                        """
                        curr_dict = copy.deepcopy(s_maxs[i-1][j-1])  
                s_maxs[i][j] = curr_dict
            nodes += 1
        return s_maxs

    # output euro puts and american puts
    def main(self):
        stocks = self.calculate_stock_q()
        s_maxs_euro = self.cal_smax_quick(stocks)
        s_maxs_usa = copy.deepcopy(s_maxs_euro)  # prepare a clone for american puts
        puts_euro = self.backward_induction_euro(stocks, s_maxs_euro)
        puts_usa = self.backward_induction_usa(stocks, s_maxs_usa)
        return puts_euro, puts_usa


class MonteCarlo(BinomialTree):
    def __init__(self, St, r, q, sigma, t, T, Smax_t, n, nums_sim, nums_rep) -> None:
        super(MonteCarlo, self).__init__(St, r, q, sigma, t, T, Smax_t, n, nums_sim, nums_rep)

    # this is for one of reps (i.e. 10000 paths) -> List
    def cal_stock(self):
        sps = np.zeros((self.nums_sim, self.n+2))
        sps[:, 0] = self.Smax_t
        sps[:, 1] = self.St
        rands = np.random.normal(0, 1, size=(self.nums_sim, self.n+2))
        for col in range(2, self.n+2):
            sps[:, col] = np.exp((np.log(sps[:, col-1]) + self.dt*(self.r - self.q - (self.sigma**2)/2)) + \
                            rands[:, col] * (self.sigma*(self.dt**0.5)))
        return sps

    # payoff for euro puts -> float
    def cal_payoff(self, sps):
        payoffs = np.max(sps, axis=1) - sps[:, -1]
        return np.exp(-self.r*(self.T - self.t)) * np.mean(payoffs)

    def main(self):
        reps = []
        for _ in range(self.nums_rep):
            sps = self.cal_stock()
            reps.append(self.cal_payoff(sps))
        reps = np.array(reps)
        return np.mean(reps), np.std(reps)


"""
The implementation of the method proposed by Cheuk and Vorst.
It aims to change the pricing units(i.e. stock price -> number of shares)
Note: the Smax_t in this method is constrainted to be equal to St
"""
class CheukandVorst(BinomialTree):
    def __init__(self, St, r, q, sigma, t, T, Smax_t, n, nums_sim, nums_rep) -> None:
        super().__init__(St, r, q, sigma, t, T, Smax_t, n, nums_sim, nums_rep)
        
        self.m = np.exp((self.r - self.q) * self.dt)
        self.p = 1 - ((self.m * self.u - 1) / (self.m * self.u - self.m * self.d))

    def cal_stocks(self):
        stocks_euro, stocks_usa = np.zeros((n + 1, n + 1)), np.zeros((n + 1, n + 1))
        for i in range(n+1):
            for j in range(n+1-i):
                stocks_euro[i][j] = (self.u**j - 1)
                stocks_usa[i][j] = (self.u**j - 1)
        return stocks_euro, stocks_usa

    def backward_induction(self, stocks_euro, stocks_usa):
        for i in range(1, n+1):
            for j in range(n+1-i):
                if j == 0:
                    stocks_euro[i][j] = ((1-self.p) * stocks_euro[i-1][0] + 
                                         self.p * stocks_euro[i-1][1])
                    stocks_usa[i][j] = max(((1-self.p) * stocks_usa[i-1][0] + 
                                            self.p * stocks_usa[i-1][1]), stocks_usa[i][j])
                else:
                    stocks_euro[i][j] = ((1-self.p) * stocks_euro[i-1][j-1] + 
                                            self.p * stocks_euro[i-1][j+1])
                    stocks_usa[i][j] = max(((1-self.p) * stocks_usa[i-1][j-1] + 
                                            self.p * stocks_usa[i-1][j+1]), stocks_usa[i][j])
        puts_euro, puts_usa = self.St*stocks_euro[self.n][0], self.St*stocks_usa[self.n][0]
        return puts_euro, puts_usa

    def main(self):
        stocks_euro, stocks_usa = self.cal_stocks()
        puts_euro, puts_usa = self.backward_induction(stocks_euro, stocks_usa)
        return puts_euro, puts_usa


if __name__ == "__main__":
    S_t = float(input())
    r = float(input())
    q = float(input())
    sigma = float(input())
    t = float(input())
    T = float(input())
    Sim_n = int(input())
    Rep_n = int(input())
    S_max = float(input())
    n = int(input())

    bt = BinomialTree(S_t, r, q, sigma, t, T, S_max, n, Sim_n, Rep_n)
    puts_euro, puts_usa = bt.main()
    print(f"\nBinomial Tree:\n(1)European puts = {round(puts_euro, 4)}\n(2)American puts = {round(puts_usa, 4)}")

    bt_quick = BinomialTreeQuickApproach(S_t, r, q, sigma, t, T, S_max, n, Sim_n, Rep_n)
    puts_euro_q, puts_usa_q = bt_quick.main()
    print(f"\nBinomial Tree Quick Approach:\n(1)European puts = {round(puts_euro_q, 4)}\n(2)American puts = {round(puts_usa_q, 4)}")

    mc = MonteCarlo(S_t, r, q, sigma, t, T, S_max, n, Sim_n, Rep_n)
    mu, std = mc.main()
    print(f"\nMonte Carlo:\n(1)European puts = {round(mu, 4)}\n(2)Confidence Interval = [{round(mu-2*std, 4)}, {round(mu+2*std, 4)}]")

    cv = CheukandVorst(S_t, r, q, sigma, t, T, S_max, n, Sim_n, Rep_n)
    puts_euro_cv, puts_usa_cv = cv.main()
    print(f"\nCheuk and Vorst Method:\n(1)European puts = {round(puts_euro_cv, 4)}\n(2)American puts = {round(puts_usa_cv, 4)}\n")
