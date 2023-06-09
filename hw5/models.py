import numpy as np
import datetime


# if A equals one of values in list, just return it.
# otherwise, use "linear interpolation" with the two values that enclose the A. 
def sequential_search(avg_lst, A):
    if A >= avg_lst[0][0]:
        return avg_lst[0][1]
    elif A <= avg_lst[-1][0]:
        return avg_lst[-1][1]
    for k in range(1, len(avg_lst)):
        if A == avg_lst[k][0]:
            return avg_lst[k][1]
        elif A > avg_lst[k][0]:
            weight = (avg_lst[k-1][0] - A) / (avg_lst[k-1][0] - avg_lst[k][0])
            return weight*avg_lst[k][1] + (1-weight)*avg_lst[k-1][1]
    raise ValueError("Something goes wrong with the sequential search function !!!")

def binary_search(avg_lst, A):
    if A >= avg_lst[0][0]:
        return avg_lst[0][1]
    elif A <= avg_lst[-1][0]:
        return avg_lst[-1][1]
    
    left, right = 0, len(avg_lst)-1
    while left < right:
        mid = (left+right) // 2
        if A == avg_lst[mid][0]:
            return avg_lst[mid][1]
        elif A > avg_lst[mid][0]:
            right = mid
        else:
            left = mid + 1
    if A == avg_lst[left][0]: 
        return avg_lst[left][1]
    else:
        weight = (avg_lst[left-1][0] - A) / (avg_lst[left-1][0] - avg_lst[left][0])
        return weight*avg_lst[left][1] + (1-weight)*avg_lst[left-1][1]

"""
Note: if we adopt the logarithmically placed method,
we have to take the logarithm to the A_max, A_min, Au(or Ad)
when using the linear interpolation.
"""
def linear_interpolation(avg_lst, A, nums_cut, logarithmically):
    if A >= avg_lst[0][0]:
        return avg_lst[0][1]
    elif A <= avg_lst[-1][0]:
        return avg_lst[-1][1]
    
    # if avg list of every node is placed logarithmically
    if logarithmically:
        maximum, minimum = np.log(avg_lst[0][0]), np.log(avg_lst[-1][0])
        pos = nums_cut * ((maximum - np.log(A)) / (maximum - minimum))
        if pos == int(pos):
            return avg_lst[int(pos)][1]
        else:
            weight = (avg_lst[int(pos)][0] - A) / (avg_lst[int(pos)][0] - avg_lst[int(pos)+1][0])
            return weight*avg_lst[int(pos)+1][1] + (1-weight)*avg_lst[int(pos)][1]
    # if avg list of every node is placed linearly
    else:
        maximum, minimum = avg_lst[0][0], avg_lst[-1][0]
        pos = nums_cut * ((maximum - A) / (maximum - minimum))
        if pos == int(pos):
            return avg_lst[int(pos)][1]
        else:
            weight = (avg_lst[int(pos)][0] - A) / (avg_lst[int(pos)][0] - avg_lst[int(pos)+1][0])
            return weight*avg_lst[int(pos)+1][1] + (1-weight)*avg_lst[int(pos)][1]


class MonteCarlo:
    def __init__(self, St, K, r, q, sigma, t, T_minus_t, M, n, S_ave_t, nums_sim, nums_rep) -> None:
        self.St = St
        self.K = K
        self.r = r
        self.q = q
        self.sigma = sigma
        self.t = t
        self.T_minus_t = T_minus_t
        self.M = M
        self.n = n
        self.S_ave_t = S_ave_t
        self.nums_sim = nums_sim
        self.nums_rep = nums_rep

        self.dt = self.T_minus_t / self.n
        self.periods_before_t = 1 + self.t/self.dt  # include t itself
        self.total_price_before_t = self.periods_before_t * self.S_ave_t  # include St itself

    # this is for one of reps (i.e. 10000 paths) -> List[avg stock price over whole period]
    def cal_avg(self):
        sps = np.zeros((self.nums_sim, self.n+2))
        rands = np.random.normal(0, 1, size=(self.nums_sim, self.n+2))
        sps[:, 0] = self.S_ave_t * self.periods_before_t - self.St
        sps[:, 1] = self.St
        std = self.sigma * (self.dt**0.5)
        for col in range(2, self.n+2):
            sps[:, col] = np.exp((np.log(sps[:, col-1]) + self.dt*(self.r - self.q - (self.sigma**2)/2)) + \
                            rands[:, col] * std)
        return np.sum(sps, axis=1) / (self.periods_before_t + self.n)

    # payoff for euro calls -> float
    def cal_payoff(self, avgs):
        payoffs = np.where(avgs > self.K, avgs - self.K, 0)
        return np.mean(payoffs) * np.exp(-self.r*(self.T_minus_t))

    def main(self):
        reps = []
        for _ in range(self.nums_rep):
            avgs = self.cal_avg()
            reps.append(self.cal_payoff(avgs))
        reps = np.array(reps)
        return np.mean(reps), np.std(reps)


class BinomialTree(MonteCarlo):
    def __init__(self, St, K, r, q, sigma, t, T_minus_t, M, n, S_ave_t, nums_sim, nums_rep) -> None:
        super().__init__(St, K, r, q, sigma, t, T_minus_t, M, n, S_ave_t, nums_sim, nums_rep)

        self.u = np.exp(self.sigma * (self.dt**0.5))
        self.d = 1 / self.u
        self.p = (np.exp((self.r - self.q) * self.dt) - self.d) / (self.u - self.d)

    def create_nodes_linearly_cut(self):
        nodes = np.zeros((self.n+1, self.n+1, self.M+1, 2))  # 4D array
        for j in range(self.n+1):
            for i in range(j+1):
                # calculate A_max, A_min
                A_max = (self.St + self.St * self.u * ((1-self.u**(j-i)) / (1-self.u)) + 
                            self.St * (self.u**(j-i)) * self.d * ((1-self.d**i) / (1-self.d)) +
                                (self.total_price_before_t - self.St)) / (self.periods_before_t + j)
                A_min = (self.St + self.St * self.d * ((1-self.d**i) / (1-self.d)) + 
                            self.St * (self.d**i) * self.u * ((1-self.u**(j-i)) / (1-self.u)) +
                                (self.total_price_before_t - self.St)) / (self.periods_before_t + j)
                # calculate representative avg price
                for k in range(self.M+1):
                    nodes[i][j][k][0] = (self.M-k)/self.M * A_max + k/self.M * A_min
                    nodes[i][j][k][1] = -1

        return nodes

    # the only diff with "linearly cut" is that we take logarithm to A_max, A_min,
    # and then take exponential to the whole formula when calculating representative avg price.
    def create_nodes_logarithmically_cut(self):
        nodes = np.zeros((self.n+1, self.n+1, self.M+1, 2))  # 4D array
        for j in range(self.n+1):
            for i in range(j+1):
                # calculate A_max, A_min
                A_max = (self.St + self.St * self.u * ((1-self.u**(j-i)) / (1-self.u)) + 
                            self.St * (self.u**(j-i)) * self.d * ((1-self.d**i) / (1-self.d)) +
                                (self.total_price_before_t - self.St)) / (self.periods_before_t + j)
                A_min = (self.St + self.St * self.d * ((1-self.d**i) / (1-self.d)) + 
                            self.St * (self.d**i) * self.u * ((1-self.u**(j-i)) / (1-self.u)) +
                                (self.total_price_before_t - self.St)) / (self.periods_before_t + j)
                # calculate representative avg price
                for k in range(self.M+1):
                    nodes[i][j][k][0] = np.exp((self.M-k)/self.M * np.log(A_max) + k/self.M * np.log(A_min))
                    nodes[i][j][k][1] = -1

        return nodes
    
    # for european calls
    def backward_induction_euro(self, nodes, search_way, logly):
        # calculate the payoffs for the last column
        for row in range(self.n+1):
            nodes[row][-1] = np.array(nodes[row][-1])
            nodes[row][-1][:, 1] = np.where(nodes[row][-1][:, 0] > self.K, 
                                            nodes[row][-1][:, 0] - self.K, 0)
        # backward induction
        for col in range(self.n-1, -1, -1):
            for row in range(col+1):
                for k in range(len(nodes[row][col])):
                    Au = ((self.periods_before_t + col) * nodes[row][col][k][0] + self.St * (self.u**(col+1-row)) * (self.d**row)) / \
                            (self.periods_before_t + col + 1)
                    Ad = ((self.periods_before_t + col) * nodes[row][col][k][0] + self.St * (self.u**(col-row)) * (self.d**(row+1))) / \
                            (self.periods_before_t + col + 1)
                    # search up and down node
                    if search_way == "sequential":
                        Cu = sequential_search(nodes[row][col+1], Au)
                        Cd = sequential_search(nodes[row+1][col+1], Ad)
                    elif search_way == "binary":
                        Cu = binary_search(nodes[row][col+1], Au)
                        Cd = binary_search(nodes[row+1][col+1], Ad)
                    elif search_way == "linear":
                        Cu = linear_interpolation(nodes[row][col+1], Au, self.M, logarithmically=logly)
                        Cd = linear_interpolation(nodes[row+1][col+1], Ad, self.M, logarithmically=logly)
                    else:
                        raise ValueError("The search way doesn't exist !!!")

                    nodes[row][col][k][1] = np.exp(-self.r*self.dt) * (self.p*Cu + (1-self.p)*Cd)

        return nodes[0][0][0][1]

    # for American calls: the only diff with euro calls is that considers the "early exercise"
    def backward_induction_usa(self, nodes, search_way, logly):    
        # calculate the payoffs for the last column
        for row in range(self.n+1):
            nodes[row][-1] = np.array(nodes[row][-1])
            nodes[row][-1][:, 1] = np.where(nodes[row][-1][:, 0] > self.K, 
                                            nodes[row][-1][:, 0] - self.K, 0)
        # backward induction
        for col in range(self.n-1, -1, -1):
            for row in range(col+1):
                for k in range(len(nodes[row][col])):
                    Au = ((self.periods_before_t + col) * nodes[row][col][k][0] + self.St * (self.u**(col+1-row)) * (self.d**row)) / \
                            (self.periods_before_t + col + 1)
                    Ad = ((self.periods_before_t + col) * nodes[row][col][k][0] + self.St * (self.u**(col-row)) * (self.d**(row+1))) / \
                            (self.periods_before_t + col + 1)
                    # search up node
                    if search_way == "sequential":
                        Cu = sequential_search(nodes[row][col+1], Au)
                        Cd = sequential_search(nodes[row+1][col+1], Ad)
                    elif search_way == "binary":
                        Cu = binary_search(nodes[row][col+1], Au)
                        Cd = binary_search(nodes[row+1][col+1], Ad)
                    elif search_way == "linear":
                        Cu = linear_interpolation(nodes[row][col+1], Au, self.M, logarithmically=logly)
                        Cd = linear_interpolation(nodes[row+1][col+1], Ad, self.M, logarithmically=logly)
                    else:
                        raise ValueError("The search way doesn't exist !!!")
                    # payoff func
                    factor = np.exp(-self.r*self.dt) * (self.p*Cu + (1-self.p)*Cd)
                    early_exercise = nodes[row][col][k][0] - self.K
                    nodes[row][col][k][1] = max(early_exercise, factor)

        return nodes[0][0][0][1]

    def main(self, compare_convergence_rates, compare_search_ways):
        # bonus1
        # compare the convergence rates of different "placement method(linearly or logarithmically)"
        if compare_convergence_rates:
            linear_nodes = self.create_nodes_linearly_cut()
            logly_nodes = self.create_nodes_logarithmically_cut()

            calls_euro_linear = self.backward_induction_euro(linear_nodes, "linear", False)
            calls_usa_linear = self.backward_induction_usa(linear_nodes, "linear", False)

            calls_euro_logly = self.backward_induction_euro(logly_nodes, "linear", True)
            calls_usa_logly = self.backward_induction_usa(logly_nodes, "linear", True)

            return calls_euro_linear, calls_usa_linear, calls_euro_logly, calls_usa_logly
        
        nodes = self.create_nodes_linearly_cut()
        # bonus2
        # compare the speed of different search algorithm(sequential, binary, linear interpolation)
        if compare_search_ways:
            results = {}
            for way in ["sequential", "binary", "linear"]:
                start_time = datetime.datetime.now()
                calls_euro = self.backward_induction_euro(nodes, way, False)
                calls_usa = self.backward_induction_usa(nodes, way, False)
                end_time = datetime.datetime.now()
                results[way] = [calls_euro, calls_usa, end_time-start_time]
            return results
        
        # basic requirement
        calls_euro = self.backward_induction_euro(nodes, "linear", False)
        calls_usa = self.backward_induction_usa(nodes, "linear", False)
        return calls_euro, calls_usa
