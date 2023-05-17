import numpy as np
from scipy.stats import norm
from hw2_models import BlackScholesModel, Monte_Carlo_Simulation, \
    BinomialTree2D, BinomialTree1D, Combinatorics


if __name__ == "__main__":
    s0 = float(input())
    k = float(input())
    r = float(input())
    q = float(input())
    sigma = float(input())
    big_t = float(input())
    nums_sim = int(input())
    nums_rep = int(input())
    n = int(input())

    bs = BlackScholesModel(s0, r, q, k, sigma, big_t)
    calls_BS, puts_BS = bs.calculate_options()
    calls_BS = round(calls_BS, 4)
    puts_BS = round(puts_BS, 4)

    mc = Monte_Carlo_Simulation(s0, r, q, sigma, big_t, k, nums_sim, nums_rep)
    calls_CI, puts_CI = mc.simulation()
    calls_CI = [round(i, 4) for i in calls_CI]
    puts_CI = [round(i, 4) for i in puts_CI]

    bt_2D = BinomialTree2D(s0, r, q, k, sigma, big_t, n)
    calls_euro_2D, puts_euro_2D, calls_ame_2D, puts_ame_2D = bt_2D.backward_induction()
    calls_euro_2D = round(calls_euro_2D, 4)
    puts_euro_2D = round(puts_euro_2D, 4)
    calls_ame_2D = round(calls_ame_2D, 4)
    puts_ame_2D = round(puts_ame_2D, 4)

    bt_1D = BinomialTree1D(s0, r, q, k, sigma, big_t, n)
    calls_euro_1D, puts_euro_1D, calls_ame_1D, puts_ame_1D = bt_1D.backward_induction()
    calls_euro_1D = round(calls_euro_1D, 4)
    puts_euro_1D = round(puts_euro_1D, 4)
    calls_ame_1D = round(calls_ame_1D, 4)
    puts_ame_1D = round(puts_ame_1D, 4)

    comb = Combinatorics(s0, r, q, k, sigma, big_t, n)
    calls_comb, puts_comb = comb.calculate()
    calls_comb = round(calls_comb, 4)
    puts_comb = round(puts_comb, 4)

    print(f"\nBy Black Scholes Formula:\n(1)calls = {calls_BS}\n(2)puts = {puts_BS}")
    print(f"\nBy Monte Carlo Simulation:\n(1)calls = {calls_CI}\n(2)puts = {puts_CI}")
    print(f"\nBy 2D Binomial Tree:\n(1)European calls = {calls_euro_2D}\n(2)European puts = {puts_euro_2D}\n(3)American calls = {calls_ame_2D}\n(4)American puts = {puts_ame_2D}")
    print(f"\nBy 1D Binomial Tree:\n(1)European calls = {calls_euro_1D}\n(2)European puts = {puts_euro_1D}\n(3)American calls = {calls_ame_1D}\n(4)American puts = {puts_ame_1D}")
    print(f"\nBy Combinational Method:\n(1)European calls = {calls_comb}\n(2)European puts = {puts_comb}\n")

# 45
# 50
# 0.8
# 0.05
# 0.4
# 0.5
# 10000
# 20
# 500
