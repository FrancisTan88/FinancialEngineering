import numpy as np
from scipy.stats import norm
from hw1_monte_carlo import Monte_Carlo_Simulation


N = norm.cdf


def calculate_d(s0, r, q, k, sigma, big_t, real_world=False):
    if not real_world:
        d = (np.log(s0/k) + big_t * (r - q - (sigma**2) / 2)) / \
            (sigma * (big_t**0.5))
    else:
        d = (np.log(s0/k) + big_t * (r - q + (sigma**2) / 2)) / \
            (sigma * (big_t**0.5))
    return d


if __name__ == "__main__":
    s0 = float(input())
    r = float(input())
    q = float(input())
    sigma = float(input())
    big_t = float(input())
    k1 = float(input())
    k2 = float(input())
    k3 = float(input())
    k4 = float(input())

    # first
    d1_rw = calculate_d(s0, r, q, k1, sigma, big_t, True)
    d2_rw = calculate_d(s0, r, q, k2, sigma, big_t, True)
    d1 = calculate_d(s0, r, q, k1, sigma, big_t)
    d2 = calculate_d(s0, r, q, k2, sigma, big_t)
    # sec
    d3 = calculate_d(s0, r, q, k3, sigma, big_t)
    # third
    d4 = calculate_d(s0, r, q, k4, sigma, big_t)
    d3_rw = calculate_d(s0, r, q, k3, sigma, big_t, True)
    d4_rw = calculate_d(s0, r, q, k4, sigma, big_t, True)

    # calculate the closed form of calls price by Martingale Pricing Method
    first_term = s0 * np.exp(-q*big_t) * (N(d1_rw) - N(d2_rw)) - \
        k1 * np.exp(-r * big_t) * (N(d1) - N(d2))
    second_term = (k2 - k1) * np.exp(-r * big_t) * (N(d2) - N(d3))
    third_term = (k2-k1)/(k4-k3) * (k4 * np.exp(-r * big_t) *
                                    (N(d3) - N(d4)) - s0 * np.exp(-q*big_t) * (N(d3_rw) - N(d4_rw)))
    ans = first_term + second_term + third_term
    print(f"\nBy Martingale Pricing Method:\nCalls = {round(ans, 4)}\n")

    # calculate the calls price by Monte Carlo Simulation
    sampling_times = 10000
    repetitions = 20
    mcs = Monte_Carlo_Simulation(
        s0, r, q, sigma, big_t, k1, k2, k3, k4, sampling_times, repetitions)
    result = mcs.simulation()
    print(
        f"By Monte Carlo Simulation:\nMean value = {np.round(result[0], 4)}\nStandard Deviation = {np.round(result[1], 4)}\nConfidence Interval = {np.round(result[2], 4)}\n")
