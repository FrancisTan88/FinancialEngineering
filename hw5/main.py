from models import MonteCarlo, BinomialTree
import numpy as np
import datetime
import matplotlib.pyplot as plt


def read_input():
    S_t = float(input())
    K = float(input())
    r = float(input())
    q = float(input())
    sigma = float(input())
    t = float(input())
    T_minus_t = float(input())
    M = int(input())
    n = int(input())
    S_ave_t = float(input())
    Sim_n = int(input())
    Rep_n = int(input())
    bonus1 = input()
    bonus2 = input()
    return S_t, K, r, q, sigma, t, T_minus_t, M, n, S_ave_t, Sim_n, Rep_n, bonus1, bonus2

def plot_curve(lst_M, euro_linear, usa_linear, euro_log, usa_log, t):
    plt.plot(lst_M, euro_linear, label ='euro_linear')
    plt.plot(lst_M, usa_linear, label ='usa_linear')
    plt.plot(lst_M, euro_log, label ='euro_log')
    plt.plot(lst_M, usa_log, label ='usa_log')

    plt.xlabel("M")
    plt.ylabel("calls")
    plt.legend()
    plt.title(f'Compare Different Convergence Rates(t={t})')
    plt.savefig(f"./convergence_rates_t_{t}.png")
    plt.show()

def main():
    S_t, K, r, q, sigma, t, T_minus_t, M, n, S_ave_t, Sim_n, Rep_n, bonus1, bonus2 = read_input()

    # Monte Carlo
    mc = MonteCarlo(S_t, K, r, q, sigma, t, T_minus_t, M, n, S_ave_t, Sim_n, Rep_n)
    mean, std = mc.main()
    print(f"\nMonte Carlo:\n(1)European calls = {round(mean, 4)}\n(2)Confidence Interval = [{round(mean-2*std, 4)}, {round(mean+2*std, 4)}]\n")
    
    # Binomial Tree
    # bonus1 only
    if bonus1 == "True" and bonus2 == "False":
        print("Bonus1")
        lst_M = []
        start_M, end_M = 50, 400
        calls_euro_linear, calls_usa_linear, calls_euro_logly, calls_usa_logly = [], [], [], []
        for m in range(start_M, end_M+1, 50):
            lst_M.append(m)
        for m in lst_M:
            print(f"Now the M == {m}")
            bt = BinomialTree(S_t, K, r, q, sigma, t, T_minus_t, m, n, S_ave_t, Sim_n, Rep_n)
            c_e_linear, c_u_linear, c_e_logly, c_u_logly = bt.main(compare_convergence_rates=True, compare_search_ways=False)
            calls_euro_linear.append(round(c_e_linear, 4))
            calls_usa_linear.append(round(c_u_linear, 4))
            calls_euro_logly.append(round(c_e_logly, 4))
            calls_usa_logly.append(round(c_u_logly, 4))
        plot_curve(lst_M, calls_euro_linear, calls_usa_linear, calls_euro_logly, calls_usa_logly)
    else:
        bt = BinomialTree(S_t, K, r, q, sigma, t, T_minus_t, M, n, S_ave_t, Sim_n, Rep_n)
        # bonus2 only
        if bonus2 == "True" and bonus1 == "False":
            print("Bonus2")
            results = bt.main(compare_convergence_rates=False, compare_search_ways=True)
            print("Binomial Tree:")
            for search_way, answer in results.items():
                print(f"By {search_way} --> European calls = {round(answer[0], 4)}, American calls = {round(answer[1], 4)}, Time cost = {answer[2]}")
            print("\n")
        # basic requirement only
        else:
            print("Basic Requirement")
            euro_calls, usa_calls = bt.main(compare_convergence_rates=False, compare_search_ways=False)
            print("Binomial Tree:")
            print(f"European calls = {round(euro_calls, 4)}, American calls = {round(usa_calls, 4)}")


if __name__ == "__main__":
    main()



