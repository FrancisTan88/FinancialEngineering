import numpy as np

# input data
def input_data():
    k = float(input())
    r = float(input())
    big_t = float(input())
    nums_sims = int(input())
    nums_reps = int(input())
    nums_assets = int(input())
    S0s = [float(s) for s in input().split(",")]  # n
    dividends = [float(div) for div in input().split(",")]  # n
    sigmas = [float(sigma) for sigma in input().split(",")]  # n
    ros = [[float(ro) for ro in input().split(",")] for _ in range(nums_assets)]  # n*n
    return k, r, big_t, nums_sims, nums_reps, nums_assets, S0s, dividends, sigmas, ros

def cal_covmatrix(ros, sigmas, big_t):
    n = len(sigmas)
    cov_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            cov_matrix[i][j] = ros[i][j] * sigmas[i] * sigmas[j] * big_t
    return cov_matrix

# calculate maxtrix A
def cholesky(cov_matrix):
    n = len(cov_matrix)
    matrix_A = np.zeros((n, n))
    # calculate first row
    matrix_A[0][0] = np.sqrt(cov_matrix[0][0])
    for j in range(1, n):
        matrix_A[0][j] = cov_matrix[0][j] / matrix_A[0][0]
    # for step2 ~ step4 in p.5-4
    for i in range(1, n):
        matrix_A[i][i] = np.sqrt(cov_matrix[i][i] - sum([(matrix_A[k][i])**2 for k in range(i)]))
        for j in range(i+1, n):
            matrix_A[i][j] = 1 / matrix_A[i][i] * \
                (cov_matrix[i][j] - sum([matrix_A[k][i] * matrix_A[k][j] for k in range(i)]))
    return matrix_A

# -> (nums_sims * nums_assets)
def cal_stock_prices(rs, S0s, r, dividends, sigmas, big_t):
    mus = []  # n
    for i in range(len(S0s)):
        mus.append(np.log(S0s[i]) + (r - dividends[i] - sigmas[i]**2 / 2)*big_t)
    # stock pices: mu + variance
    for j in range(len(mus)):
        rs[:, j] = rs[:, j] + mus[j]
    rs = np.exp(rs)
    return rs

def cal_payoff(sps, k, r, big_t):
    payoffs = []
    for i in range(len(sps)):
        payoffs.append(np.max(sps[i, :]) - k)
    payoffs = np.array(payoffs)
    payoffs = np.where(payoffs < 0, 0, payoffs)
    return np.exp(-r*big_t) * np.mean(payoffs)


# calculate the random variables (r) matrix -> (nums_sims * nums_assets)
def cal_random_vars(matrix_A, zs):
    return np.dot(zs, matrix_A)

# sampling random vars using "variance reduction"
def variance_reduction(matrix_A, zs):
    zs_new = np.matrix(np.append(zs[:len(zs)//2], -zs[:len(zs)//2], axis=0))
    zs_new /= np.std(zs_new)
    rs = np.dot(zs_new, matrix_A)
    return rs

def inverse_cholesky(matrix_A, zs):
    n = nums_assets
    zs_new = np.array(np.append(zs[:len(zs)//2], -zs[:len(zs)//2], axis=0))
    zs_new /= np.std(zs_new)
    zs_cof = np.corrcoef([zs_new[:, ass] for ass in range(n)])
    cho = cholesky(zs_cof)
    rs = np.dot(np.matrix(zs_new), np.dot(np.linalg.inv(cho), matrix_A))
    return rs
    
    

if __name__ == "__main__":
    k, r, big_t, nums_sims, nums_reps, nums_assets, S0s, dividends, sigmas, ros = \
        input_data()
    
    # cholesky
    cov_matrix = cal_covmatrix(ros, sigmas, big_t)
    matrix_A = cholesky(cov_matrix)

    # calculate stock prices
    arr_basics, arr_bonus1, arr_bonus2 = [], [], []
    for rep in range(nums_reps):
        rand_seeds = np.matrix(np.random.normal(0, 1, size=(nums_sims, nums_assets)))

        # basics
        matrix_r = cal_random_vars(matrix_A, rand_seeds)
        sps = cal_stock_prices(matrix_r, S0s, r, dividends, sigmas, big_t)
        payoff = cal_payoff(sps, k, r, big_t)
        arr_basics.append(payoff)

        # bonus 1
        matrix_r_varred = variance_reduction(matrix_A, rand_seeds)
        sps_red = cal_stock_prices(matrix_r_varred, S0s, r, dividends, sigmas, big_t)
        payoff_red = cal_payoff(sps_red, k, r, big_t)
        arr_bonus1.append(payoff_red)

        # bonus 2
        matrix_r_inv = inverse_cholesky(matrix_A, rand_seeds)
        sps_inv = cal_stock_prices(matrix_r_inv, S0s, r, dividends, sigmas, big_t)
        payoff_inv = cal_payoff(sps_inv, k, r, big_t)
        arr_bonus2.append(payoff_inv)
        

    mean_basic, mean_bonus1, mean_bonus2 = np.mean(arr_basics), \
            np.mean(arr_bonus1), np.mean(arr_bonus2)
    std_basic, std_bonus1, std_bonus2 = np.std(arr_basics), \
        np.std(arr_bonus1), np.std(arr_bonus2)
        
    print(f"\nFor basic requirements:")
    print(f"Option Price = {round(mean_basic, 4)}")
    print(f"Confidence Interval = [{round(mean_basic-2*std_basic, 4)}, {round(mean_basic+2*std_basic, 4)}]\n")

    print(f"For bonus 1:")
    print(f"Option Price = {round(mean_bonus1, 4)}")
    print(f"Confidence Interval = [{round(mean_bonus1-2*std_bonus1, 4)}, {round(mean_bonus1+2*std_bonus1, 4)}]\n")

    print(f"For bonus 2:")
    print(f"Option Price = {round(mean_bonus2, 4)}")
    print(f"Confidence Interval = [{round(mean_bonus2-2*std_bonus2, 4)}, {round(mean_bonus2+2*std_bonus2, 4)}]")
