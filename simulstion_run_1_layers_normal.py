import Global_optimizer as glo
import numpy as np


def run_optimization(bounds_alpha, bounds_beta, max_summ_beta, seed, result_path):
    if __name__ == "__main__":
        thres = 16
        width = 30
        optp = glo.OptimizationParameters()
        optp.penalty_c = 0.0
        optp.mode = 'work'
        optp.n = 1
        optp.bounds_alpha = bounds_alpha
        optp.bounds_beta = bounds_beta
        optp.bounds_alpha_screen = [(0, 20.0), (-20, 0.0)]
        optp.bounds_dipole_shift = [(0.3, np.pi)]
        optp.bounds_beta_d = [(0.0, 9.0)]
        optp.eps = 1e-3
        optp.limit = 200
        optp.sigma_thres = thres - 0.5
        optp.segment_width_percent = width
        optp.segment_shift_percent = -5
        optp.max_summ_beta = max_summ_beta
        de_p = glo.DifferentialEvolutionParameters()
        de_p.maxiter = 200
        de_p.popsize = 50
        de_p.seed = seed
        optimizer = glo.DifferentialEvolutionOptimizer(optp, de_p, result_path=result_path, detailed=False, mode='work')
        res = optimizer.optimize()
    return None

seeds = [90]
bounds_items = [["c", [(-20.0, 0)]], ["l", [(0, 20.0)]]]
bounds_beta = [(0.2, 3.2)]
l=40
max_summ_beta = np.pi*l/40
for seed in seeds:
    for i1 in bounds_items:
            mode_str = i1[0]
            bounds_alpha = i1[1]
            result_path = f"results/1_sheets_l_{l}/{mode_str}_seed_{seed}.json"
            run_optimization(bounds_alpha, bounds_beta, max_summ_beta, seed, result_path)