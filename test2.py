import Global_optimizer as glo
import numpy as np

if __name__ == "__main__":
    for seed in [90]:
        thres = [17.5]
        widths = [30]
        for i in range(len(thres)):
            optp = glo.OptimizationParameters()
            optp.penalty_c = 0.0
            optp.mode = 'work'
            optp.n = 3
            optp.bounds_alpha = [(20.0, 0.0), (20.0, 0.0), (20.0, 0.0)]
            optp.bounds_beta = [(0.2, 7.0), (0.3, 1.0), (0.3, 1.0)]
            optp.bounds_alpha_screen = [(0, 20.0), (-20, 0.0)]
            optp.bounds_dipole_shift = [(0.3, np.pi)]
            optp.bounds_beta_d = [(0.0, 9.0)]
            optp.eps = 1e-3
            optp.limit = 200
            optp.sigma_thres = thres[i] - 0.5
            optp.segment_width_percent = widths[i]
            optp.segment_shift_percent = -5
            optp.max_summ_beta = np.pi*8/4
            de_p = glo.DifferentialEvolutionParameters()
            de_p.maxiter = 200
            de_p.popsize = 16
            de_p.seed = seed
            optimizer = glo.DifferentialEvolutionOptimizer(optp, de_p, result_path=f"results/final/cl10_wide.json", detailed=False, mode='work')
            res = optimizer.optimize()
