import Global_optimizer as glo
import numpy as np

if __name__ == "__main__":
    for seed in [90, 21, 45, 18]:
        thres = [16.0]
        widths = [30]
        for i in range(len(thres)):
            optp = glo.OptimizationParameters()
            optp.mode = 'work'
            optp.n = 2
            optp.bounds_alpha = [(-30, 0), (-30, 0)]
            optp.bounds_beta = [(0.2, 7.0)] + [(0.2, 7.0)]
            optp.bounds_alpha_screen = [(0, 20.0), (0.0, 20.0)]
            optp.bounds_dipole_shift = [(0.1, np.pi)]
            optp.bounds_beta_d = [(0.0, 9.0)]
            optp.eps = 1e-3
            optp.limit = 200
            optp.sigma_thres = thres[i] - 0.5
            optp.segment_width_percent = widths[i]
            de_p = glo.DifferentialEvolutionParameters()
            de_p.maxiter = 200
            de_p.popsize = 16
            de_p.seed = seed
            optimizer = glo.DifferentialEvolutionOptimizer(optp, de_p, result_path=f"results/test/sas3{seed}.json", detailed=False, mode='work')
            res = optimizer.optimize()
