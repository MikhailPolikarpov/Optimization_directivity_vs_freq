import Global_optimizer as glo
import numpy as np

if __name__ == "__main__":
    for n in [1]:
        for seed in [90]:
            thres = [14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0]
            widths = np.array([45.0, 40.0, 35.0, 30.0, 27.0, 25.0, 23.0, 20.0, 18.0])*(1+(n-3)*0.3)
            for i in range(len(thres)):
                optp = glo.OptimizationParameters()
                optp.n = n
                optp.bounds_alpha = [(50.0, 0)] + [(30.0, 0)]*(n-1)
                optp.bounds_beta = [(0.1, 3.0)] + [(0.1, 3.0)]*(n-1)
                optp.eps = 1e-3
                optp.limit = 200
                optp.sigma_thres = thres[i] - 0.5
                optp.segment_width_percent = widths[i]
                de_p = glo.DifferentialEvolutionParameters()
                de_p.maxiter = 200
                de_p.popsize = 32
                de_p.seed = seed
                optimizer = glo.DifferentialEvolutionOptimizer(optp, de_p, result_path=f"kz/{n}sh/de_{thres[i]}_seed_{seed}.json", detailed=False)
                res = optimizer.optimize()
