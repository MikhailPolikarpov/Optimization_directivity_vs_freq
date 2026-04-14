import Global_optimizer as glo

if __name__ == "__main__":
    for seed in [90]:
        thres = [45.0, 40.0, 35.0, 30.0, 25.0]
        widths = [30]
        for i in range(len(thres)):
            optp = glo.OptimizationParameters()
            optp.n = 3
            optp.bounds_alpha = [(-30, 0), (-30.0, 0), (-30.0, 0)]
            optp.bounds_beta = [(0.1, 7.0)] + [(0.1, 7.0)]*2
            optp.eps = 1e-3
            optp.limit = 200
            optp.sigma_thres = thres[i] - 0.5
            optp.segment_width_percent = widths[i]
            de_p = glo.DifferentialEvolutionParameters()
            de_p.maxiter = 200
            de_p.popsize = 32
            de_p.seed = seed
            optimizer = glo.DifferentialEvolutionOptimizer(optp, de_p, result_path=f"results/test/width{widths[i]}des_{thres[i]}_seed_{seed}.json", detailed=False)
            res = optimizer.optimize()
