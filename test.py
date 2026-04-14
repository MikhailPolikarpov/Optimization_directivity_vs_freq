import Global_optimizer as glo

if __name__ == "__main__":
    for seed in [90]:
        thres = [19]
        widths = [30]
        for i in range(len(thres)):
            optp = glo.OptimizationParameters()
            optp.n = 3
            optp.bounds_alpha = [(40, 0), (-30.0, 0), (30.0, 0)]
            optp.bounds_beta = [(0.05, 7.0)] + [(0.05, 4.0)]*2
            optp.eps = 1e-3
            optp.limit = 200
            optp.sigma_thres = thres[i] - 0.5
            optp.segment_width_percent = widths[i]
            de_p = glo.DifferentialEvolutionParameters()
            de_p.maxiter = 200
            de_p.popsize = 16
            de_p.seed = seed
            optimizer = glo.DifferentialEvolutionOptimizer(optp, de_p, result_path=f"results/test/width{widths[i]}de2_{thres[i]}_seed_{seed}.json", detailed=False)
            res = optimizer.optimize()
