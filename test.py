import Global_optimizer as glo

if __name__ == "__main__":
    for seed in [90]:
        thres = [15.0, 17.0, 19.0, 21.0, 23.0]
        widths = [45.0, 30.0, 25.0, 23.0, 20.0]
        for i in range(len(thres)):
            optp = glo.OptimizationParameters()
            optp.n = 3
            optp.bounds_alpha = [(35.0, 0), (30.0, 0), (30.0, 0)]
            optp.bounds_beta = [(0.1, 3.0)] + [(0.1, 3.0)]*2
            optp.eps = 1e-3
            optp.limit = 200
            optp.sigma_thres = thres[i] - 0.5
            optp.segment_width_percent = widths[i]
            de_p = glo.DifferentialEvolutionParameters()
            de_p.maxiter = 80
            de_p.popsize = 16
            de_p.seed = seed
            optimizer = glo.DifferentialEvolutionOptimizer(optp, de_p, result_path=f"kz/de_{thres[i]}_seed_{seed}.json", detailed=False)
            res = optimizer.optimize()
