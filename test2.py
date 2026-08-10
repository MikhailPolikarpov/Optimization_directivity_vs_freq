import Global_optimizer as glo

if __name__ == "__main__":
    for seed in [4, 58, 1000, 222]:
        thres = [16.0]
        widths = [30]
        for i in range(len(thres)):
            optp = glo.OptimizationParameters()
            n = 2
            optp.n = n
            optp.bounds_alpha = [(-30, 30)]*n
            optp.bounds_beta = [(0.1, 3.0)]+[(0.1, 1.0)]*(n-1)
            optp.eps = 1e-3
            optp.limit = 200
            optp.sigma_thres = thres[i] - 0.5
            optp.segment_width_percent = widths[i]
            de_p = glo.DifferentialEvolutionParameters()
            de_p.maxiter = 300
            de_p.popsize = 80
            de_p.seed = seed
            optimizer = glo.DifferentialEvolutionOptimizer(optp, de_p, result_path=f"project/N={optp.n}_seed_{seed}_limitated1.json", detailed=False)
            res = optimizer.optimize()
