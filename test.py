import Global_optimizer as glo

if __name__ == "__main__":
    for seed in [2, 43, 90, 5]:
        optp = glo.OptimizationParameters()
        de_p = glo.DifferentialEvolutionParameters()
        de_p.maxiter = 80
        de_p.popsize = 15
        de_p.seed = seed
        optimizer = glo.DifferentialEvolutionOptimizer(optp, de_p, result_path=f"different_seeds/test_de_seed_{seed}.json", detailed=True, is_single_segment=True)
        res = optimizer.optimize()
