import Global_optimizer as glo

if __name__ == "__main__":
    optp = glo.OptimizationParameters()
    de_p = glo.DifferentialEvolutionParameters()
    de_p.maxiter = 160
    de_p.popsize = 40
    optimizer = glo.DifferentialEvolutionOptimizer(optp, de_p, result_path="kaka/test_de.json", detailed=True, is_single_segment=True)
    res = optimizer.optimize()
    print(optp.segment_points)