from multiprocessing.pool import Pool
import os
from pathlib import Path
import traceback
import numpy as np 
import matplotlib.pyplot as plt
from T_method import LayeredStructure
from My_plotter import Style, Plotter
from Global_optimizer import OptimizationParameters, StateTracker, my_json_load, my_json_save, format_time
import Local_optimizer as loc


def run_seed(seed):
    try:
        path = f"different_seeds/test_de_seed_{seed}.json"
        path_local = f"different_seeds/test_local_post_de_seed_{seed}.json"
        data_local = my_json_load(path_local)
        data = my_json_load(path)
        optp = loc.postoptimization(path)
        optp.bounds_beta = [(0.1, 3.0) for _ in range(5)]
        optp.start_alpha = data_local["best_alpha"]
        optp.start_beta = data_local["best_beta"]
        optp.limit = 1000
        optp.eps = 1e-5
        local_optp = loc.LocalOptParameters()
        local_optp.maxiter = 100

        print(f"[PID {os.getpid()}] seed={seed}, start score={data['best_score']}")

        optimizer = loc.LocalOptimizer(
            optp,
            local_optp,
            result_path=f"different_seeds/test_local_post_de_seed_{seed}.json",
            is_single_segment=True
        )
        res = optimizer.optimize()

        return seed, res, None

    except Exception:
        return seed, None, traceback.format_exc()
    
if __name__ == "__main__":
    seeds = [2, 5, 43, 90]

    n_workers = min(os.cpu_count(), len(seeds))

    with Pool(processes=n_workers, maxtasksperchild=10) as pool:
        for seed, res, err in pool.imap_unordered(run_seed, seeds):
            if err is None:
                print(f"seed={seed} done")
            else:
                print(f"seed={seed} failed:\n{err}")

# if __name__ == "__main__":
#     for seed in [2, 5]:
#         path = f"different_seeds/test_de_seed_{seed}.json"
#         data = my_json_load(path)
#         optp = loc.postoptimization(path)
#         local_optp = loc.LocalOptParameters()
#         local_optp.maxiter = 3
#         print("Starting score from DE:", data["best_score"])
#         optimizer = loc.LocalOptimizer(optp, local_optp, result_path=f"different_seeds/test_local_post_de_seed_{seed}.json", is_single_segment=True)
#         res = optimizer.optimize()
        