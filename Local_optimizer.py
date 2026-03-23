from dataclasses import dataclass, field
from pathlib import Path
import numpy as np 
import matplotlib.pyplot as plt
from T_method import LayeredStructure
from My_plotter import Style, Plotter
from scipy.optimize import minimize
from functools import partial
from Global_optimizer import OptimizationParameters, StateTracker, my_json_load, my_json_save, format_time
import time

def postoptimization(path):
    data = my_json_load(path)
    ndf = data["ndf"]
    min_df = data["min_df"]
    max_df = data["max_df"]
    segment_width_percent = data["segment_width_percent"]
    sigma_thres = data["sigma_thres"]
    sigma_k = data["sigma_k"]
    penalty_c = data["penalty_c"]
    bounds_alpha = data["bounds_alpha"]
    bounds_beta = data["bounds_beta"]
    start_alpha = data["best_alpha"]
    start_beta = data["best_beta"]
    optp = OptimizationParameters(
        ndf=ndf,
        min_df=min_df,
        max_df=max_df,
        segment_width_percent=segment_width_percent,
        sigma_thres=sigma_thres,
        sigma_k=sigma_k,
        penalty_c=penalty_c,
        bounds_alpha=bounds_alpha,
        bounds_beta=bounds_beta,
        start_alpha=start_alpha,
        start_beta=start_beta
    )
    return optp

@dataclass
class LocalOptParameters:
    method: str = "Powell"
    maxfev: int = 100 # iterations of Powell outer loop
    maxiter: int = 5 # maxiter is not used for Powell, but we can set it to None for consistency
    ftol: float = 1e-6 # objective tolerance
    xtol: float = 1e-8 # parameter tolerance
    disp: bool = True # Note: Powell does not use 'disp' in the same way as other methods, but we can still set it for consistency)
    
    
class LocalOptimizer:
    def __init__(self, optp: OptimizationParameters, local_optp: LocalOptParameters, result_path=None, detailed: bool = False, is_single_segment: bool = True):
        self.optp = optp
        self.local_optp = local_optp
        self.result_path = result_path
        self.detailed = detailed
        self.is_single_segment = is_single_segment
        if result_path is not None:
            self.result_path = Path(result_path)
        self.state_tracker = StateTracker()
        self.optimization_data = {}

    def optimize(self):
        
        initial_params = (np.array(self.optp.start_params)-self.optp.b)/self.optp.k  # Use the start parameters from OptimizationParameters

        n = self.optp.n

        bounds = list([(0,1) for _ in range(2*n)])  # Use the bounds from OptimizationParameters



        if self.is_single_segment:
            objective_func = self.optp.objective_function_single_segment
        else:
            objective_func = self.optp.objective_function_multisegments
        time_start = time.time()
        res = minimize(
            objective_func,
            x0=initial_params,
            method=self.local_optp.method,
            bounds=bounds,
            callback=self.state_tracker.callback_local,
            options=dict(
                maxiter=self.local_optp.maxiter,
                ftol=self.local_optp.ftol,
                xtol=self.local_optp.xtol,
                disp=self.local_optp.disp,
                return_all=True
            )
        )
        simulation_time = time.time() - time_start

        best_params = res.x
        best_score = -res.fun  # because we minimized -J

        print("Success:", res.success)
        print("Message:", res.message)
        print("Best score (sum of sigmoids):", best_score/self.optp.ndf*(self.optp.max_df - self.optp.min_df))
        print("Best params:", best_params)

        n = len(best_params) // 2
        best_params = best_params*self.optp.k + self.optp.b  # Denormalize the parameters
        best_alpha = list(best_params[:n])
        best_beta = list(best_params[n:])
        success = res.success
        message = res.message
        best_score = best_score
        simulation_time_str = format_time(simulation_time)

        if self.result_path is not None:
            self.optimization_data["N"] = n
            self.optimization_data["best_alpha"] = best_alpha
            self.optimization_data["best_beta"] = best_beta
            self.optimization_data["best_score"] = best_score
            self.optimization_data["success"] = success
            self.optimization_data["message"] = message
            self.optimization_data["start_alpha"] = self.optp.start_alpha
            self.optimization_data["start_beta"] = self.optp.start_beta
            self.optimization_data["simulation_time"] = simulation_time_str
            self.optimization_data["ndf"] = self.optp.ndf
            self.optimization_data["min_df"] = self.optp.min_df
            self.optimization_data["max_df"] = self.optp.max_df
            self.optimization_data["segment_width_percent"] = self.optp.segment_width_percent
            self.optimization_data["segment_points"] = self.optp.segment_points
            self.optimization_data["sigma_thres"] = self.optp.sigma_thres
            self.optimization_data["sigma_k"] = self.optp.sigma_k
            self.optimization_data["penalty_c"] = self.optp.penalty_c
            self.optimization_data["bounds_alpha"] = self.optp.bounds_alpha
            self.optimization_data["bounds_beta"] = self.optp.bounds_beta
            self.optimization_data["is_single_segment"] = self.is_single_segment
            self.optimization_data["local_opt_method"] = self.local_optp.method
            self.optimization_data["local_opt_maxfev"] = self.local_optp.maxfev
            self.optimization_data["local_opt_maxiter"] = self.local_optp.maxiter
            self.optimization_data["local_opt_ftol"] = self.local_optp.ftol
            self.optimization_data["local_opt_disp"] = self.local_optp.disp
            my_json_save(self.result_path, self.optimization_data)
        return res
