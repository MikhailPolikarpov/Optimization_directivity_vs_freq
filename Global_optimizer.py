from dataclasses import dataclass, field
from pathlib import Path
import numpy as np 
import matplotlib.pyplot as plt
from T_method import LayeredStructure
from My_plotter import Style, Plotter
from scipy.optimize import differential_evolution
import json
import time
from datetime import timedelta

def format_time(seconds):
    """Форматирует секунды в читаемый вид"""
    return str(timedelta(seconds=int(seconds)))

def my_np_save(path, **kwargs):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **kwargs)

def my_json_save(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=4)

def my_json_load(path):
    with open(path, 'r', encoding='utf-8') as file:
        return json.load(file)

@dataclass 
class OptimizationParameters:
    ndf: int = 150 # кол-во точек по df
    min_df: float = -50 # левая граница по df в процентах   
    max_df: float = 50  # правая граница по df в процентах
    segment_width_percent: float = 25.0 # целевая ширина по df в процентах
    sigma_thres: float = 18.0 # целевая КНД в дБ
    sigma_k: float = 3.0 # крутизна функции штрафа и сигмоиды
    penalty_c: float = 0.1 # коэффициент штрафа
    bounds_alpha: list = field(default_factory=lambda: [(0,  20.0), (-20.0, 0), (0, 20.0), (-20.0, 0), (0, 20.0)]) # границы для alpha
    bounds_beta: list = field(default_factory=lambda: [(0.1,  7.0), (0.1, 1.0), (0.1, 1.0), (0.1, 1.0), (0.1, 1.0)])  # границы для beta

    def __post_init__(self):
        self.bounds = self.bounds_alpha + self.bounds_beta
        self.df = np.linspace(self.min_df/100, self.max_df/100, self.ndf) # относительный сдвиг частоты
        self.segment_points = int(self.ndf * self.segment_width_percent / (self.max_df - self.min_df)) # кол-во точек в целевом сегменте
        self.df_center_segment = np.linspace(-self.segment_width_percent/200, self.segment_width_percent/200, self.segment_points)# частотная сетка для центрального целевого сегмента
    
    @property
    def sigma_penalty(self):
        def sigma(x):
            return 1/(1+np.exp(-self.sigma_k*(x - self.sigma_thres)))
        
        def penalty(x):
            return self.penalty_c/self.sigma_k * np.log(1 + np.exp(-self.sigma_k*(x - self.sigma_thres)))
        
        def combined(x):
            return sigma(x) - penalty(x)
        
        return combined
    
    def show_sigma_penalty(self, range_min=-3, range_max=3):
        st = Style()
        fig, ax = plt.subplots()
        pl = Plotter(ax, st)
        x = np.linspace(range_min+self.sigma_thres, range_max+self.sigma_thres, 300)
        y = self.sigma_penalty(x)
        pl.plot(x, y, label="Combined Sigma and Penalty")
        pl.set_xlabel('Directivity (dB)')
        pl.set_title('Combined Sigma and Penalty')
        pl.set_ylim((-2, 2))
        pl.finalize()
        ax.axvline(self.sigma_thres, color='gray', linestyle='--', alpha=0.5)
        ax.axvline(self.sigma_thres - 2, color='gray', linestyle='--', alpha=0.5)
        ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
        ax.axhline(1, color='gray', linestyle='--', alpha=0.5)
        plt.show()

    def objective_function_single_segment(self, params):
        n = len(params) // 2
        alpha = np.array(params[:n])
        beta = np.array(params[n:])
        structure = LayeredStructure(alpha, beta=beta)
        directivity = 10*np.log10(structure.directivity(self.df_center_segment))
        target_f = np.sum(self.sigma_penalty(directivity))
        return -target_f  # Для поиска максимума находим минимум отрицательной целевой функции
    
    def objective_function_multisegments(self, params):
        n = len(params) // 2
        alpha = np.array(params[:n])
        beta = np.array(params[n:])
        structure = LayeredStructure(alpha, beta=beta)
        directivity = 10*np.log10(structure.directivity(self.df))
        target_f = -np.inf
        for i in range(0, self.ndf-self.segment_points):
            segment_i = directivity[i:i+self.segment_points]
            target_f_i = np.sum(self.sigma_penalty(segment_i))
            if target_f_i > target_f:
                target_f = target_f_i
        return -target_f  # Для поиска максимума находим минимум отрицательной целевой функции
    


class StateTracker:
    def __init__(self):
        self.iteration = 0
        self.convergence_history = []
        self.alpha_history = []
        self.beta_history = []
        self.target_f_history = []

    def callback(self, intermediate_result):
        self.iteration += 1
        current_target_f = -intermediate_result.fun
        x = intermediate_result.x
        n = len(x) // 2
        alpha = list(x[:n])
        beta = list(x[n:])
        self.alpha_history.append(alpha)
        self.beta_history.append(beta)
        self.convergence_history.append(intermediate_result.convergence)
        self.target_f_history.append(current_target_f)
        print(f"Iteration {self.iteration}: best target_f = {current_target_f:.4f}")
        print(f"Convergence: {intermediate_result.convergence:.4f}")
        



@dataclass
class DifferentialEvolutionParameters:
    strategy: str = 'best1bin'
    maxiter: int = 2
    popsize: int = 2
    workers: int = -1
    updating: str = 'deferred'  # 'deferred' or 'immediate'
    polish: bool = False  # Отключаем полировку для более чистого теста
    seed: int = 4

class DifferentialEvolutionOptimizer:

    def __init__(self, optimization_params: OptimizationParameters, de_params: DifferentialEvolutionParameters, result_path=None, detailed=False, is_single_segment=True):
        self.optimization_params = optimization_params
        self.de_params = de_params
        if result_path is not None:
            self.result_path = Path(result_path)
        self.state_tracker = StateTracker()
        self.optimization_data = {}
        self.detailed = detailed
        self.single_segment = is_single_segment

    def optimize(self):
        if self.single_segment:
            objective_func = self.optimization_params.objective_function_single_segment
        else:
            objective_func = self.optimization_params.objective_function_multisegments
        time_start = time.time()
        res = differential_evolution(
            objective_func,
            bounds=self.optimization_params.bounds,
            strategy=self.de_params.strategy,
            popsize=self.de_params.popsize,
            maxiter=self.de_params.maxiter,
            workers=self.de_params.workers,
            updating=self.de_params.updating,
            polish=self.de_params.polish,
            seed=self.de_params.seed,
            callback=self.state_tracker.callback
        )
        best_params = res.x
        best_score = -res.fun
        time_end = time.time()
        simulation_time = time_end - time_start
        print(f"Optimization completed in {format_time(simulation_time)}")
        print("Best value:", best_score)
        print("Success:", res.success)
        print("Message:", res.message)
        print("Best score (sum of sigmoids):", best_score/self.optimization_params.ndf*(self.optimization_params.max_df - self.optimization_params.min_df))
        print("Best params:", best_params)

        if self.result_path is not None:
            n = len(best_params) // 2
            best_alpha = list(best_params[:n])
            best_beta = list(best_params[n:])
            success = res.success
            message = res.message
            best_score = best_score
            simulation_time_str = format_time(simulation_time)
            self.optimization_data["N"] = n
            self.optimization_data["best_alpha"] = best_alpha
            self.optimization_data["best_beta"] = best_beta
            self.optimization_data["best_score"] = best_score
            self.optimization_data["success"] = success
            self.optimization_data["message"] = message
            self.optimization_data["simulation_time"] = simulation_time_str
            self.optimization_data["ndf"] = self.optimization_params.ndf
            self.optimization_data["min_df"] = self.optimization_params.min_df
            self.optimization_data["max_df"] = self.optimization_params.max_df
            self.optimization_data["segment_width_percent"] = self.optimization_params.segment_width_percent
            self.optimization_data["segment_points"] = self.optimization_params.segment_points
            self.optimization_data["sigma_thres"] = self.optimization_params.sigma_thres
            self.optimization_data["sigma_k"] = self.optimization_params.sigma_k
            self.optimization_data["penalty_c"] = self.optimization_params.penalty_c
            self.optimization_data["bounds_alpha"] = self.optimization_params.bounds_alpha
            self.optimization_data["bounds_beta"] = self.optimization_params.bounds_beta
            self.optimization_data["is_single_segment"] = self.single_segment
            self.optimization_data["de_strategy"] = self.de_params.strategy
            self.optimization_data["de_maxiter"] = self.de_params.maxiter
            self.optimization_data["de_popsize"] = self.de_params.popsize
            self.optimization_data["de_workers"] = self.de_params.workers
            self.optimization_data["de_updating"] = self.de_params.updating
            self.optimization_data["de_seed"] = self.de_params.seed
            self.optimization_data["convergence_history"] = self.state_tracker.convergence_history  
            if self.detailed:
                self.optimization_data["alpha_history"] = self.state_tracker.alpha_history
                self.optimization_data["beta_history"] = self.state_tracker.beta_history
                self.optimization_data["target_f_history"] = self.state_tracker.target_f_history
            my_json_save(self.result_path, self.optimization_data)
        return res

