from dataclasses import dataclass, field
from pathlib import Path
import numpy as np 
import matplotlib.pyplot as plt
from T_method import LayeredStructure
from My_plotter import Style, Plotter
from scipy.optimize import differential_evolution, NonlinearConstraint
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
    _n: int = 5 # кол-во слоев
    segment_width_percent: float = 25.0 # целевая ширина по df в процентах
    segment_points: int = 35 # кол-во точек в целевом сегменте
    sigma_thres: float = 18.0 # целевая КНД в дБ
    sigma_k: float = 3.0 # крутизна функции штрафа и сигмоиды
    penalty_c: float = 0.1 # коэффициент штрафа
    eps = 1e-3 # точность для интегрирования в функции directivity
    limit = 200 # предел для интегрирования в функции directivity
    bounds_alpha: list = field(default_factory=lambda: [(0,  20.0)]*5) # границы для alpha
    bounds_beta: list = field(default_factory=lambda: [(0.1,  3.0)]*5)  # границы для beta
    start_alpha: list = field(default_factory=lambda: [10.0]*5) # начальные значения для alpha
    start_beta: list = field(default_factory=lambda: [1.0]*5) # начальные значения для beta
    mode: str = 'diploma'
    alpha_l = 1.e16 # индуктивная составляющая проводимости экрана
    alpha_c = 1.e16 # емкостная составляющая проводимости экрана
    dipole_shift = np.pi/2 # расстояние между диполем и экраном
    beta_d = 0.0 # расстояние между двумя диполями в случае, когда источник состоит из двух диполей под 45 градусов
    bounds_alpha_screen: list = field(default_factory=lambda: [(0,  20.0), (0.0, 20.0)]) # границы для alpha экрана
    bounds_dipole_shift: list = field(default_factory=lambda: [(0.1, np.pi)]) # границы для dipole_shift
    bounds_beta_d: list = field(default_factory=lambda: [(0.0,  11.0)]) # границы для beta_d
    max_summ_beta: float = np.pi
    @property
    def n(self):
        return self._n

    @n.setter
    def n(self, value):
        self._n = value
        self.bounds_alpha = [(0,  20.0)]*value
        self.bounds_beta = [(0.1,  3.0)]*value
        self.start_alpha = [10.0]*value
        self.start_beta = [1.0]*value

    @property
    def bounds(self):
        if self.mode == 'diploma':
            return self.bounds_alpha + self.bounds_beta
        if self.mode == 'work':
            return self.bounds_alpha + self.bounds_beta + self.bounds_alpha_screen + self.bounds_dipole_shift + self.bounds_beta_d
    
    @property
    def start_params(self):
        if self.mode == 'diploma':
            return self.start_alpha + self.start_beta
        if self.mode == 'work':
            return self.start_alpha + self.start_beta + [self.alpha_l, self.alpha_c] + [self.dipole_shift] + [self.beta_d]

    @property
    def k(self):
        return np.array([b[1] - b[0] for b in self.bounds])
    
    @property
    def b(self):
        return np.array([b[0] for b in self.bounds])
    
    @property
    def df_center_segment(self):
        return np.linspace(-self.segment_width_percent/200, self.segment_width_percent/200, self.segment_points)

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

    def objective_function_single_segment(self, norm_params):        
        n = len(norm_params) // 2
        param = np.array(norm_params)*self.k + self.b
        alpha = param[:n]
        beta = param[n:]
        structure = LayeredStructure(alpha, beta=beta)
        directivity = 10*np.log10(structure.directivity(self.df_center_segment, eps=self.eps, limit=self.limit))
        target_f = np.sum(self.sigma_penalty(directivity))
        return -target_f  # Для поиска максимума находим минимум отрицательной целевой функции
    
    def objective_function_two_sources(self, norm_params):
        n = (len(norm_params) - 4) // 2
        param = np.array(norm_params)*self.k + self.b
        alpha = param[:n]
        beta = param[n:n+n]
        alpha_l = param[n+n]
        alpha_c = param[n+n+1]
        dipole_shift = param[n+n+2]
        beta_d = param[n+n+3]
        structure = LayeredStructure(alpha, beta=beta, alpha_l=alpha_l, alpha_c=alpha_c, dipole_shift=dipole_shift)
        directivity = 10*np.log10(structure.directivity_two_sources_diagonal(self.df_center_segment, beta_d=beta_d, eps=self.eps, limit=self.limit))
        target_f = np.sum(self.sigma_penalty(directivity))
        return -target_f  # Для поиска максимума находим минимум отрицательной целевой функции
    
    def constraint_sum_beta(self, norm_params):
        n = (len(norm_params) - 4) // 2
        param = np.array(norm_params)*self.k + self.b
        beta = param[n:n+n]
        return self.max_summ_beta - np.sum(beta)
    
    def constraint_dipole_before_first_sheet(self, norm_params):
        n = (len(norm_params) - 4) // 2
        param = np.array(norm_params)*self.k + self.b
        dipole_shift = param[n+n+2]
        beta = param[n:n+n]
        return  beta[0] - dipole_shift  # dipole_shift должен быть меньше beta[0]

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

    def callback_local(self, intermediate_result):
        self.iteration += 1
        resdict = intermediate_result.copy()
        current_target_f = -resdict['fun']
        x = resdict['x']
        n = len(x) // 2
        alpha = list(x[:n])
        beta = list(x[n:])
        self.alpha_history.append(alpha)
        self.beta_history.append(beta)
        print(f"Iteration {self.iteration}, target_f = {current_target_f:.4f}")
        



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

    def __init__(self, optp: OptimizationParameters, de_params: DifferentialEvolutionParameters, result_path=None, detailed=False, mode='diploma'):
        self.optp = optp
        self.de_params = de_params
        if result_path is not None:
            self.result_path = Path(result_path)
        self.state_tracker = StateTracker()
        self.optimization_data = {}
        self.detailed = detailed
        self.mode = mode
    def optimize(self):
        if self.mode == 'diploma':
            objective_func = self.optp.objective_function_single_segment
            constraints = None
        elif self.mode == 'work':
            objective_func = self.optp.objective_function_two_sources
            constraint_sum_beta = NonlinearConstraint(self.optp.constraint_sum_beta, 0, np.inf)
            constraint_dipole_before_first_sheet = NonlinearConstraint(self.optp.constraint_dipole_before_first_sheet, 0, np.inf)
            constraints = (constraint_sum_beta, constraint_dipole_before_first_sheet)
        else:
            raise ValueError("Invalid mode. Choose 'diploma' or 'work'.")
        time_start = time.time()
        res = differential_evolution(
            objective_func,
            bounds=[(0, 1.0)]*(len(self.optp.start_params)),
            strategy=self.de_params.strategy,
            popsize=self.de_params.popsize,
            maxiter=self.de_params.maxiter,
            workers=self.de_params.workers,
            updating=self.de_params.updating,
            polish=self.de_params.polish,
            seed=self.de_params.seed,
            callback=self.state_tracker.callback,
            constraints=constraints
        )
        best_normalized_params = res.x
        best_params = best_normalized_params*self.optp.k + self.optp.b
        best_score = -res.fun
        time_end = time.time()
        simulation_time = time_end - time_start
        print(f"Optimization completed in {format_time(simulation_time)}")
        print("Best value:", best_score)
        print("Success:", res.success)
        print("Message:", res.message)
        print("Best score (sum of sigmoids):", best_score/self.optp.segment_points*(self.optp.segment_width_percent))
        print("Best params:", best_params)

        if self.result_path is not None:
            n = self.optp.n
            best_alpha = list(best_params[:n])
            best_beta = list(best_params[n:n+n])
            best_alpha_l = best_params[n+n]
            best_alpha_c = best_params[n+n+1]
            best_dipole_shift = best_params[n+n+2]
            best_beta_d = best_params[n+n+3]
            success = res.success
            message = res.message
            best_score = best_score
            simulation_time_str = format_time(simulation_time)
            self.optimization_data["N"] = n
            self.optimization_data["best_alpha"] = best_alpha
            self.optimization_data["best_beta"] = best_beta
            self.optimization_data["best_alpha_l"] = best_alpha_l
            self.optimization_data["best_alpha_c"] = best_alpha_c
            self.optimization_data["best_dipole_shift"] = best_dipole_shift
            self.optimization_data["best_beta_d"] = best_beta_d
            self.optimization_data["best_score"] = best_score
            self.optimization_data["success"] = success
            self.optimization_data["message"] = message
            self.optimization_data["simulation_time"] = simulation_time_str
            self.optimization_data["segment_width_percent"] = self.optp.segment_width_percent
            self.optimization_data["segment_points"] = self.optp.segment_points
            self.optimization_data["sigma_thres"] = self.optp.sigma_thres
            self.optimization_data["sigma_k"] = self.optp.sigma_k
            self.optimization_data["penalty_c"] = self.optp.penalty_c
            self.optimization_data["bounds_alpha"] = self.optp.bounds_alpha
            self.optimization_data["bounds_beta"] = self.optp.bounds_beta
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

