import numpy as np 
import matplotlib.pyplot as plt
from T_method import LayeredStructure
from My_plotter import Style, Plotter
from scipy.optimize import differential_evolution
import multiprocessing as mp
        

st = Style()

ndf = 150 # кол-во точек по df
min_df = -0.5
max_df = 0.5
df_width = (max_df - min_df)*100 # ширина по df в процентах 
points_per_m_percent = lambda m: int(ndf * m / df_width) # кол-во точек на m процентов по df
df = np.linspace(min_df, max_df, ndf) # относительный сдвиг частоты

def sigma(x, thres=0, k=3):
    return 1/(1+np.exp(-k*(x - thres)))

def penalty(x, thres=0, k=3, c=3):
    return c/k * np.log(1 + np.exp(-k*(x - thres)))

m_target = points_per_m_percent(25)  # 10% of df width


def objective_function(params):
    n = len(params) // 2
    alpha = np.array(params[:n])
    beta = np.array(params[n:])
    structure = LayeredStructure(alpha, beta=beta)
    directivity = 10*np.log10(structure.directivity(df))
    target_f = -np.inf
    for i in range(0, ndf-m_target):
        segment_i = directivity[i:i+m_target]
        target_f_i = np.sum(sigma(segment_i, thres=18, k=3) - penalty(segment_i, thres=18, k=3, c=0.1))
        if target_f_i > target_f:
            target_f = target_f_i
    return -target_f  # We want to maximize target_f, so we minimize the negative of it

bounds = [
    (0,  20.0), (-20.0, 0), (0, 20.0), (-20.0, 0), (0, 20.0),  # alpha
    (0.1,  7.0), (0.1, 1.0), (0.1, 1.0), (0.1, 1.0), (0.1, 1.0),  # beta
]


alpha0 = np.array([3, -1.2, 2.9, -1.2, 1])*1.4
beta0 = np.array([2.8, 0.75, 0.65, 0.6, 0.5])

initial_params = np.concatenate([alpha0, beta0])

#print(objective_function(initial_params)/ndf*df_width)  # Check initial score

# класс отслеживания состояния
class StateTracker:
    def __init__(self):
        self.iteration = 0

    def callback(self, intermediate_result):
        self.iteration += 1
        current_target_f = -intermediate_result.fun 
        # - .x : лучшее решение на данный момент
        # - .fun : значение функции для лучшего решения
        # - .population : вся популяция (доступно с версии 1.12)
        # - .population_energies : значения функции для всей популяции
        print(f"Iteration {self.iteration}: best target_f = {current_target_f:.4f}")
        print(f"🔹 Сходимость: {intermediate_result.convergence:.4f}")

if __name__ == "__main__":
    state_tracker = StateTracker()
    res = differential_evolution(
        objective_function,
        bounds=bounds,
        strategy="best1bin",
        popsize=2,            # 10 * dim = 100 особей
        maxiter=5,            # можно увеличить позже
        workers=-1,            # использовать все ядра
        updating="deferred",   # обязательно для параллелизма
        polish=False,           # polishing сделаем потом Powell
        seed=4,
        callback=state_tracker.callback
    )
    best_params = res.x
    best_score = -res.fun

    print("Best value:", best_score)
    print("Success:", res.success)
    print("Message:", res.message)
    print("Best score (sum of sigmoids):", best_score/ndf*df_width)
    print("Best params:", best_params)

    alph_opt = np.array(best_params[:5])
    beta_opt = np.array(best_params[5:])
    alph_0 = np.array([3, -1.2, 2.9, -1.2, 1])*1.4
    beta_0 = np.array([2.8, 0.75, 0.65, 0.6, 0.5])
    structure_opt = LayeredStructure(alph_opt, beta=beta_opt)
    structure_0 = LayeredStructure(alph_0, beta_0)
    dir_prev = 10*np.log10(structure_0.directivity(df))
    dir1 = 10*np.log10(structure_opt.directivity(df))
    fig, ax = plt.subplots()
    pl = Plotter(ax, st)
    pl.plot(df*100, dir1, label='struct_0')
    pl.plot(df*100, dir_prev, label='struct_opt')
    pl.set_xlabel('df/f %')
    pl.set_ylabel('Directivity (dB)')
    pl.set_title('Directivity vs Frequency Offset')
    pl.set_ylim((0, 25))
    pl.finalize()
    ax.axhline(18, color='gray', linestyle='--', alpha=0.5)
    plt.show()