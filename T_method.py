

import numpy as np
from scipy.integrate import quad_vec

MU_0 = 4e-7 * np.pi
EPS_0 = 8.8541878188e-12
ETA_0 = np.sqrt(MU_0/EPS_0)
C = 1/np.sqrt(MU_0*EPS_0)

class FreeSpaceLayer: # beta=k_z*l на основной частоте при нормальном падении, df - относительная расстройка
    def __init__(self, beta, theta, df):
        self.theta = theta[None, :, None]
        self.df = df[None, None, :]
        self.beta_real = beta*(1+self.df)*np.cos(self.theta)
    def Tmatrix(self):
        t_mat = np.array([[np.cos(self.beta_real), np.sin(self.beta_real)],
                        [-np.sin(self.beta_real), np.cos(self.beta_real)]], dtype=float)
        return t_mat.transpose(2,3,4,0,1) # [phi][theta][df][*][*] (последние 2 измерения - ОБРАТНАЯ Т-матрица)
    
class ImpSheetLayer: # alpha - ETA_0/X_s на основной частоте при нормальном падении, df - относительная расстройка, polarization ('TE' или 'TM'), dispersion ('ind' или 'cap')
    def __init__(self, alpha, theta, df, polarization, dispersion = 'ind'):
        self.theta = theta[None, :, None]
        self.df = df[None, None, :]
        self.like_df = np.ones_like(self.df)
        self.like_theta = np.ones_like(self.theta)
        if dispersion == 'ind':
            self.alpha_real = alpha/(1+self.df)
        elif dispersion == 'cap':
            self.alpha_real = alpha*(1+self.df)
        if polarization == 'TE':
            self.alpha_real = self.alpha_real/np.cos(self.theta)
        elif polarization == 'TM':
            self.alpha_real = self.alpha_real*np.cos(self.theta)
    def Tmatrix(self):
        t_mat = np.array([[1*self.like_df*self.like_theta, 0*self.like_df*self.like_theta],
                        [self.alpha_real, 1*self.like_df*self.like_theta]], dtype=float)
        return t_mat.transpose(2,3,4,0,1) # [phi][theta][df][*][*] (последние 2 измерения - ОБРАТНАЯ Т-матрица)
    
class LayeredStructure:

    def __init__(self, alpha, beta='first_approx'):
        self.N = len(alpha)
        self.alpha = alpha
        if type(beta) == str and beta == 'first_approx':
            self.beta = self.first_approx_max_directivity()
        else:
            self.beta = beta
        self.dispersion = ['ind']*self.N
        for i in range(self.N):
            if self.alpha[i] < 0:
                self.dispersion[i] = 'cap'

    def directivity(self, df): # df - относительная расстройка (np.array) 
        def denom_func(theta):
            theta = np.array([theta])
            T_shift = FreeSpaceLayer(np.pi/2, theta, df).Tmatrix()
            vec_0 = np.array([0,1])[None, None, None, :,None]
            p_s = ((T_shift@vec_0).transpose(0,1,4,3,2)[0, 0, 0, 0, :])**2
            vec_TM = np.array([0,1])[None, None, None, :,None]
            vec_TE = np.array([0,1])[None, None, None, :,None]
            for i in range(self.N):
                vec_TM = FreeSpaceLayer(self.beta[i], theta, df).Tmatrix()@vec_TM
                vec_TM = ImpSheetLayer(self.alpha[i], theta, df, 'TM', self.dispersion[i]).Tmatrix()@vec_TM
                vec_TE = FreeSpaceLayer(self.beta[i], theta, df).Tmatrix()@vec_TE
                vec_TE = ImpSheetLayer(self.alpha[i], theta, df, 'TE', self.dispersion[i]).Tmatrix()@vec_TE
            vec_TM = vec_TM.transpose(0,1,4,3,2)[0, 0, 0, :, :]
            vec_TE = vec_TE.transpose(0,1,4,3,2)[0, 0, 0, :, :]
            p_TM = (vec_TM[0, :]**2 + vec_TM[1, :]**2)
            p_TE = (vec_TE[0, :]**2 + vec_TE[1, :]**2)
            p_xi_TM = p_s/p_TM
            p_xi_TE = p_s/p_TE
            return  np.cos(theta)**2 *p_xi_TM + p_xi_TE
        p_norm = denom_func(0)/2
        integral, eps = quad_vec(lambda theta: denom_func(theta)*np.sin(theta), 0, np.pi/2*0.99, epsrel=1e-3)
        return 4*p_norm/integral # выводит массив directivity для каждой df
    
    def radiation_pattern(self, phi, theta, df, mode='normalized'):
        def p_on_direction_array(phi, theta, df):
            T_shift = FreeSpaceLayer(np.pi/2, theta, df).Tmatrix() # [phi][theta][df][*][*]
            vec_0 = np.array([0,1])[None, None, None, :,None]
            p_s = ((T_shift@vec_0).transpose(4,3,2,0,1)[0, 0, :, :, :])**2 # [df][phi][theta]
            vec_TM = np.array([0,1])[None, None, None, :,None]
            vec_TE = np.array([0,1])[None, None, None, :,None]
            for i in range(self.N):
                vec_TM = FreeSpaceLayer(self.beta[i], theta, df).Tmatrix()@vec_TM
                vec_TM = ImpSheetLayer(self.alpha[i], theta, df, 'TM', self.dispersion[i]).Tmatrix()@vec_TM
                vec_TE = FreeSpaceLayer(self.beta[i], theta, df).Tmatrix()@vec_TE
                vec_TE = ImpSheetLayer(self.alpha[i], theta, df, 'TE', self.dispersion[i]).Tmatrix()@vec_TE
            vec_TM = vec_TM.transpose(4, 3, 2, 0, 1)[0, :, :, :, :] 
            vec_TE = vec_TE.transpose(4, 3, 2, 0, 1)[0, :, :, :, :]
            # удалена фиктивная матричная ось, теперь [компоненты вектора][df][phi][theta]
            p_TM = (vec_TM[0, :, :, :]**2 + vec_TM[1, :, :, :]**2) # [df][phi][theta]
            p_TE = (vec_TE[0, :, :, :]**2 + vec_TE[1, :, :, :]**2)
            p_xi_TM = p_s/p_TM
            p_xi_TE = p_s/p_TE
            phi = phi[None, :, None]
            p = np.cos(theta)**2*np.cos(phi)**2*p_xi_TM + np.sin(phi)**2*p_xi_TE
            return p
        p_total = p_on_direction_array(phi, theta, df)/p_on_direction_array(np.array([0]), np.array([0]), df)
        if mode == 'normalized':
            return p_total
        elif mode == 'absolute':
            return p_total*(self.directivity(df)[:, None, None])
        
    def first_approx_max_directivity(self):
        alpha = self.alpha
        beta = np.arctan((alpha-np.sign(alpha)*np.sqrt(alpha**2+4))/2)+np.pi
        psi = np.arctan(np.tan(beta)/(alpha*np.tan(beta)+1))
        beta1 = np.zeros_like(beta)
        for i in range(1, len(beta)):
            beta1[i] = beta[i] - psi[i-1]
            if beta1[i] > np.pi:
                beta1[i] -= np.pi
        beta1[0] = beta[0]
        return beta1
    
    def max_directivity_and_bandwidth(self, df):
        directivity = self.directivity(df)
        max_dir_i = np.argmax(directivity)
        max_dir = directivity[max_dir_i]
        i = max_dir_i
        while i > 0 and (max_dir - directivity[i]) < 3:
            i -= 1
        left_df = df[i]
        i = max_dir_i
        while i < len(df)-1 and (max_dir - directivity[i]) < 3:
            i += 1
        right_df = df[i]
        return max_dir, (right_df - left_df)*100

