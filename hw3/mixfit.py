#!/usr/bin/env python3
import numpy as np
import scipy.stats as sps
from scipy import optimize

def norm_distribution(x, mu, sigma):
    return (1/(np.sqrt(2*np.pi)*sigma) * np.exp(-0.5*(x-mu)**2 / sigma**2))
    
def l_norm_distribution(x, tau, mu1, sigma1, mu2, sigma2):
        L = tau * norm_distribution(x, mu1, sigma1) + (1-tau) * norm_distribution(x, mu2, sigma2)
        return - np.sum(np.log(L))
    
def max_likelihood(x, tau, mu1, sigma1, mu2, sigma2, rtol=1e-3):
    init = np.array([tau, mu1, sigma1, mu2, sigma2])
    fun = lambda par: l_norm_distribution(x, *par)
    minimum2  = optimize.least_squares(fun, init, xtol=rtol, bounds=([0, -np.inf, 0, -np.inf, 0], [1, np.inf, np.inf, np.inf, np.inf]))
    return minimum2


def em_double_gauss(x, tau, mu1, sigma1, mu2, sigma2, rtol=1e-3):
    def tau_z(x, tau, mu1, mu2, sigma1, sigma2):
        T1 = tau * norm_distribution(x, mu1, sigma1)
        T2 = (1 - tau) * norm_distribution(x, mu2, sigma2)
        T_norm = T1 + T2
        T1 = np.divide(T1, T_norm, out=np.full_like(T1, 0.5), where=T_norm!=0)
        T2 = np.divide(T2, T_norm, out=np.full_like(T2, 0.5), where=T_norm!=0)
        return T1, T2
    
    def one_step_EM(x, tau, mu1, sigma1, mu2, sigma2):
        T1, T2 = tau_z(x, tau, mu1, sigma1, mu2, sigma2)
        tau_new = np.sum(T1)/np.sum(T1+T2)
        mu1_new = np.sum(T1*x)/np.sum(T1)
        mu2_new = np.sum(T2*x)/np.sum(T2)
        sigma1_new = np.sqrt(abs(np.sum(T1 * (x-mu1_new)**2)/np.sum(T1)))
        sigma2_new = np.sqrt(abs(np.sum(T2 * (x-mu2_new)**2)/np.sum(T2)))
        return np.array([tau_new, mu1_new, mu2_new, sigma1_new, sigma2_new])
    
    new = one_step_EM(x, tau, mu1, sigma1, mu2, sigma2)
    old = np.array([tau, mu1, sigma1, mu2, sigma2])
    while np.linalg.norm(new - old) > rtol:
        old = new
        new = one_step_EM(x, *old)
    return new


def em_double_cluster(x, tau1, tau2, muv, mu1, mu2, sigma02, sigmax2, sigmav2, rtol=1e-5):
    def dnorm_distrebutaion(x, mu, sigma):
        dist = []
        for i in x:
            dist.append(sps.multivariate_normal(mean=mu, cov=sigma).pdf(i))
        return np.array(dist)
    
    def norm2(x, mu, sigma):
        ans = []
        for i in x:
            vec = (i - mu)
            ans.append(np.exp(-0.5 * vec.T @ np.linalg.inv(sigma) @ vec) / ((2 * np.pi)*np.sqrt(np.linalg.det(sigma))))
        return np.array(ans)
    
    def only_v(data):
        dv = []
        for i in data:
            dv.append(i[2:])
        return np.array(dv)
    
    def only_x(data):
        dx = []
        for i in data:
            dx.append(i[:2])
        return np.array(dx)

    def multiply(T, x):
        for i in range(len(x)):
            x[i] = x[i] * T[i]
        return x
    
    def Tall (x, tau1, tau2, muv, mu1, mu2, sigma02, sigmax2, sigmav2):
        T1 = tau1 * norm2(only_x(x), mu1, sigmax2) * norm2(only_v(x), muv, sigmav2)
        T2 = tau2 * norm2(only_x(x), mu2, sigmax2) * norm2(only_v(x), muv, sigmav2)
        T3 = (1 - tau1 - tau2) * norm2(only_v(x), [0, 0], sigma02)
        T_norm = T1 + T2 + T3
        T1 = np.divide(T1, T_norm, out=np.full_like(T1, 0.5), where=T_norm!=0)
        T2 = np.divide(T2, T_norm, out=np.full_like(T2, 0.5), where=T_norm!=0)
        T3 = np.divide(T3, T_norm, out=np.full_like(T3, 0.5), where=T_norm!=0)
        return T1, T2, T3
    
    
    def one_step_EM(x, tau1, tau2, muv, mu1, mu2, sigma02, sigmax2, sigmav2):
        T1, T2, T3 = Tall(x, tau1, tau2, muv, mu1, mu2, sigma02, sigmax2, sigmav2)
        tau1_new = np.sum(T1)/np.sum(T1+T2+T3)
        tau2_new = np.sum(T2)/np.sum(T1+T2+T3)
        mu1_new = (T1 @ x)/np.sum(T1)
        mu2_new = (T2 @ x)/np.sum(T2)
        mu1 = np.array(mu1_new)[:2]
        mu2 = np.array(mu2_new)[:2]
        muv = (mu1_new[2:] + mu2_new[2:])*0.5
        sigma1_new = (np.dot((x - mu1_new).T, multiply(T1, (x - mu1_new)))) / np.sum(T1)
        sigma2_new = (np.dot((x - mu2_new).T, multiply(T2, (x - mu2_new)))) / np.sum(T2)
        sigma3_new = (np.dot((only_v(x)).T, multiply(T3, only_v(x)))) / np.sum(T3)
        sigma02 = sigma3_new
        sigmax2 = 0.5 * (sigma1_new + sigma2_new)[0:2, 0:2]
        sigmav2 = 0.5 * (sigma1_new + sigma2_new)[2:, 2:]
        return np.array([tau1_new, tau2_new, muv, mu1, mu2, sigma02, sigmax2, sigmav2], dtype=object)   
    
    def leng(x):
        s = 0
        for i in x:
            s += i**2
        return(np.sqrt(s))
    
    def stop(new, old, rtol):
        return not np.allclose([new[0], new[1], leng(new[2]), leng(new[3]), leng(new[4])], [old[0], old[1], leng(old[2]), leng(old[3]), leng(old[4])], rtol = rtol)
    
    new = one_step_EM(x, tau1, tau2, muv, mu1, mu2, sigma02, sigmax2, sigmav2)
    old = np.array([tau1, tau2, muv, mu1, mu2, sigma02, sigmax2, sigmav2], dtype=object)
    while stop(new, old, rtol=1e0):
        old = new
        new = one_step_EM(x, *old)
    return new


if __name__ == "__main__":
    pass
