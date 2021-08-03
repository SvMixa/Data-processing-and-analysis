import numpy as np
import matplotlib.pyplot as plt
import json
from scipy import integrate

from opt import gauss_newton, lm

data = np.genfromtxt('jla_mub.txt',delimiter=' ')
col_z = data [:, 0]
col_mu = data [:, 1]

def integral (z, teta): # Считаем интеграл
    i = lambda x: 1/np.sqrt((1-teta)*(1+x)**3 + teta)
    return np.array(list([integrate.quad(i, 0, _) for _ in z]))

def dependens(z, H0, teta): # Вычесляем функцию
    i = lambda x: 1/np.sqrt((1-teta)*(1+x)**3 + teta)
    I = np.array(list([integrate.quad(i, 0, _) for _ in z]))
    d = 3*10**11/H0 * (1 + z) * I[:,0]
    mu = 5 * np.log10(d) -5
    return(mu)

def j(x, H0, teta): # Находим якобиан нашей функции
    jac = np.empty((len(x), 2), dtype=np.float)
    jac[:, 0] = -5/H0/np.log(10)
    i = lambda x: ((x+1)**3-1)/(2*(teta - (teta-1)*(x+1)**3)**(3/2))    
    int_2 = np.array(list([integrate.quad(i, 0, _) for _ in x]))[:, 0]
    jac[:, 1] = 5/ integral(x, teta)[:, 0] * int_2 /np.log(10)
    return jac

GN = gauss_newton(col_mu, lambda _x: dependens(col_z, *_x) , lambda _: j(col_z, *_), 
	x0=(50, 0.5))

LM = lm(col_mu, lambda _x: dependens(col_z, *_x) , lambda _: j(col_z, *_), 
	x0=(50, 0.5))

plt.figure(figsize=(15, 10))
plt.rcParams.update({'font.size': 16})
plt.plot(col_z, dependens(col_z, *LM.x), label='Метод Левенберга—Марквардта', color='yellowgreen', linewidth=4)
plt.plot(col_z, dependens(col_z, *GN.x), linestyle='dashed', label='Метод Гаусса-Ньютона', linewidth=4, color = (0.1, 0.2, 0.9, 0.5))
plt.scatter(col_z, col_mu, color='indianred', s=69, label='Данные')
plt.legend()
plt.title('Зависимость модуля расстояния до сверхновых от красного смещения')
plt.xlabel('Красное смещение, z')
plt.ylabel('Модуль расстояния μ, пк')
plt.savefig('mu-z.png')

plt.figure(figsize=(15, 10))
plt.rcParams.update({'font.size': 16})
plt.plot(range(1, GN.nfev + 1), GN.cost, linestyle=':', label='Метод Гаусса-Ньютона')
plt.plot(range(1, LM.nfev + 1), LM.cost, label='Метод Левенберга—Марквардта')
plt.legend()
plt.title('Зависимость функции потерь от иетераций')
plt.xlabel('Итерации')
plt.ylabel('Функция потерь')
plt.savefig('cost.png')

file = {}
file["Gauss-Newton"] = ({"H0": int(GN.x[0]), "Omega": float('{:.2f}'.format(GN.x[1])), "nfev": GN.nfev})
file["Levenberg-Marquardt"] = ({"H0": int(LM.x[0]), "Omega": float('{:.2f}'.format(LM.x[1])), "nfev": LM.nfev})
json_file = json.dumps(file, indent=2) 
with open("parameters.json", "w") as outfile: 
    outfile.write(json_file)
