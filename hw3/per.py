#!/usr/bin/env python3


from mixfit import em_double_cluster
import astropy.units as u
from astropy.coordinates import SkyCoord
from astroquery.vizier import Vizier
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as sps
from matplotlib.patches import Ellipse
import json


def to_ra(x, dec_):
    return x/np.cos(dec_/180 * np.pi) + ra.mean()

def circle(sigma, cx, cy):
    angle = np.linspace( 0 , 2 * np.pi , 150 ) 
    x = np.sqrt(sigma[0,0]) * np.cos( angle ) - cx
    y = np.sqrt(sigma[1,1]) * np.sin( angle ) - cy
    return x, y

if __name__ == "__main__":
    center_coord = SkyCoord('02h21m00s +57d07m42s')
    vizier = Vizier(
        columns=['RAJ2000', 'DEJ2000', 'pmRA', 'pmDE'],
        column_filters={'BPmag': '<16', 'pmRA': '!=', 'pmDE': '!='}, # число больше — звёзд больше
        row_limit=10000
    )
    stars = vizier.query_region(
        center_coord,
        width=1.0 * u.deg,
        height=1.0 * u.deg,
        catalog=['I/350'], # Gaia EDR3
    )[0]
    
    ra = stars['RAJ2000']._data  # прямое восхождение, аналог долготы
    dec = stars['DEJ2000']._data  # склонение, аналог широты
    x1 = (ra - ra.mean()) * np.cos(dec / 180 * np.pi)
    x2 = dec
    v1 = stars['pmRA']._data
    v2 = stars['pmDE']._data
    
    tau10 = tau20 =  0.3
    muv0 = np.array([-0.75/2, -1.2/2])
    mu10 = np.array([-0.25, 58])
    mu20 = np.array([0.2, 57.1])
    sigma020 = [[5,0], [0, 3]] #[[np.cov(np.vstack((x1,x2)))[0,0], 0], [0, np.cov(np.vstack((x1,x2)))[1,1]]]
    sigmax20 = [[0.023,0], [0, 0.3]] #np.cov(np.vstack((v1,v2)))#
    sigmav20 = [[2,0], [0, 4]] #np.cov(np.vstack((v1,v2)))#
    
    data = np.array(list(zip(x1, x2, v1, v2)))
    answer = em_double_cluster(data, tau10, tau20, muv0, mu10, mu20, sigma020,  sigmax20, sigmav20)
    tau1, tau2 = answer[:2]
    muv = answer[2]
    mu1 = answer[3]
    mu2 = answer[4]
    
    #print('Tau1, Tau2 = ', answer[:2])
    #print('muv = ', answer[2])
    #print("mu1 = ", answer[3][0], answer[3][1])
    #print("mu2 = ", answer[4][0],  answer[4][1])
    #print(answer)


    plt.figure()
    ax = plt.gca()
    #
    ellipse1 = Ellipse(xy=(answer[3][0], answer[3][1]), width=np.sqrt(answer[6][0,0]), height=np.sqrt(answer[6][1,1]), 
                            edgecolor='r', fc='None', lw=2)
    ellipse2 = Ellipse(xy=(answer[4][0], answer[4][1]), width=np.sqrt(answer[6][0,0]), height=np.sqrt(answer[6][1,1]), 
                            edgecolor='r', fc='None', lw=2)
    ax.add_patch(ellipse1)
    ax.add_patch(ellipse2)
    plt.scatter(x1, x2, s=0.3)
    plt.scatter([answer[3][0], answer[4][0]], [answer[3][1], answer[4][1]])
    plt.savefig('per.png')
            
    file = {"size ratio": tau1/tau2, 
         "motion": {"ra": muv[0], "dec": muv[1]}, 
         "cluster":[ 
         { "center": {"ra": mu1[0], "dec": mu1[1]}}, 
         { "center": {"ra": mu2[0], "dec": mu2[1]} } ] 
         } 
    with open("per.json", "w") as f: 
        f.write(json.dumps(file, indent = 1))