#!/usr/bin/env python3

import argparse
import numpy as np
import matplotlib.pyplot as plt 
from scipy.io import netcdf
import json
import os

try:
    import geopy
except ImportError:
    os.system('pip3 install geopy')
    import geopy

from geopy.geocoders import Nominatim
geolocator = Nominatim(user_agent="SkittBot")

parser = argparse.ArgumentParser()
parser.add_argument('data', metavar='dat', help='coordinates or name', nargs="*")
if __name__ == "__main__":
    args = parser.parse_args()
    if len(args.data) == 1:
        loc = geolocator.geocode(*args.data)
        cord = float(loc.longitude), float(loc.latitude)
        #print(*args.data, cord)
    else:
        #print(args.data)
        cord = float(args.data[0]), float(args.data[1])
        #print(cord)
    
    
#cord = 37.66, 55.77
#cord = args.longitude, args.latitude
# Создаем заготовку под выходной файл с результатами
file = {}
file['coordinates'] = [(cord[0]), (cord[1])]
# Выбираем нужные нам месяца
data = netcdf.netcdf_file('MSR-2.nc', 'r', mmap=False)
jan = data.variables['time'].data[0::12] # January
jul = data.variables['time'].data[6::12] # July
year = data.variables['time'].data # all time
month_delta = 108 # с какого номера начинаем отсчитывать месяцы. Пригодится для подписи временной оси в графике.

jan_index = np.searchsorted(data.variables['time'].data, jan)
jul_index = np.searchsorted(data.variables['time'].data, jul)
lon_index = np.searchsorted(data.variables['longitude'].data, cord[1])
lat_index = np.searchsorted(data.variables['latitude'].data, cord[0])
# Собираем данные по Январю
about_jan = data.variables['Average_O3_column'].data[jan_index, lat_index, lon_index]
file['jan'] = ({
    'min': float('{:.1f}'.format(min(about_jan))),
    'max': float('{:.1f}'.format(max(about_jan))),
    'mean': float('{:.1f}'.format(np.mean(about_jan))),
})
# Собираем данные по Июлю
about_jul = data.variables['Average_O3_column'].data[jul_index, lat_index, lon_index]
file['jul'] = ({
    'min': float('{:.1f}'.format(min(about_jul))),
    'max': float('{:.1f}'.format(max(about_jul))),
    'mean': float('{:.1f}'.format(np.mean(about_jul))),
})
# Собираем данные по всему временному отрезку
about_year = data.variables['Average_O3_column'].data[:, lat_index, lon_index]
file['all'] = ({
    'min': float('{:.1f}'.format(min(about_year))),
    'max': float('{:.1f}'.format(max(about_year))),
    'mean': float('{:.1f}'.format(np.mean(about_year))),
})
# Записываем данные в файл "ozon.json"
json_file = json.dumps(file, indent = 2) 
with open("ozon.json", "w") as outfile: 
    outfile.write(json_file)

# Функция переводит количество месяцев в год (нужно для построения графика)
def month_to_year(m, month_delta=108):
    return (m -  month_delta) / 12 + 1979
# Строим график
plt.figure(figsize=(17,12))
plt.rcParams.update({'font.size': 16})

plt.plot(month_to_year(year), about_year, label='Зависимость для всего доступного интервала')
plt.scatter(month_to_year(jan), about_jan, label='Зависимость для январей', s=79, color='indianred')
plt.plot(month_to_year(jan), about_jan, color='indianred', linestyle='dashed')
plt.scatter(month_to_year(jul), about_jul, label='Зависимость для июлей', s=79, color='yellowgreen')
plt.plot(month_to_year(jul), about_jul, color='yellowgreen', linestyle='dashed')

plt.legend()
plt.title('Содержание озона в атмосфере', fontsize=20, pad = 30)
plt.xlabel('Года', labelpad=15)
plt.ylabel('Содержание озона в атмосфере, единицы Добсона', labelpad=20)
plt.savefig('ozon.png')
data.close()