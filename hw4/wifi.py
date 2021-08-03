#!/usr/bin/env python3


import argparse
import json
import numpy as np
from scipy.signal import find_peaks

parser = argparse.ArgumentParser()
parser.add_argument('input',  metavar='FILENAME', type=str)

if __name__ == "__main__":
    args = parser.parse_args()

    data = np.loadtxt(args.input)
    
    code = np.array([1,1,1,-1,-1,-1,1,-1,-1,1,-1], dtype=np.int8)
    code = np.repeat(code, 5)
    
    # convolve our data and Barker's code
    dconv = np.convolve(data, code[::-1], mode='full')
    
    # find coordinates of peaks
    peaks_max, _ = find_peaks(dconv, distance=10, height=30)
    peaks_min, _ = find_peaks(-dconv, distance=10, height=30)
    peaks = np.sort(np.append(peaks_max, peaks_min))
    
    # transform peaks to list of bits
    bts = np.array(list(map(int, (dconv[peaks] > 0))))
    
    # transform bits to bytes and then to char
    answer = ''.join(list(map( lambda i: chr(int(''.join(map(str, i)), 2)), np.array_split(bts, len(bts)//8))))
    
    # save results
    file = {}
    file["message"] = (answer)
    json_file = json.dumps(file) 
    with open("wifi.json", "w") as outfile: 
        outfile.write(json_file)
        