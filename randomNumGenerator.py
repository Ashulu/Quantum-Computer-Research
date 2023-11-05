import sys
import time
import logging
import functools
import math
import matplotlib.pyplot as plt
import copy
from scipy.special import gamma, gammaincc, exp1, hyp1f1
from scipy.fftpack import fft
from collections import OrderedDict
from itertools import product
import scipy.stats as stats
from scipy.stats import chi2
from scipy.special import erfc
import numpy as np


import dwavebinarycsp as dbc
from dwave.system import DWaveSampler, EmbeddingComposite

import pandas as pd
import numpy

from pyqubo import Constraint, Array
import neal
from dwave.system import LeapHybridSampler

#method for incomplete gamma function
def inc_gamma(a, x):
    return exp1(x) if a == 0 else gamma(a)*gammaincc(a, x)

# first test for randomness
def frequencyTest(dictionary):
    sum = 0
    for i in range(len(dictionary)):
        value = 2*dictionary['x['+str(i)+']'] -1
        sum = sum + value
    if sum < 0:
        sum = 0 - sum
    testStat = sum/(math.sqrt(len(best_sample.sample)))
    pVal = math.erf(testStat/(math.sqrt(2)))
    return pVal

# second test
def frequencyInBlock(dictionary):
    block_size = 128
    bin_data = ""
    for i in range(len(dictionary)):
        bin_data = bin_data + str(dictionary['x['+str(i)+']'])
    num_blocks = math.floor(len(bin_data) / block_size)
    block_start, block_end = 0, block_size
    # Keep track of the proportion of ones per block
    proportion_sum = 0.0
    for i in range(num_blocks):
        # Slice the binary string into a block
        block_data = bin_data[block_start:block_end]
        # Keep track of the number of ones
        ones_count = 0
        for char in block_data:
            if char == '1':
                ones_count += 1
        pi = ones_count / block_size
        proportion_sum += pow(pi - 0.5, 2.0)
        # Update the slice locations
        block_start += block_size
        block_end += block_size
    # Calculate the p-value
    chi_squared = 4.0 * block_size * proportion_sum
    p_val = inc_gamma(num_blocks / 2, chi_squared / 2)
    return p_val

# third test
def runsTest(dictionary):
    if frequencyTest(dictionary):
        return True
    blockData = ""
    for i in range(len(dictionary)):
        blockData = blockData + str(dictionary['x['+str(i)+']'])
    onesCount = 0
    for char in blockData:
        if char == '1':
            onesCount += 1
    pi = onesCount / len(dictionary)
    sum = 0
    for i in range(len(dictionary)):
        value = 2*dictionary['x['+str(i)+']'] -1
        sum = sum + value
    pVal = math.erf((abs(sum - 2*len(dictionary) * pi * (1-pi) ))/(2 * math.sqrt(len(dictionary)) * pi*(1-pi)))
    return pVal

#fourth test
def longestRunTest(dictionary):
    bin_data = ""
    for i in range(len(dictionary)):
        bin_data = bin_data + str(dictionary['x['+str(i)+']'])
    if len(bin_data) < 128:
        print("\t", "Not enough data to run test!")
        return -1.0
    elif len(bin_data) < 6272:
        k, m = 3, 8
        v_values = [1, 2, 3, 4]
        pik_values = [0.21484375, 0.3671875, 0.23046875, 0.1875]
    elif len(bin_data) < 75000:
        k, m = 5, 128
        v_values = [4, 5, 6, 7, 8, 9]
        pik_values = [0.1174035788, 0.242955959, 0.249363483, 0.17517706, 0.102701071, 0.112398847]
    else:
        k, m = 6, 10000
        v_values = [10, 11, 12, 13, 14, 15, 16]
        pik_values = [0.0882, 0.2092, 0.2483, 0.1933, 0.1208, 0.0675, 0.0727]

    # Work out the number of blocks, discard the remainder
    # pik = [0.2148, 0.3672, 0.2305, 0.1875]
    num_blocks = math.floor(len(bin_data) / m)
    frequencies = numpy.zeros(k + 1)
    block_start, block_end = 0, m
    for i in range(num_blocks):
        # Slice the binary string into a block
        block_data = bin_data[block_start:block_end]
        # Keep track of the number of ones
        max_run_count, run_count = 0, 0
        for j in range(0, m):
            if block_data[j] == '1':
                run_count += 1
                max_run_count = max(max_run_count, run_count)
            else:
                max_run_count = max(max_run_count, run_count)
                run_count = 0
        max_run_count = max(max_run_count, run_count)
        if max_run_count < v_values[0]:
            frequencies[0] += 1
        for j in range(k):
            if max_run_count == v_values[j]:
                frequencies[j] += 1
        if max_run_count > v_values[k - 1]:
            frequencies[k] += 1
        block_start += m
        block_end += m
    # print(frequencies)
    chi_squared = 0
    for i in range(len(frequencies)):
        chi_squared += (pow(frequencies[i] - (num_blocks * pik_values[i]), 2.0)) / (num_blocks * pik_values[i])
    pVal = inc_gamma(float(k / 2), float(chi_squared / 2))
    return pVal


#fifth test
def spectralTest(dictionary):
    bin_data = ""
    for i in range(len(dictionary)):
        bin_data = bin_data + str(dictionary['x['+str(i)+']'])
    n = len(bin_data)
    plus_minus_one = []
    for char in bin_data:
        if char == '0':
            plus_minus_one.append(-1)
        elif char == '1':
            plus_minus_one.append(1)
    # Product discrete fourier transform of plus minus one
    s = fft(plus_minus_one)
    modulus = numpy.abs(s[0:int(n/2)])
    tau = numpy.sqrt(numpy.log(1 / 0.05) * n)
    # Theoretical number of peaks
    count_n0 = 0.95 * (n / 2)
    # Count the number of actual peaks m > T
    count_n1 = len(numpy.where(modulus < tau)[0])
    # Calculate d and return the p value statistic
    d = (count_n1 - count_n0) / numpy.sqrt(n * 0.95 * 0.05 / 4)
    pVal = math.erf(abs(d) / numpy.sqrt(2))
    return pVal

# sixth test

def nonOverlap(dictionary, template):
    bin_data = ""
    for i in range(len(dictionary)):
        bin_data = bin_data + str(dictionary['x['+str(i)+']'])
    n = len(bin_data)

    B = template
    m = len(template)
    blockSize = m
    N = n // blockSize

    templateCount = []
    for i in range(N):
        block = bin_data[i * blockSize : (i + 1) * blockSize]
        count = block.count(B)
        templateCount.append(count)

    mu = (blockSize - m + 1) / (2**m)
    sigma2 = blockSize / (2**(2 * m))

    chi_squared = sum(((count - mu) ** 2) / sigma2 for count in templateCount)
    pVal = 1 - chi2.cdf(chi_squared, N -1)
    return pVal


# seventh test 

def overlappingTemplate(dictionary):
    pattern_size = 9
    block_size = 1032
    bin_data = ""
    for i in range(len(dictionary)):
        bin_data = bin_data + str(dictionary['x['+str(i)+']'])
    n = len(bin_data)
    pattern = ""
    for i in range(pattern_size):
        pattern += "1"
    num_blocks = math.floor(n / block_size)
    lambda_val = float(block_size - pattern_size + 1) / pow(2, pattern_size)
    eta = lambda_val / 2.0

    piks = [get_prob(i, eta) for i in range(5)]
    diff = float(numpy.array(piks).sum())
    piks.append(1.0 - diff)

    pattern_counts = numpy.zeros(6)
    for i in range(num_blocks):
        block_start = i * block_size
        block_end = block_start + block_size
        block_data = bin_data[block_start:block_end]
        # Count the number of pattern hits
        pattern_count = 0
        j = 0
        while j < block_size:
            sub_block = block_data[j:j + pattern_size]
            if sub_block == pattern:
                pattern_count += 1
            j += 1
        if pattern_count <= 4:
            pattern_counts[pattern_count] += 1
        else:
            pattern_counts[5] += 1

    chi_squared = 0.0
    for i in range(len(pattern_counts)):
        if(num_blocks * piks[i] == 0):
            continue
        else :
            chi_squared += pow(pattern_counts[i] - num_blocks * piks[i], 2.0) / (num_blocks * piks[i])
    return inc_gamma(5.0 / 2.0, chi_squared / 2.0)

def get_prob(u, x):
    out = 1.0 * numpy.exp(-x)
    if u != 0:
        out = 1.0 * x * numpy.exp(2 * -x) * (2 ** -u) * hyp1f1(u + 1, 2, x)
    return out

#eighth test
def linear_complexity(dictionary):
    block_size = 500
    bin_data = ""
    for i in range(len(dictionary)):
        bin_data = bin_data + str(dictionary['x['+str(i)+']'])
    dof = 6
    piks = [0.01047, 0.03125, 0.125, 0.5, 0.25, 0.0625, 0.020833]

    t2 = (block_size / 3.0 + 2.0 / 9) / 2 ** block_size
    mean = 0.5 * block_size + (1.0 / 36) * (9 + (-1) ** (block_size + 1)) - t2

    num_blocks = int(len(bin_data) / block_size)
    if num_blocks > 1:
        block_end = block_size
        block_start = 0
        blocks = []
        for i in range(num_blocks):
            blocks.append(bin_data[block_start:block_end])
            block_start += block_size
            block_end += block_size

        complexities = []
        for block in blocks:
            complexities.append(berlekamp_massey_algorithm(block))

        t = ([-1.0 * (((-1) ** block_size) * (chunk - mean) + 2.0 / 9) for chunk in complexities])
        vg = numpy.histogram(t, bins=[-9999999999, -2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 9999999999])[0][::-1]
        im = ([((vg[ii] - num_blocks * piks[ii]) ** 2) / (num_blocks * piks[ii]) for ii in range(7)])

        chi_squared = 0.0
        for i in range(len(piks)):
            chi_squared += im[i]
        p_val = inc_gamma(dof / 2.0, chi_squared / 2.0)
        return p_val
    else:
        return -1.0
    
def berlekamp_massey_algorithm(block_data):
    n = len(block_data)
    c = numpy.zeros(n)
    b = numpy.zeros(n)
    c[0], b[0] = 1, 1
    l, m, i = 0, -1, 0
    int_data = [int(el) for el in block_data]
    while i < n:
        v = int_data[(i - l):i]
        v = v[::-1]
        cc = c[1:l + 1]
        d = (int_data[i] + numpy.dot(v, cc)) % 2
        if d == 1:
            temp = copy.copy(c)
            p = numpy.zeros(n)
            for j in range(0, l):
                if b[j] == 1:
                    p[j + i - m] = 1
            c = (c + p) % 2
            if l <= 0.5 * i:
                l = i + 1 - l
                m = i
                b = temp
        i += 1
    return l

#ninth test
def serial(dictionary):
    pattern_length = 16
    method = "first"
    bin_data = ""
    for i in range(len(dictionary)):
        bin_data = bin_data + str(dictionary['x['+str(i)+']'])
    n = len(bin_data)
    # Add first m-1 bits to the end
    bin_data += bin_data[:pattern_length - 1:]

    # Get max length one patterns for m, m-1, m-2
    max_pattern = ''
    for i in range(pattern_length + 1):
        max_pattern += '1'

    # Keep track of each pattern's frequency (how often it appears)
    vobs_one = numpy.zeros(int(max_pattern[0:pattern_length:], 2) + 1)
    vobs_two = numpy.zeros(int(max_pattern[0:pattern_length-1:], 2) + 1)
    vobs_thr = numpy.zeros(int(max_pattern[0:pattern_length-2:], 2) + 1)

    for i in range(n):
        # Work out what pattern is observed
        vobs_one[int(bin_data[i:i + pattern_length:], 2)] += 1
        vobs_two[int(bin_data[i:i + pattern_length-1:], 2)] += 1
        vobs_thr[int(bin_data[i:i + pattern_length-2:], 2)] += 1

    vobs = [vobs_one, vobs_two, vobs_thr]
    sums = numpy.zeros(3)
    for i in range(3):
        for j in range(len(vobs[i])):
            sums[i] += pow(vobs[i][j], 2)
        sums[i] = (sums[i] * pow(2, pattern_length-i)/n) - n

    # Calculate the test statistics and p values
    del1 = sums[0] - sums[1]
    del2 = sums[0] - 2.0 * sums[1] + sums[2]
    p_val_one = inc_gamma(pow(2, pattern_length-1)/2, del1/2.0)
    p_val_two = inc_gamma(pow(2, pattern_length-2)/2, del2/2.0)

    # For checking the outputs
    if method == "first":
        return p_val_one
    else:
        # I am not sure if this is correct, but it makes sense to me.
        return min(p_val_one, p_val_two)


#tenth test
def entropy(dictionary):
    pattern_length = 10
    bin_data = ""
    for i in range(len(dictionary)):
        bin_data = bin_data + str(dictionary['x['+str(i)+']'])
    n = len(bin_data)
    # Add first m+1 bits to the end
    # NOTE: documentation says m-1 bits but that doesnt make sense, or work.
    bin_data += bin_data[:pattern_length + 1:]

    # Get max length one patterns for m, m-1, m-2
    max_pattern = ''
    for i in range(pattern_length + 2):
        max_pattern += '1'

    # Keep track of each pattern's frequency (how often it appears)
    vobs_one = numpy.zeros(int(max_pattern[0:pattern_length:], 2) + 1)
    vobs_two = numpy.zeros(int(max_pattern[0:pattern_length + 1:], 2) + 1)

    for i in range(n):
        # Work out what pattern is observed
        vobs_one[int(bin_data[i:i + pattern_length:], 2)] += 1
        vobs_two[int(bin_data[i:i + pattern_length + 1:], 2)] += 1

    # Calculate the test statistics and p values
    vobs = [vobs_one, vobs_two]
    sums = numpy.zeros(2)
    for i in range(2):
        for j in range(len(vobs[i])):
            if vobs[i][j] > 0:
                sums[i] += vobs[i][j] * math.log(vobs[i][j] / n)
    sums /= n
    ape = sums[0] - sums[1]
    chi_squared = 2.0 * n * (math.log(2) - ape)
    p_val = inc_gamma(pow(2, pattern_length-1), chi_squared/2.0)
    return p_val

# HYBRID SOLVER

# trying this with 1024 qubits, which has a minimum of 3 seconds annealing time : 42.666 bytes/s
x = Array.create('x', shape=1024, vartype='BINARY')

# Hamiltonian for the objective function
HZ = sum(0 * x for x in x)
H = 1 * HZ 

model = H.compile()
qubo, offset = model.to_qubo()
bqm = model.to_bqm()

# sampler = LeapHybridSampler()
# sampleset = sampler.sample(bqm,
#                                  time_limit=3,
#                                  label="QRNG TEST")

# decoded_samples = model.decode_sampleset(sampleset)
# # best sample is what holds all the binary values but also has the energy
# best_sample = min(decoded_samples, key=lambda x: x.energy)
# print(frequencyInBlock(best_sample.sample))

fValues = []
fibValues = []
runValues = []
spectralValues = []
longestRunValues = []
nonOverValues = []
overValues = []
linearValues = []
serialValues = []
entropyValues = []

# using template 000000001
template = "000000001"
#running multiple instances of generating random numbers and counting how many times each test runs successfully
for i in range(1):
    sampler = LeapHybridSampler()
    sampleset = sampler.sample(bqm,
                                time_limit=3,
                                label="QRNG TEST")

    decoded_samples = model.decode_sampleset(sampleset)
    # best sample is what holds all the binary values but also has the energy
    best_sample = min(decoded_samples, key=lambda x: x.energy)
    # by taking best_sample.sample, we get the dictionary by itself and can analyze just the binary values
    fValues.append(frequencyTest(best_sample.sample))
    fibValues.append(frequencyInBlock(best_sample.sample))    
    runValues.append(runsTest(best_sample.sample))
    spectralValues.append(spectralTest(best_sample.sample))
    longestRunValues.append(longestRunTest(best_sample.sample))
    nonOverValues.append(nonOverlap(best_sample.sample, template))
    overValues.append(overlappingTemplate(best_sample.sample))
    linearValues.append(linear_complexity(best_sample.sample))
    serialValues.append(serial(best_sample.sample))
    entropyValues.append(entropy(best_sample.sample))

fSum = 0.0
fibSum = 0.0
runSum = 0.0
spectralSum = 0.0
longestRunSum = 0.0
nonOverlapSum = 0.0
overSums = 0.0
linearSums = 0.0
serialSums = 0.0
entropySums = 0.0

for i in range(len(fValues)):
    fSum += fValues[i]
    fibSum += fibValues[i]
    runSum += runValues[i]
    spectralSum += spectralValues[i]
    longestRunSum += longestRunValues[i]
    nonOverlapSum += nonOverValues[i]
    overSums += overValues[i]
    linearSums += linearValues[i]
    serialSums += serialValues[i]
    entropySums += entropyValues[i]

avgFValue = fSum/1
avgFibValue = fibSum/1
avgRunValue = runSum/1
avgSpectralValue = spectralSum/1
avgLongestRun = longestRunSum/1
avgNonOver = nonOverlapSum/1
avgOver = overSums/1
avgLinear = linearSums/1
avgSerial = serialSums/1
avgEntropy = entropySums/1

# print(avgFValue)
# print(avgFibValue)
# print(avgRunValue)
# print(avgSpectralValue)
# print(avgLongestRun)
# print(avgNonOver)
# print(avgOver)

# drawing the plot
df = pd.DataFrame({
    # 'group':'A',
    'Frequency' : [avgFValue],
    'Frequency in Block' : [avgFibValue],
    'Runs' : [avgRunValue],
    'Spectral' : [avgSpectralValue],
    'Longest Run' : [avgLongestRun],
    'Non Overlapping tmeplate' : [avgNonOver],
    'Overlapping Template' : [avgOver],
    "Linear complexity test" : [avgLinear],
    "Serial test" : [avgSerial],
    "Approximate Entropy test" : [avgEntropy]
})

categories = list(df)[0:]
N = len(categories)

values = df.loc[0].drop('group').values.flatten().tolist()
values += values[:1]
values

angles = [n / float(N) * 2 * math.pi for n in range(N)]
angles += angles[:1]

ax = plt.subplot(111, polar = True)

plt.xticks(angles[:-1], categories, color = 'grey', size = 8)

plt.yticks([0.1,0.2,0.4,0.6,0.8], ["0.1", "0.2", "0.4", "0.6", "0.8"], color ='grey', size = 8)
plt.ylim(0,1)

ax.plot(angles, values, linewidth = 1, linestyle = 'solid')
ax.fill(angles, values, 'b', alpha = 0.1)
plt.show()


# sum = 0
# for i in range(len(best_sample.sample)):
#     value = 2*best_sample.sample['x['+str(i)+']'] -1
#     sum = sum + value
# if sum < 0:
#     sum = 0 - sum
# print(sum)
# testStat = sum/(math.sqrt(len(best_sample.sample)))
# pVal = math.erf(testStat/(math.sqrt(2)))
# if pVal < 0.01:
#     print("We have concluded that the sequence is not random.")
# else:
#     print("We have concluded that the sequence is random.")

# lineup_df = pd.DataFrame(best_sample.sample.items())
# lineup_df.columns = ['Variable', 'Selected']
# # lineup_df = players_df.merge(lineup_df, on=['Variable'])
# lineup_df.sort_values(by=['Variable'])
# print(lineup_df)
# numstr = ''
# for s in lineup_df['Selected']:
#     numstr += str(s)
# number = str(int(numstr, 2))

# with open('result.txt', 'a') as a:
#     a.write(number+'\n')
#     a.close()

