import sys
import time
import logging
import functools
import math
from scipy.special import gamma, gammaincc, exp1
from scipy.fftpack import fft
from collections import OrderedDict
from itertools import product

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
def frequencyInBlock(dictionary, blockSize):
    if blockSize == 1:
        frequencyTest(dictionary)
    numBlocks = math.floor(len(dictionary) / blockSize)
    blockStart, blockEnd = 0, blockSize
    proportionSum = 0.0
    for i in range(numBlocks):
        blockData = ""
        for i in range(blockStart, blockEnd):
            blockData = blockData + str(dictionary['x['+str(i)+']'])
        onesCount = 0
        for char in blockData:
            if char == '1':
                onesCount += 1
        pi = onesCount / blockSize
        proportionSum += pow(pi - 0.5, 2.0)
        blockStart += blockSize
        blockEnd += blockSize
    chiSquared = 4.0 * blockSize * proportionSum
    pVal = inc_gamma(numBlocks / 2, chiSquared / 2)
    return pVal

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


#third test
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
    p_val = math.erf(abs(d) / numpy.sqrt(2))
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


fValues = []
fibValues = []
runValues = []
spectralValues = []

#running multiple instances of generating random numbers and counting how many times each test runs successfully
for i in range(50):
    sampler = LeapHybridSampler()
    sampleset = sampler.sample(bqm,
                                time_limit=3,
                                label="QRNG TEST")

    decoded_samples = model.decode_sampleset(sampleset)
    # best sample is what holds all the binary values but also has the energy
    best_sample = min(decoded_samples, key=lambda x: x.energy)
    # by taking best_sample.sample, we get the dictionary by itself and can analyze just the binary values
    fValues.append(frequencyTest(best_sample.sample))
    fibValues.append(frequencyInBlock(best_sample.sample, 100))    
    runValues.append(runsTest(best_sample.sample))
    spectralValues.append(spectralTest(best_sample.sample))
fSum = 0.0
fibSum = 0.0
runSum = 0.0
spectralSum = 0.0
for i in range(len(fValues)):
    fSum += fValues[i]
    fibSum += fibValues[i]
    runSum += runValues[i]
    spectralSum += spectralValues[i]

avgFValue = fSum/50
avgFibValue = fibSum/50
avgRunValue = runSum/50
avgSpectralValue = spectralSum/50

if avgFValue > 0.01:
    print("We have concluded that the random generator does generate random numbers based on the results of the Frequency test")
else:
    print("We have concluded that the random generator does not generate random numbers based on the results of the Frequency test.")
if avgFibValue > 0.01:
    print("We have concluded that the random generator does generate random numbers based on the results of the blocks test.")
else:
    print("We have concluded that the random generator does not generate random numbers based on the results of the block test.")
if avgRunValue > 0.01:
    print("We have concluded that the random generator does generate random numbers based on the results of the Runs Test")
else:
    print("We have concluded that the random generator does not generate random numbers based on the results of the Runs Test")
if avgSpectralValue > 0.01:
    print("We have concluded that the ranodm generator does generate random numbers based on the results of the Spectral Test")
else:
    print("We have concluded that the random generator does not generate random numbers based on the results of the Spectral Test")

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
# lineup_df = players_df.merge(lineup_df, on=['Variable'])
# lineup_df.sort_values(by=['Variable'])
# print(lineup_df)
# numstr = ''
# for s in lineup_df['Selected']:
#     numstr += str(s)
# number = str(int(numstr, 2))

# with open('result.txt', 'a') as a:
#     a.write(number+'\n')
#     a.close()

