import sys
import time
import logging
import functools
import math
from scipy.special import gamma, gammaincc, exp1
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


#fourth test
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

# Test 6
# Define your binary sequence (ε)
epsilon = "1100100100001111110110101010001000100001011010001100001000110100110001001100011000110010100010111000"

# Length of the sequence
n = len(epsilon)

# Convert the binary sequence to a sequence of ±1 values
sequence = [2 * int(bit) - 1 for bit in epsilon]

# Calculate the Discrete Fourier Transform (DFT)
dft_result = fft(sequence)

# Calculate the modulus of the DFT result for the first half of the sequence
modulus_dft = np.abs(dft_result[:n // 2])

# Calculate the threshold value T for the 95% peak heights
threshold = np.percentile(modulus_dft, 95)

# Calculate the observed number of peaks that are less than the threshold
observed_peaks = np.sum(modulus_dft < threshold)

# Calculate the expected number of peaks under the assumption of randomness
expected_peaks = 0.95 * n / 2

# Calculate the normalized difference (d)
d = (observed_peaks - expected_peaks) / np.sqrt(n * 0.95 * 0.05 / 4)

# Calculate the P-value using the complementary error function (erfc)
p_value = erfc(np.abs(d))

# Define the significance level (1%)
alpha = 0.01

# Perform the test and make a conclusion
if p_value < alpha:
    conclusion = "Non-random"
else:
    conclusion = "Random"

# Output the results
print(f"Test Statistic (d): {d}")
print(f"P-value: {p_value}")
print(f"Conclusion: The sequence is {conclusion}")


# seventh test

# Define your binary sequence (ε)
epsilon = "your_binary_sequence_here"
n = len(epsilon)

# Define the template B
B = "your_template_here"
m = len(B)

# Partition the sequence into blocks (adjust the block size if needed)
block_size = m
N = n // block_size

# Count the number of template matches in each block
template_count = []
for i in range(N):
    block = epsilon[i * block_size : (i + 1) * block_size]
    count = block.count(B)
    template_count.append(count)

# Calculate expected mean and variance
mu = (block_size - m + 1) / (2**m)
sigma2 = block_size / (2**(2 * m))

# Compute the test statistic (χ^2)
chi_squared = sum(((count - mu) ** 2) / sigma2 for count in template_count)

# Calculate P-value
p_value = 1 - chi2.cdf(chi_squared, N - 1)

# Define the significance level (1%)
alpha = 0.01

# Perform the test and make a conclusion
if p_value < alpha:
    conclusion = "Non-random"
else:
    conclusion = "Random"

# Output the results
print(f"Test Statistic (χ^2): {chi_squared}")
print(f"P-value: {p_value}")
print(f"Conclusion: The sequence is {conclusion}")


#8th test 

def overlapping_template_matching_test(binary_sequence, template):
    # Count occurrences of the template in the binary sequence
    counts = []
    pattern_len = len(template)
    for i in range(len(binary_sequence) - pattern_len + 1):
        window = binary_sequence[i:i + pattern_len]
        if window == template:
            counts.append(1)
        else:
            counts.append(0)

    # Calculate test statistics
    observed_counts = counts.count(1)
    expected_counts = len(binary_sequence) / pattern_len
    chi_square = sum([(count - expected_counts) ** 2 / expected_counts for count in counts])

    # Calculate P-value using the chi-squared distribution
    df = len(counts) - 1  # degrees of freedom
    p_value = 1 - stats.chi2.cdf(chi_square, df)

    return p_value

# Example usage:
binary_sequence = "10111011110010110100011100101110111110000101101001"
template = "111111111"
p_value = overlapping_template_matching_test(binary_sequence, template)
print(f"P-value: {p_value}")

# Decide if the sequence is random or non-random based on the P-value
if p_value < 0.01:
    print("Non-random")
else:
    print("Random")


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
longestRunValues = []

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
    longestRunValues.append(longestRunTest(best_sample.sample))
fSum = 0.0
fibSum = 0.0
runSum = 0.0
spectralSum = 0.0
longestRunSum = 0.0
for i in range(len(fValues)):
    fSum += fValues[i]
    fibSum += fibValues[i]
    runSum += runValues[i]
    spectralSum += spectralValues[i]
    longestRunSum += longestRunValues[i]

avgFValue = fSum/50
avgFibValue = fibSum/50
avgRunValue = runSum/50
avgSpectralValue = spectralSum/50
avgLongestRun = longestRunSum/50

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
    print("We have concluded that the random generator does generate random numbers based on the results of the Spectral Test")
else:
    print("We have concluded that the random generator does not generate random numbers based on the results of the Spectral Test")
if avgLongestRun > 0.01:
    print("We have concluded that the random generator does generate random numbers based on the results of the Longest Run of Ones Test")
else:
    print("We have concluded that the random generator does not generate random numbers based on the results of the Longest Run of Ones Test")



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

