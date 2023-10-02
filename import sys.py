import sys
import time
import logging
import functools
from collections import OrderedDict

import dwavebinarycsp as dbc
from dwave.system import DWaveSampler, EmbeddingComposite

import pandas as pd
​
from pyqubo import Constraint, Array
import neal
from dwave.system import LeapHybridSampler
​
# HYBRID SOLVER
​
# trying this with 1024 qubits, which has a minimum of 3 seconds annealing time : 42.666 bytes/s
x = Array.create('x', shape=1024, vartype='BINARY')
​
# Hamiltonian for the objective function
HZ = sum(0 * x for x in x)
​
H = -1 * HZ 
​
model = H.compile()
qubo, offset = model.to_qubo()
bqm = model.to_bqm()
​
# Solve problem with QPU
sampler = LeapHybridSampler()
sampleset = sampler.sample(bqm,
                            time_limit=3,
                            label="QRNG TEST")
​
decoded_samples = model.decode_sampleset(sampleset)
best_sample = min(decoded_samples, key=lambda x: x.energy)
​
lineup_df = pd.DataFrame(best_sample.sample.items())
lineup_df.columns = ['Variable', 'Selected']
#lineup_df = players_df.merge(lineup_df, on=['Variable'])
lineup_df.sort_values(by=['Variable'])
print(lineup_df)
numstr = ''
for s in lineup_df['Selected']:
    numstr += str(s)
number = str(int(numstr, 2))
​
with open('result.txt', 'w') as w:
    w.write(number+'\n')
    w.close()