import sys
from chain_mrf import ChainMRFPotentials, SumProduct, MaxSum

data_file = 'sample_mrf_potentials.txt'

p = ChainMRFPotentials(data_file)
sp = SumProduct(p)

for i in range(1, p.chain_length() + 1):
    print(sp.marginal_probability(i))