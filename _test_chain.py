import sys
from chain_mrf import ChainMRFPotentials, SumProduct, MaxSum

data_file = 'sample_mrf_potentials_small.txt'

p = ChainMRFPotentials(data_file)
sp = SumProduct(p)

print()