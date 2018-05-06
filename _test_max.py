import sys
from chain_mrf import ChainMRFPotentials, SumProduct, MaxSum

data_file = 'sample_mrf_potentials.txt'

p = ChainMRFPotentials(data_file)
ms = MaxSum(p)
