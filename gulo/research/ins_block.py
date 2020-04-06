import ib_insync
import datetime
import numpy
import pandas
from gulo.utils import expected_sd as exp_sd


instr = Future('CL', 'NYMEX')
instr = ib.reqContractDetails(instr)
instr = instr[0].contract # first expiration date

p = ??? # price in instrument currency
block = int(instr.multiplier)  # unit size of an instrument:
# 1 for a stock, 100 for an option, 1000 barrels WTI futures...

blockv = p * 0.01 * block  # block value:
# cash gain of a 1% variation
# in the instrument currency

vol = exp_sd()  # price volatility

icvol = blockv * vol  # instrument currency volatility:
# expected deviation of daily returns
# from owning 1 instrument block
# in the currency of the instrument.

e = ???
ivvol = icvol * e 
