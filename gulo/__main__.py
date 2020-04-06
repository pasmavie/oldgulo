import numpy as np
import pandas as pd
from gulo.research.mean_reversion.strats import SimpleMrStrat

mes = pd.read_csv("~/Desktop/mes.csv", delimiter=";")

mes["date"] = pd.to_datetime(mes.date)

train = mes[(mes.date>pd.to_datetime("20200201")) & (mes.date<pd.to_datetime("20200311"))].index

test = mes[mes.date>pd.to_datetime("20200310 22:00:00")].index

sr = SimpleMrStrat(max_hl=24000,entry=1,exit=0)

sr.fit(mes.loc[train].average)

pos = []
for t in test[:-1]:
    pos += [sr.trade(mes.loc[:t+1].average)]
    

r = pos*mes.loc[test].average.values[:-1]*5

