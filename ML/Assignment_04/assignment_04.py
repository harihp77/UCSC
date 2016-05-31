#!/usr/bin/env python
import pandas as pd
import numpy as np

df = pd.read_excel("a4.xlsx")
df.insert(0,"DummyWeight", 1)
table = df.as_matrix()
print "Rows and columes", np.shape(table)
print table

