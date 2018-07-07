import numpy as np
import pandas as pd
from collections import Counter
from random import shuffle

train_data = np.load('data/training_data.npy')

df = pd.DataFrame(train_data)
print(df.head())
print(Counter(df[1].apply(str)))
print(df[1])
lefts = []
rights = []
no_turns = []

shuffle(train_data)


for data in train_data:
    img = data[0]
    choice = list(data[1])
    if choice == [-1,0,0]:
    	lefts.append([img,choice])
    elif choice == [1,0,0]:
        rights.append([img,choice])
    elif choice == [0,0,0]:
        no_turns.append([img,choice])


no_turns = no_turns[:len(lefts)][:len(rights)]
lefts = lefts[:len(no_turns)]
rights = rights[:len(no_turns)]

print(len(lefts),len(rights),len(no_turns))

final_data = no_turns + lefts + rights

shuffle(final_data)

np.save('data/training_data_balanced.npy',final_data)