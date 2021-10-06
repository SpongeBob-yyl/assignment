from sklearn.neural_network import MLPClassifier
import pandas as pd
import numpy as np
import csv
fixed_star = pd.read_csv('train.csv')
fixed_star['abs_Plx'] = abs(fixed_star.Plx)
fixed_star['Amag'] = fixed_star.Vmag + (fixed_star.abs_Plx.apply(np.log10) + 1) * 5
fixed_star = fixed_star[fixed_star.e_Plx < 2]
stars = fixed_star[['Vmag', 'Plx', 'B-V', 'Amag']]
star_target = fixed_star['TargetClass']

fixed_star = pd.read_csv('test.csv')
fixed_star['abs_Plx'] = abs(fixed_star.Plx)
fixed_star['Amag'] = fixed_star.Vmag + (fixed_star.abs_Plx.apply(np.log10) + 1) * 5
testSet = fixed_star[['Vmag', 'Plx', 'B-V', 'Amag']]

network = MLPClassifier(hidden_layer_sizes=(100,200), activation='relu', alpha=0.0001, shuffle=True, learning_rate_init=0.001, random_state=1)
network.fit(stars, star_target)
classifierResults = network.predict(testSet)

with open('samples.csv', 'r') as f:
    f_csv = csv.reader(f)
    headers = next(f_csv)
    k = 0
    data = []
    for row in f_csv:
        print(row)
        row[1] = str(classifierResults[k])
        data.append(row)
        k += 1
with open('samples.csv', 'w', newline='') as f:
    f_csv = csv.writer(f)
    f_csv.writerow(headers)
    f_csv.writerows(data)
