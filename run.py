import os

n_estimators = [100,200,210,190,250]
max_depth = [10,20,30,40,50]

for n in n_estimators:
    for m in max_depth:
        os.system(f"python basic_ml_model.py -n{n} -m{m}")


