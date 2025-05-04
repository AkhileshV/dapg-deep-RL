import csv
import matplotlib.pyplot as plt 
from collections import defaultdict

paths = ["dapg_exp/logs/log.csv", "dapg_exp_PPO/logs/log.csv"]

# for i in range(1, 4):
#     file = open("multiple_experiments/relocate_shuffle_demopaths/exp_{0}/logs/log.csv".format(i))
for path in paths:
    file = open(path)
    csvFile = csv.DictReader(file)
    iter=[]
    # exps_success_map = {}
    success_rate=[]

    for lines in csvFile:
        iter.append(int(lines["iteration"]))
        success_rate.append(float(lines["success_rate"]))

    # exps_success_map[i] = success_rate
    print(path.split("/")[0], max(success_rate))
    plt.plot(iter, success_rate, label=path.split("/")[0])
    
plt.legend()
plt.xlabel("iteration")
plt.ylabel("Success rate")
plt.savefig("compare_dapg_ppo.png")
plt.show()


