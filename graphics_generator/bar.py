import sys

import numpy as np
import matplotlib.pyplot as plt


class_names = ['0', '1', '2', '3', '5', '8', '13', '20', '40', '89']
file = np.loadtxt(sys.argv[1]).tolist()
qtd = []
file = list(map(lambda x: int(x), file))

for classe in class_names:
    qtd.append(file.count(int(classe)))

plt.bar(class_names, qtd, color="indigo") #orange
plt.xlabel("Classes (Pontos de história)")
plt.ylabel("Frequência")
plt.title("Distribuição das Classes")
plt.show()