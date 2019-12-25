import sys

import pandas as pd

data = pd.read_csv(sys.argv[1], sep=';', encoding='utf-8')
points = data['Custom field (Story Points)']


def verify_lenght(approach):
    contador = 0
    with open('vetores_abordagem%d.txt' % approach, 'r') as old:
        for line in old:
            print(len(line.split(" ")))
            contador += 1
    print(contador)


def append_class(approach):
    count = -1
    with open('vetores_abordagem%d.txt' % approach, 'r') as old:
        with open('VetoresAbordagem%d.txt' % approach, 'w') as new:
            for line in old:
                count += 1
                line = line.rstrip('\n') + " %d" % points[count]
                print(line, file=new)
    old.close()
    new.close()


approaches = [1, 2, 3]
for approach in approaches:
    append_class(approach)
