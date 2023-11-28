import numpy as np
ii = 0
uu = 0
labels = []
string = []

def commandHook(self, topLabels):
    global ii, uu, string, labels
    if ii < 20 and uu < 2:
        labels[ii] = topLabels[0]
        ii = ii+1
    else:
        ii = 0
        labels = []
        string[uu] = np.round(np.average(labels))
        uu = uu+1
    if uu == 1:
        uu = 0
        command = string
        string = []

    if command == [0, 2]:
        return 1
    else:
        return 0
