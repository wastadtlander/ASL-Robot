import numpy as np
ii = 0
uu = 0
labels = []
string = []

def commandHook(topLabels):
    global ii, uu, string, labels
    if ii < 40 and uu < 2:
        labels.append(topLabels[0])
        ii = ii+1
    else:
        ii = 0
        if len(labels) > 0:
            avg_label = np.round(np.average(labels))
            string.append(avg_label)
            labels = []
            uu += 1
            if uu == 2:
                uu = 0
                command = string
                string = []

                if command == [0, 1]:
                    return 2
                elif command == [1, 2]:
                    return 3
                elif command == [2, 0]:
                    return 1

    return 0
