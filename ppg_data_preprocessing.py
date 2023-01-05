import numpy as np

def WindowGet(signal, samplingRate):
    window = []
    index = 0
    windowLen = 8*samplingRate
    overlapLen = 2*samplingRate
    while(index<=(len(signal) - windowLen)):
        win = signal[index: index+windowLen]
        window.append(win)
        index = index + overlapLen
    window = np.squeeze(np.array(window))
    return window