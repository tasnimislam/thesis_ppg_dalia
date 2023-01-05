import numpy as np

ECG_SAMPLING_RATE = 700
PPG_SAMPLING_RATE = 64
ACTIVITY_SAMPLING_RATE = 4
EMG_SAMPLING_RATE = 700

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

def preprocess_according_paper2(subject_no):
    ecgTrain = []
    labelTrain = []
    ecgTest = []
    labelTest = []
    for i in tqdm(range(1,16)):
        pklFilePath = f'../input/ppgdalia/data/PPG_FieldStudy/S{i}/S{i}.pkl'
        with open(pklFilePath, 'rb') as f:
            u = pickle._Unpickler(f)
            u.encoding = 'latin1'
            data = u.load()
        label = data['label']
        ecg = np.array(data['signal']['wrist']['BVP'])
        ecg_window = WindowGet(ecg, PPG_SAMPLING_RATE)

        if i==(subject_no):
            labelTest.extend(label)
            ecgTest.extend(ecg_window)
        else:
            labelTrain.extend(label)
            ecgTrain.extend(ecg_window)

    X_valid, X_test, y_valid, y_test = train_test_split(np.array(ecgTest), np.array(labelTest), test_size=0.5, random_state=42)
    return (np.array(ecgTrain), np.array(labelTrain)), (X_valid, y_valid), (X_test, y_test)
