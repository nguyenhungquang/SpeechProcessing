import librosa
import numpy as np
import os
import math
from sklearn.cluster import KMeans
import sklearn
import hmmlearn.hmm
import pickle
def get_class_data(data_dir):
    files = os.listdir(data_dir)
    mfcc = [get_mfcc(os.path.join(data_dir,f)) for f in files if f.endswith(".wav")]
    return mfcc
def get_mfcc(file_path):
    y, sr = librosa.load(file_path) # read .wav file
    hop_length = math.floor(sr*0.010) # 10ms hop
    win_length = math.floor(sr*0.025) # 25ms frame
    # mfcc is 12 x T matrix
    mfcc = librosa.feature.mfcc(
        y, sr, n_mfcc=12, n_fft=1024,
        hop_length=hop_length, win_length=win_length)
    energy= librosa.feature.rms(y=y,frame_length=win_length,hop_length=hop_length)
    mean=np.mean(mfcc, axis=1).reshape((-1,1)) #calculate mean 
    var= np.sum(mfcc**2-mean**2,axis=1).reshape((-1,1))/mfcc.shape[1] #calculate variance
    mfcc = (mfcc - mean)/np.sqrt(var) #mean and variance normalisation
    if energy.shape[1]!=mfcc.shape[1]:
        energy=np.append(energy,np.zeros(mfcc.shape[1]-energy.shape[1]))
    mfcc=np.vstack([mfcc,energy]) #add energy
    # delta feature 1st order and 2nd order
    delta1 = librosa.feature.delta(mfcc, width=9, order=1)
    delta2 = librosa.feature.delta(mfcc, width=9, order=2)
    # X is 36 x T
    X = np.concatenate([mfcc, delta1, delta2], axis=0) # O^r
    # return T x 36 (transpose of X)
    return X.T # hmmlearn use T x N matrix
class_names = ["ngay","nhieu","nhung","gia_dinh","dich_benh"]
test_classes=["test_ngay","test_nhieu","test_nhung","test_gia_dinh","test_dich_benh"]
dataset = {}
testset={}
def get_data(class_list):
    dt={}
    for cname in class_list:
        print(f"Load {cname}")
        dt[cname] = get_class_data(os.path.join("data", cname))
    return dt
dataset=get_data(class_names)
testset=get_data(test_classes)
word_models={"ngay":824,"nhieu":811,"nhung":1145,"gia_dinh":215,"dich_benh":216}
for name in word_models.keys():
    word_models[name]=np.log(word_models[name])
def transmat(size):
    pri=5*np.eye(size)+5*np.eye(size,k=1)+2*np.eye(size,k=2)
    pri/=12
    for i in range((size-1)//3):
        pri[3*i+2,3*i+2]=0
    pri[-1,-1]=1
    return pri
def start_prior(size):
    pri=np.zeros(size)
    pri[0]=pri[1]=0.5
    return pri
def train():
    components_dict=dict(zip(class_names,[9,9,9,18,18]))
    models={}
    for cname in class_names:
        #if cname[:4] != 'test':
        # convert all vectors to the cluster index
        # dataset['one'] = [O^1, ... O^R]
        # O^r = (c1, c2, ... ct, ... cT)
        # O^r size T x 1
        hmm = hmmlearn.hmm.GMMHMM(
                n_components=components_dict[cname], n_mix=2,random_state=0, n_iter=50, verbose=False, 
                 params='mctw',
                init_params='ts',
                transmat_prior=transmat(components_dict[cname]), 
                startprob_prior = start_prior(components_dict[cname])
        )
        X = np.concatenate(dataset[cname])
        lengths = list([len(x) for x in dataset[cname]])
        print("training class", cname)
    #     print(len(X))
    #    print(X.shape, lengths, len(lengths))
        hmm.fit(X)#, lengths=lengths)
        models[cname] = hmm
    return models
def test(models):
    print("Testing")
    f=0
    for name in test_classes:
        for O in testset[name]:
            true_cname=name[5:]
            score = {cname : (model.score(O, [len(O)])+word_models[cname]) for cname, model in models.items() }
            inverse = [(value, key) for key, value in score.items()]
            pre = max(inverse)[1]
            if true_cname!=pre:
                f+=1
    print("Error:", f,"%")
if __name__=="__main__":
    load=True
    save=False
    models={}
    if not load:
        models=train()
    else:
        for name in class_names:
            with open(name+".pkl", "rb") as file: 
                models[name]=pickle.load(file)
    test(models)
    if save:
        for name in class_names:
            with open(name+".pkl", "wb") as file: pickle.dump(models[name], file)

