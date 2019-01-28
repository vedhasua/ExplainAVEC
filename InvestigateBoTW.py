# Contact: Vedhas Pandit

from __future__ import unicode_literals
import math
import time
import innvestigate
FlagGroupFeatures = 0
PP = 1
CreateCloud = 0
dbool = {False: '0', True: '1'}
applymedfilt = False
Emotion = ['Arousal', 'Valence', 'Liking ']
methods = ["input_t_gradient", 'gradient', 'lrp.z', 'lrp.alpha_2_beta_1', 'deep_lift']  # , 'pattern.attribution']
methods = ["input_t_gradient","lrp.epsilon","deep_taylor","guided_backprop"] #"integrated_gradients" "pattern.net"
methods = ['lrp.z','lrp.alpha_beta',"lrp.epsilon","lrp.alpha_1_beta_0","lrp.alpha_2_beta_1"]
methods = ['lrp.z','deep_lift.wrapper']
Emo = 'AVL'
numMetr = 3
nTrees = 20
sr_labels = 0.1
import collections
import matplotlib
matplotlib.use("TkAgg")
from matplotlib import gridspec
import seaborn as sns
# sns.set(style='ticks', palette='Set2')
# sns.set(color_codes=True)
sns.set_palette("bright")

# sns.palplot(current_palette)

import pylab
# from pylab import *
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import tensorflow as tf
import os
import fnmatch
import modules
from sklearn import svm
from load_features import load_all
from calc_scores import calc_scores
import numpy as np
import keras
from keras.models import Model, Sequential
from keras.models import load_model
from keras.optimizers import RMSprop, SGD, Adam, Adagrad, Adamax, Nadam, Adadelta
from keras.layers.normalization import BatchNormalization
from keras.layers import Input, LSTM, Masking, merge, Average, GRU, Bidirectional, TimeDistributed, Concatenate, \
    concatenate
from keras.layers import Dense, Dropout, Lambda
from keras import backend as K
from keras.regularizers import l2  # L2-regularisation
from keras.callbacks import ModelCheckpoint,EarlyStopping
from os import environ
from scipy.signal import medfilt
from keras.utils.training_utils import multi_gpu_model

np.set_printoptions(threshold=np.nan)
# K.set_learning_phase(1)


def set_keras_backend(backend):
    if K.backend() != backend:
        environ['KERAS_BACKEND'] = backend
        reload(K)
        assert K.backend() == backend


# ==== For Data Hyperparams
SuccessThresh = np.array([[0.373, 0.375],  # CCC    DevelThresh,TestThresh  Arousal
                          [0.390, 0.425],  # CCC    DevelThresh,TestThresh  Valence
                          [0.314, 0.246]])  # CCC    DevelThresh,TestThresh  Liking
verbose = 2
ComputeSVM = False;
C = [2 ^ -6, 2 ^ -7, 1]
ComputeFFN = True
PlotFlag = False
FindFeatureWeights = True
typeScaling = 2
typeScalingInput = 0
typeScalingDict = {0: ' (No scaling)', 1: ' (Normalise)', 2: ' (Standardise)'}
delay = 2
numAnno = 3
b_lldio = 0;
b_audio = 0;
b_video = 0;
b_text = 1;

if applymedfilt: medfiltK = [31, 1]


def movingaverage(values, window, lengthtype='valid'):
    weights = np.repeat(1.0, window) / window
    sma = np.convolve(values, weights, lengthtype)
    return sma


def QuickNormalise(x):
    return (x - np.min(x)) / (np.max(x) - np.min(x))


if FlagGroupFeatures:
    GroupDict = {
        'at': [7, 8],
        'sehen':      [11,12,229,464],
        'because':  [18,24,110,186],
        'articles': [21, 493, 517],
        'hallo': [236, 71, 332],
        'yes':      [36,90],    #324
        'filler': [118, 136, 171],  # 123,160,
        'must':     [175,241,148,272],#166,
        'hear':     [17,188],
        'boring': [201, 450],
        'ein':      [130,131,132,100,114],
        'in': [83, 84],
        'all': [184, 187, 224],
        'ganz': [299, 346, 502, 138],
        'has':      [437,345,95,461],
        'diese': [102, 280, 281, 292, 294],
        'weiss':    [113,333],
        'jede': [349, 350],
        'irgend': [105, 116, 164, 177, 501],
        'gut': [364, 520],
        'meinst':   [227,325,402],
        'mein': [330, 492],
        'komm':     [400,206],
        'komisch': [486, 60],
        'mach':     [413,127,153,319],
        'andere': [234, 257],
        'beispiel': [194, 301, 406],
        'best': [79, 363],
        'no':       [91,379], #remove ne,na 128, 129,
        'schlecht': [69, 159],
        'gleich': [232, 465],
        'hand': [14, 275, 16],
        'lang': [19, 137],
        'letzte': [399, 246],
        'situation': [419, 23],
        'viel': [307, 209],
        'verschiedener': [221, 225],
        'tap': [318, 488, 336],
        'advertisement': [142, 143, 167, 375, 156]
    }
    del GroupDict['sehen'], \
        GroupDict['because'], \
        GroupDict['yes'], \
        GroupDict['must'], \
        GroupDict['hear'], \
        GroupDict['ein'], \
        GroupDict['has'], \
        GroupDict['weiss'], \
        GroupDict['meinst'], \
        GroupDict['komm'], \
        GroupDict['mach'], \
        GroupDict['no']
    GroupDict = {
        'articles': [21, 493, 517],
    }
    GroupDict = collections.OrderedDict(sorted(GroupDict.items()))
    def GroupFeatures(datacopy, GroupDict):
        GroupedFeatIds = [GroupDict[featType] for featType in GroupDict]
        GroupedFeatIds = [item for sublist in GroupedFeatIds for item in sublist]
        UnGroupedFeatIds = [_ for _ in range(521) if _ not in GroupedFeatIds]
        dataMerged = np.zeros((datacopy.shape[0], 0))  # zeross of size [Num of instances x 0]
        # data = np.power(10, datacopy) - 1                   #data contains (tf-1)
        data = np.power(10, datacopy) * np.where(datacopy > 0, 1, 0)  # data contains tf (>2, or 0)
        newvocab = []
        assert np.all(data >= 0)
        for featId, featType in enumerate(GroupDict):
            newvocab = newvocab + [featType]
            dataMerged = np.hstack((dataMerged, np.expand_dims(data[:, GroupDict[featType]].sum(axis=1), 1)))
        for featIndex in UnGroupedFeatIds:
            newvocab = newvocab + [Vocab[featIndex]]
            dataMerged = np.hstack((dataMerged, np.expand_dims(data[:, featIndex], 1)))
        dataMerged = dataMerged + np.where(dataMerged == 0, 1, 0)
        dataMerged = np.log10(dataMerged)
        newvocab = np.array(newvocab)
        return dataMerged, newvocab

def GoodRegions(test_sample_indices,maxGap):
    diff_test_sample_indices=np.diff(test_sample_indices.flatten())
    return test_sample_indices[np.where(diff_test_sample_indices<=maxGap)]

def ContRegions(test_sample_indices,maxGap,minlength=20):
    contlist=[]
    newlist=True
    for curid,curtindex in enumerate(test_sample_indices[:-1]):
        if test_sample_indices[curid+1]-curtindex<=maxGap and newlist==True:
            start=curtindex
            newlist = False
        elif (test_sample_indices[curid + 1] - curtindex > maxGap or curid==test_sample_indices.shape[0]-2) and newlist==False:
            end = curtindex
            newlist = True
            if end-start>=minlength:
                contlist.append([start,end])
    return contlist

# Function used for log file, converts list to string format, removing []' and space, replacing , with -
def Lstr(mylist, delim='-'):
    if type(mylist) is np.ndarray:
        mylist = mylist.tolist()
    mystr = str(mylist)
    for i, j in {'[': '', ']': '', ',': delim, ' ': '', "'": ''}.items():
        mystr = mystr.replace(i, j)
    return mystr


### For model
# loss = 'mse'
loss = 'mccc'
SelOptimizer = 'rmsprop'  # 'rmsprop'  #'adam' #'adagrad'
epochs = 50 # Number of epochs to train for.
lr = 0.001
batch_size=8192
l2w = 0.001 * 0
l2a = 0.001 * 0
patienceSt = 30
# activation='linear'
mvlength = 51
NumNodes = [256, 64, 16]  # [    240, 120, 30,  6]
DpOut = [0.25]
DpOut = DpOut + [0.1] * len(NumNodes)  # [0.3,0.3, 0.2, 0.1, 0.0]
activations = ['linear','relu', 'relu']  # ['softmax']*len(NumNodes)
activations = activations + ['linear']

ActDict = {'relu': modules.Rect(),
           'selu': modules.Selu(),
           'elu': modules.Elu(),
           'tanh': modules.Tanh(),
           'linear': modules.Identity(),
           'softmax': modules.SoftMax(),
           # 'maxpool':modules.MaxPool(),
           # 'maxpool':modules.SumPool()
           }


SaveDir = 'ai4hb2e/2019/'
if not os.path.exists(SaveDir):
    os.makedirs(SaveDir)
ExplainWhat = [2]
ResultsFile = \
    SaveDir + '/0A_BnTBL_GroupN_' + Lstr(NumNodes) + '_' + Lstr(activations) + '_sc' + str(typeScaling) + '_pp' + dbool[
        PP] + 'sYC' + '_G2r' + dbool[FlagGroupFeatures] if \
        (l2w and l2a) else \
        SaveDir + '/0BnTBL_GroupN_' + Lstr(NumNodes) + '_' + Lstr(activations) + '_sc' + str(typeScaling) + '_pp' + dbool[
            PP] + '_sNC' + '_G2r' + dbool[FlagGroupFeatures]

ResultsFile = ResultsFile + '_Expl' + Lstr(ExplainWhat) + 'ep' + str(epochs)

ResultsFile = ResultsFile + '.csv'

# SuccessThresh = SuccessThresh - 0.05 if not ('BTBL' in ResultsFile) else SuccessThresh
print(SuccessThresh)

# optimizer
adam = Adam(lr=lr);
adagrad = Adagrad(lr=lr);
rmsprop = RMSprop(lr=lr);
nadam = Nadam(lr=lr);
adamax = Adamax(lr=lr)
adadelta = Adadelta(lr=lr);
sgd = SGD(lr=lr)
optizid = {'adamax': 'm', 'adagrad': 'g', 'rmsprop': 'p', 'nadam': 'n', 'adamax': 'x', 'adadelta': 'd', 'sgd': 's'}
OptimDict = {'adam': adam, 'adagrad': adagrad, 'rmsprop': rmsprop, 'nadam': nadam, 'adamax': adamax,
             'adadelta': adadelta, 'sgd': sgd}
KerasOptimiser = OptimDict[SelOptimizer]
numGens = 16

del OptimDict, adam, adagrad, rmsprop, nadam, adamax, adadelta, sgd


def correlation_coefficient_loss(y_true, y_pred):
    x = y_true
    y = y_pred
    mx = K.mean(x)
    my = K.mean(y)
    xm, ym = x - mx, y - my
    r_num = K.sum(tf.multiply(xm, ym))
    r_den = K.sqrt(tf.multiply(K.sum(K.square(xm)), K.sum(K.square(ym))))
    r = r_num / r_den

    r = K.maximum(K.minimum(r, 1.0), -1.0)
    return 1 - K.square(r)


def mccc(y_true, y_pred):
    # mccc= mse/sigma_xy=mse/cov (since ccc=1/(1+0.5mccc))
    cov = 2*K.abs(K.mean(tf.multiply(y_true - K.mean(y_true), y_pred - K.mean(y_pred))))+1e-7
    mse = K.mean(K.square(y_pred - y_true))
    # cov2= K.pow(cov,0.1)
    return tf.math.divide(mse,cov)
    # return mse / cov

def mccc2(y_true, y_pred):
    # mccc= mse/sigma_xy=mse/cov (since ccc=1/(1+0.5mccc))
    cov = K.square(K.mean(tf.multiply(y_true - K.mean(y_true), y_pred - K.mean(y_pred))))
    mse = K.square(K.mean(K.square(y_pred - y_true)))
    # cov2= K.pow(cov,0.1)
    return tf.math.divide(mse,cov)

def mycov1(y_true, y_pred):
    cov = K.mean(tf.multiply(y_true - K.mean(y_true), y_pred - K.mean(y_pred)))
    return cov


def mycov2(y_true, y_pred):
    cov = K.mean(tf.multiply(y_true - K.mean(y_true), y_pred - K.mean(y_pred)), axis=-1)
    return cov

def mymse1(y_true, y_pred):
    # mccc= mse/sigma_xy=mse/cov (since ccc=1/(1+0.5mccc))
    mse = K.mean(K.square(y_pred - y_true), axis=-1)
    return mse

def mymse2(y_true, y_pred):
    # mccc= mse/sigma_xy=mse/cov (since ccc=1/(1+0.5mccc))
    mse = K.mean(K.square(y_pred - y_true))
    return mse


lossW = loss
metric = ['mse'];
metricW = metric if metric is not 'pmse' else metric + '%.3f' % LossPow
dictLosses = {'mse': 'mse', 'mae': 'mae', 'msle': 'mean_squared_logarithmic_error', 'mccc': mccc,'mccc2': mccc2}
# 'pmse':pmse, 'huber':huber,
ModelDir = SaveDir  # '/home/panditve/workspace/Sewa/AVEC2017/Results/Models/L'
if not os.path.exists(ModelDir):        os.makedirs(ModelDir)
VocabFile = '/home/panditve/workspace/Sewa/AVEC2017/OrigSetFromMax/DE_Orig_Feats/BACKUP_scripts/dictionary.txt'


def VocabList(boxwCB):
    with open(boxwCB, "r") as fid:
        vocablist = fid.readlines()
    # vocablist = [a.decode('utf-8').replace('\r\n', '') for a in vocablist]
    vocablist = [a.replace('\n', '') for a in vocablist]
    vocablist = vocablist[6:]
    vocablist = [a.lower() for a in vocablist]
    # vocablist =[a.encode('ascii', 'ignore').decode('ascii') for a in vocablist]
    return vocablist


def FindScaler(array2d, typeScaling=typeScaling):
    if typeScaling == 1:
        maxi = array2d.max(axis=0) + np.random.rand(1) / 1e6
        mini = array2d.min(axis=0) + np.random.rand(1) / 1e6
        scal = maxi - mini
        offs = mini
    elif typeScaling == 2:
        scal = array2d.std(axis=0)
        offs = array2d.mean(axis=0)
    elif typeScaling == 0:
        scal = 1
        offs = 0
    return scal, offs


def ScaleIt(array2d, scal, offs):
    # newarray2d = (array2d - offs) / scal
    newarray2d = array2d/scal - offs/scal
    return newarray2d


def cgx(array1d):
    array1d = np.squeeze(array1d)
    sindice = np.argsort(array1d)
    srray1d = array1d[sindice]
    indices = np.arange(array1d.shape[0])
    cgv = np.sum(indices * srray1d) / np.sum(srray1d)
    return cgv, sindice


# ===== MODEL DEFINITION ===========================================

def FindAbsGradSummations(mymodel, myXdata):
    # ==============================================================================
    #     from keras.objectives import mse
    #     m = get_your_model()
    #     y_true = K.placeholder(*your_models_output_shape)
    #     loss = K.mean(mse(y_true, m.output))
    #     get_grads = K.function([m.input, y_true], K.gradients(loss, m.input))
    #
    #     grads = get_grads([np.random.rand(*your_models_input_shape), np.random.rand(*your_models_output_shape)])
    # ==============================================================================
    # ==============================================================================
    #    outputTensor = model.output
    #    variableTensors = model.trainable_weights[0]
    #    print('Variable Tensors are: ',variableTensors.shape)
    #    gradients = k.gradients(outputTensor, variableTensors)
    #    y_true = K.placeholder(1)
    #    get_grads = K.function([m.input, y_true], K.gradients(loss, m.input))
    #
    #    grads = get_grads([np.random.rand(*your_models_input_shape), np.random.rand(*your_models_output_shape)])
    #
    #    sess = tf.InteractiveSession()
    #    sess.run(tf.initialize_all_variables())
    #    evaluated_gradients = sess.run(gradients,feed_dict={model.input:trainingExample})
    # ==============================================================================
    outputTensor = mymodel.output  # Or model.layers[index].output
    listOfVariableTensors = mymodel.trainable_weights[0]  # 0 indicates 0th layer, meaning model inputs
    #    print(listOfVariableTensors)
    gradients = K.gradients(outputTensor, listOfVariableTensors)  # gradients of output wr.t. input tensors
    sess = tf.InteractiveSession()
    sess.run(tf.initialize_all_variables())
    evaluated_gradients = sess.run(gradients, feed_dict={mymodel.input: myXdata})
    #    print(evaluated_gradients[0], evaluated_gradients[0].shape)
    return evaluated_gradients[0]


def FFNModel(featsize, opsize):
    KIw = keras.initializers.glorot_uniform(seed=seed)  # 'glorot_uniform'#keras.initializers.Constant(value=-0.001)
    KIwb = 'zeros'  # keras.initializers.Constant(value=-0.001)
    model = Sequential()
    # model.add(Lambda(lambda x: x, input_shape=(featsize,)))
    model.add(Dense(NumNodes[0], input_shape=(featsize,),
                    kernel_initializer=KIw,
                    bias_initializer=KIwb
                    ))
    # model.add(BatchNormalization())
    model.add(Dropout(DpOut[0], name='encoded'))
    for nid, numN in enumerate(NumNodes[1:]):
        model.add(Dense(numN, activation=activations[nid],
                        kernel_initializer=KIw, bias_initializer=KIwb))
        # model.add(BatchNormalization())
        model.add(Dropout(DpOut[nid + 1]))
    model.add(Dense(opsize, name="absolute_output", activation=activations[len(NumNodes)],
                    kernel_initializer=KIw, bias_initializer=KIwb,
                    # use_bias=0,
                    kernel_regularizer=l2(l2w), activity_regularizer=l2(l2a)))
    # model.compile(loss = dictLosses[loss], optimizer=KerasOptimiser,
    #               metrics=[dictLosses[m] for m in metric])
    # metrics=[dictLosses[metric]])#'mse', optimizer = rmsprop)
    #    print model.summary()
    return model


# ======== MAIN SCRIPT =========================

# Set folders here
# 'AVEC_17_Emotion_Sub-Challenge'
path_test_predictions = "test_predictions/"
b_test_available = True  # If the test labels are not available, the predictions on test are written into the folder 'path_test_predictions'
if not b_test_available and not os.path.exists(path_test_predictions):
    os.mkdir(path_test_predictions)

# Folders with provided features and labels
path_audio_features = "../AVEC_17_Emotion_Sub-Challenge/audio_features_xbow_6s/"
path_video_features = "../AVEC_17_Emotion_Sub-Challenge/video_features_xbow_6s/"
path_text_features = "../AVEC_17_Emotion_Sub-Challenge/text_features_xbow_6s/"
path_labels = "../AVEC_17_Emotion_Sub-Challenge/labels/"
path_lldio_features = "../AVEC_17_Emotion_Sub-Challenge/audio_features_functionals_6s/"
path_features = []
if b_audio:    path_features.append(path_audio_features)
if b_video:    path_features.append(path_video_features)
if b_text:    path_features.append(path_text_features); Vocab = VocabList(VocabFile);Vocab = np.array(
    Vocab)
if b_lldio:    path_features.append(path_lldio_features)

# Compensate the delay (quick solution)
shift = int(np.round(delay / sr_labels))
shift = np.ones(len(path_features), dtype=int) * shift

files_train = sorted(fnmatch.filter(os.listdir(path_features[0]),
                             "Train*"))  # Filenames are the same for audio, video, text & labels
files_devel = sorted(fnmatch.filter(os.listdir(path_features[0]), "Devel*"))
files_test = sorted(fnmatch.filter(os.listdir(path_features[0]), "Test*"))

# Load features and labels
Train = load_all(files_train, path_features, shift)
Devel = load_all(files_devel, path_features, shift)
Train_L = load_all(files_train, [path_labels])  # Labels are not shifted
Devel_L = load_all(files_devel, [path_labels])

if b_test_available:
    TestSt = load_all(files_test, path_features, shift,separate=True)
    Test_LSt = load_all(files_test, [path_labels],separate=True)  # Test labels are not available in the challenge
else:
    Test = load_all(files_test, path_features, shift,
                    separate=True)  # Load test features separately to store the predictions in separate files
# print(Test.shape,Test_L.shape,load_all(files_test, path_features, shift,separate=0).shape,load_all(files_test, [path_labels],separate=0).shape)
# print([a.shape[0] for a in Test])
# print([a.shape[0] for a in Test_L])
# print(np.vstack(TestSt).shape)
# print(np.vstack(Test_LSt).shape)
Test=np.vstack(TestSt)
Test_L=np.vstack(Test_LSt)
if applymedfilt:
    Train_L = medfilt(Train_L, medfiltK)
    Devel_L = medfilt(Devel_L, medfiltK)
    Test_L = medfilt(Test_L, medfiltK)
if FlagGroupFeatures:
    Train, _ = GroupFeatures(Train, GroupDict)
    Devel, _ = GroupFeatures(Devel, GroupDict)
    Test, Vocab = GroupFeatures(Test, GroupDict)

Train = np.delete(Train   , [21, 493, 517], axis=1)
Test  = np.delete(Test    , [21, 493, 517], axis=1)
Devel = np.delete(Devel   , [21, 493, 517], axis=1)
Vocab = np.array([v for i, v in enumerate(Vocab) if i not in [21, 493, 517]])
# print(Vocab,Vocab.shape)

# Run FNN, optimise network size, do feature-selection
# if ComputeSVM:
#     scores_develS = np.empty([numAnno,numGens,numMetr])
#     scores_testtS =  np.empty([numAnno,numGens,numMetr])
# if ComputeRFT:
#     scores_develR = np.empty([numAnno,numGens,numMetr])
#     scores_testtR =  np.empty([numAnno,numGens,numMetr])
if ComputeFFN:
    scores_develF = np.empty([numAnno, numGens, numMetr + 2])
    scores_testtF = np.empty([numAnno, numGens, numMetr + 2 + 8])
# if ComputeNNN:
#     scores_develN = np.empty([numAnno,numGens,numMetr])
#     scores_testtN =  np.empty([numAnno,numGens,numMetr])

def BringXption(timeregion):
    # path_Xption='/home/vedhas/workspace/panditve/workspace/Sewa/AVEC2017/AVEC_17_Emotion_Sub-Challenge/text_features/'
    # path_Xption='/home/vedhas/workspace/panditve/workspace/Sewa/AVEC2017/AVEC_17_Emotion_Sub-Challenge/transcriptions/'
    path_Xption='/home/vedhas/workspace/panditve/workspace/Sewa/AVEC2017/AVEC_17_Emotion_Sub-Challenge/text_features_xbow_6s/'

    [start, end] = timeregion
    testcumsum = np.array([0] + [a.shape[0] for a in Test_LSt]).cumsum()
    # [0 1563 5602... 145623] n+1 long,
    fileindex = np.where(start < testcumsum)[0] - 1  # (1-1)=0 or 1, ... n-1
    assert np.where(start < testcumsum)[0] == np.where(end < testcumsum)[0]
    TargetCsv=path_Xption+files_test[fileindex]

    return None


ipFeatCount = Train.shape[1]
seed = 16
selFeats = np.arange(
    ipFeatCount)
# np.array([38,75,20,25,78,30,77,44,66,67,17,37,43,50,73,19,63,32, 5,15,46,76,29, 7, 13,80,18,62,61,16,64, 6,53,42, 0,70,48,31, 1,12,23,71,36,72,54,51,22,65, 41, 3,28,45,39,47,82,52,40,57,79, 2])#np.arange(ipFeatCount)
# selReats=np.arange(ipFeatCount) #np.array([38,75,20,25,78,30,77,44,66,67,17,37,43,50,73,19,63,32, 5,15,46,76,29, 7, 13,80,18,62,61,16,64, 6,53,42, 0,70,48,31, 1,12,23,71,36,72,54,51,22,65, 41, 3,28,45,39,47,82,52,40,57,79, 2])#np.arange(ipFeatCount)
# selFeatsNew=np.ones([3,selFeats.shape[0]])
# selFeatsNew=selFeats
# selFeatsDict={0:selFeats,1:selFeats,2:selFeats}
# selReatsDict={0:selFeats,1:selReats,2:selReats}
# selFeats=np.array([                     7,                                        40, 42,     46, 48,     52, 54,         75])
# selFeats=np.array([                 6,  7,  8,  9,                                40, 42, 44, 46, 48, 50, 52, 54, 56, 63, 75])
# selFeats=np.array([ 0,  2,  3,  4,  6,  7,  8,  9, 16, 17, 18, 22, 25, 29, 37, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 60, 63, 68, 71, 87])
# selFeats=np.array([ 0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 31, 32, 34, 35, 36, 37, 38, 39, 40, 42, 44, 45, 46, 48, 50, 51, 52, 54, 56, 57, 58, 59, 60, 63, 65, 66, 68, 70, 71, 72, 73, 74, 75, 76, 77, 81, 82, 87])
# TrainO=Train[:]
# DevelO=Devel[:]
# TestO=Test[:]
with open(ResultsFile, 'a') as writefile:
    writefile.write('Features    =: ' + Lstr(path_features) + '\n')
    # writefile.write('\t\t\t\tb_lldio = '+str(b_lldio)+',\t')
    # writefile.write('b_audio = '+str(b_audio)+',\t')
    # writefile.write('b_video = '+str(b_video)+',\t')
    # writefile.write('b_text  = '+str(b_text)+'\n')
    writefile.write('#Features   =: ' + str(ipFeatCount) + '\n')
    writefile.write('Scaling     =: ' + str(typeScaling) + typeScalingDict[typeScaling] + '\n')
    writefile.write('delay       =: ' + str(delay) + '\n')
    if applymedfilt:    writefile.write('MedianFilter= ' + str(medfiltK) + '\n')
    writefile.write('mvlength    =: ' + str(mvlength) + '\n')
    writefile.write('activations  =: ' + Lstr(activations) + '\n')
    writefile.write('Num. Nodes  =: ' + Lstr(NumNodes) + '\n')
    writefile.write('Dropout     =: ' + str(DpOut) + '\n')
    writefile.write('Optimizer   =: ' + str(SelOptimizer) + '\n')
    writefile.write('LearningRate=: ' + str(lr) + '\n')
    writefile.write('epochs      =: ' + str(epochs) + '\n')
    writefile.write('#Genrations =: ' + str(numGens) + '\n')
    writefile.write('l2w         =: ' + '%.0e' % l2w + '\n')
    writefile.write('l2a         =: ' + '%.0e' % l2a + '\n')
    writefile.write('Loss        =: ' + lossW + '\n')
    writefile.write('Metric      =: ' + Lstr(metricW) + '\n')

if 1:
    print(ResultsFile)
    print('Features    =: ' + Lstr(path_features) + '\n')
    # print('\t\t\t\tb_lldio = '+str(b_lldio)+',\t')
    # print('b_audio = '+str(b_audio)+',\t')
    # print('b_video = '+str(b_video)+',\t')
    # print('b_text  = '+str(b_text)+'\n')
    print('#Features   =: ' + str(ipFeatCount) + '\n')
    print('Scaling     =: ' + str(typeScaling) + typeScalingDict[typeScaling] + '\n')
    print('delay       =: ' + str(delay) + '\n')
    if applymedfilt:    print('MedianFilter= ' + str(medfiltK) + '\n')
    print('activations  =: ' + Lstr(activations) + '\n')
    print('Num. Nodes  =: ' + Lstr(NumNodes) + '\n')
    print('Dropout     =: ' + str(DpOut) + '\n')
    print('Optimizer   =: ' + str(SelOptimizer) + '\n')
    print('LearningRate=: ' + str(lr) + '\n')
    print('epochs      =: ' + str(epochs) + '\n')
    print('#Genrations =: ' + str(numGens) + '\n')
    print('l2w         =: ' + '%.0e' % l2w + '\n')
    print('l2a         =: ' + '%.0e' % l2a + '\n')
    print('Loss        =: ' + lossW + '\n')
    print('Metric      =: ' + Lstr(metricW) + '\n')
    print('Methods     =: ' + Lstr(methods) + '\n')

SuccessRun = np.zeros((numGens, numAnno))
# CumWeights = np.zeros((numAnno, 4, ipFeatCount))
# CurWeights = np.zeros((numGens, numAnno, 4, ipFeatCount))
CumWeights = np.zeros((numGens,numAnno, len(methods), ipFeatCount))

# for curGen in range(numGens):

# ==============================================================================
#     selFeats=selFeatsNew
#     Train=TrainO[:,selFeats]
#     Devel=DevelO[:,selFeats]
#     Test=TestO[:,selFeats]
#     ipFeatCount=Train.shape[1]
#
#     scale, offset = FindScaler(Train, typeScaling=typeScaling)
#     if (not b_lldio) and (not b_audio) and (not b_video) and (b_text): Xticks=[Vocab[_] for _ in selFeats]
#
#     Train=ScaleIt(Train,scale,offset)
#     Devel=ScaleIt(Devel,scale,offset)
#     Test=ScaleIt(Test,scale,offset)
#     print(ipFeatCount)
# ==============================================================================
# for curGen in range(numGens):
#     for curAnno in range(numAnno):

# ZeroTraining
ZeroTrainingSize=20000
AverageAnno=np.median(Train_L,axis=0)
# print(Train.shape,np.zeros([ZeroTrainingSize,  Train.shape[1]]).shape)
# print(AverageAnno*np.ones([ZeroTrainingSize,Train_L.shape[1]]))
Train   = np.vstack((  Train, np.zeros([ZeroTrainingSize,  Train.shape[1]])))
Train_L = np.vstack((Train_L, AverageAnno*np.ones([ZeroTrainingSize,Train_L.shape[1]])))

# Noise addition to inpius
Train   = Train + np.random.randn(*Train.shape)/200


for curGen in range(numGens):
    for curAnno in [2]:
        while (SuccessRun[curGen, curAnno] == 0):
            # selFeats=selFeatsDict[curAnno]           #selFeatsNew
            # Train=TrainO[:,selFeats]
            # Devel=DevelO[:,selFeats]
            # Test=TestO[:,selFeats]
            ipFeatCount = Train.shape[1]
            if typeScaling:
                scale, offset = FindScaler(Train[:-ZeroTrainingSize,:], typeScaling=typeScalingInput)
                if (not b_lldio) and (not b_audio) and (not b_video) and (b_text): Xticks = [Vocab[_] for _ in selFeats]

                Train = ScaleIt(Train, scale, offset)
                Devel = ScaleIt(Devel, scale, offset)
                Test = ScaleIt(Test, scale, offset)
            print(ipFeatCount)
            with open(ResultsFile, 'a') as writefile:
                # ==============================================================================
                if ComputeSVM:
                    num_steps = 16
                    complexities = np.logspace(-15, 0, num_steps, base=2.0)
                    for comp in range(0, num_steps):
                        modsvm = svm.LinearSVR(C=complexities[comp],
                                               random_state=seed)  # C=C[curAnno],random_state=seed)
                        modsvm.fit(Train, Train_L[:, curAnno])
                        pred = modsvm.predict(Devel)
                        scores_develS[curAnno, curGen, :] = calc_scores(np.expand_dims(Devel_L[:, curAnno], axis=1),
                                                                        np.expand_dims(pred, axis=1))

                        pred = modsvm.predict(Test)
                        scores_testtS[curAnno, curGen, :] = calc_scores(np.expand_dims(Test_L[:, curAnno], axis=1),
                                                                        np.expand_dims(pred, axis=1))

                        print("\n" + Emotion[curAnno] + " SVM (CCC,PCC,RMSE): devel " + str(
                            scores_develS[curAnno, curGen, :]))
                        writefile.write("\n" + Emotion[curAnno] + " SVM [CCC,PCC,RMSE]: devel " + str(
                            scores_develS[curAnno, curGen, :]))
                        print("\n" + Emotion[curAnno] + " SVM (CCC,PCC,RMSE): test  " + str(
                            scores_testtS[curAnno, curGen, :]))
                        writefile.write("\n" + Emotion[curAnno] + " SVM [CCC,PCC,RMSE]: test  " + str(
                            scores_develS[curAnno, curGen, :]))
                        print("\n-------------------------------------")
                        writefile.write("\n-------------------------------------")
                # ==============================================================================
                if ComputeFFN:
                    seed = seed + 1
                    modffn = FFNModel(ipFeatCount, 1)
                    modffn = Model(modffn.input,modffn.output)
                    # modffn= multi_gpu_model(modffn, gpus=4)
                    modffn.compile(loss='mse', optimizer=KerasOptimiser,
                                   metrics=[dictLosses[m] for m in metric])
                    checkpointer = ModelCheckpoint(filepath=SaveDir + str(Emotion[curAnno]) + '.hdf5',
                                                   monitor='val_loss', verbose=1, save_best_only=True)
                    earlystop = EarlyStopping(monitor='val_loss', min_delta=0, patience=patienceSt, verbose=1,
                                              mode='auto', restore_best_weights=True)
                    modffn.fit(Train, Train_L[:, curAnno:curAnno + 1],
                               validation_data=(Devel, Devel_L[:, curAnno:curAnno + 1]), epochs=15,
                               verbose=verbose, batch_size=batch_size,callbacks=[ checkpointer, earlystop])
                    # modffn = load_model(SaveDir + str(Emotion[curAnno]) + '.hdf5', custom_objects={'mccc': mccc})

                    modffn.compile(loss=dictLosses[loss], optimizer=KerasOptimiser,
                                   metrics=[dictLosses[m] for m in metric])
                    checkpointer = ModelCheckpoint(filepath=SaveDir + str(Emotion[curAnno]) + '.hdf5',
                                                   monitor='val_loss', verbose=1, save_best_only=True)
                    earlystop = EarlyStopping(monitor='val_loss', min_delta=0, patience=patienceSt, verbose=1,
                                              mode='auto', restore_best_weights=True)
                    print(modffn.layers, Train.shape, Devel.shape, Test.shape)
                    print(Train_L[:, curAnno:curAnno+1].shape,Devel_L[:, curAnno:curAnno+1].shape)
                    #                fweights=FindAbsGradSummations(modffn,Train)
                    #                 SaveDir + Emo[curAnno] + str(curGen) + '.hdf5'

                    # modffn.fit(Train, movingaverage(Train_L[:, curAnno],500, 'same'),
                    #            validation_data=(Devel, movingaverage(Devel_L[:, curAnno],500, 'same')), epochs=epochs,
                    #            verbose=verbose)
                    # modffn.save('partly_trained.h5')
                    # modffn = load_model(SaveDir + str(Emotion[curAnno]) + '.hdf5',custom_objects={'mccc': mccc})
                    modffn.load_weights(SaveDir + str(Emotion[curAnno]) + '.hdf5')
                    modffn.fit(Train, Train_L[:, curAnno:curAnno+1], validation_data=(Devel, Devel_L[:, curAnno:curAnno+1]), epochs=epochs,
                               verbose=verbose,batch_size=batch_size,callbacks=[ checkpointer, earlystop])
                    modffn = load_model(SaveDir + str(Emotion[curAnno]) + '.hdf5', custom_objects={'mccc': mccc})
                    # modffn=load_model('/tmp/weights.hdf5',custom_objects={'pmse': pmse,'huber':huber}) if any(c in [loss, metric] for c in ('pmse' ,'huber')) else load_model('/tmp/weights.hdf5')
                    # '''
                    if PP:
                        scal1, offs1 = FindScaler(Train_L[:-ZeroTrainingSize, curAnno:curAnno + 1], typeScaling)
                        scal2, offs2 = FindScaler(modffn.predict(Train).flatten(), typeScaling)
                        # scal2, offs2 = FindScaler(movingaverage(modffn.predict(Train[:-ZeroTrainingSize,:]).flatten(), mvlength, 'same'), typeScaling)
                        scal, offs = [scal2 / scal1, offs2 - offs1 * scal2 / scal1]
                    else:
                        scal, offs = [np.array([1]), np.array([0])]
                    print(scal, offs )
                    if 1: #if PP:
                        originalop = modffn.get_layer('absolute_output').output
                        newoutput = Dense(units=1, activation='linear',
                                          kernel_initializer=keras.initializers.Constant(value=1.0/scal),
                                          bias_initializer=keras.initializers.Constant(value=-offs/scal))(originalop)
                        new_model = Model(modffn.input, newoutput)
                        new_model.layers[-1].set_weights([np.array([1.0/scal]),
                                                          np.array([-offs / scal]).flatten()])
                    '''
                    a10 = modffn.predict(Train)
                    a20 = modffn.predict(Devel)
                    a30 = modffn.predict(Test)
                    # a1 = new_model.predict(Train)
                    # a2 = new_model.predict(Devel)
                    # a3 = new_model.predict(Test)
                    print(a10.shape, a20.shape)

                    with tf.Session() as sess:
                        a = tf.placeholder(tf.float32, shape=[None, 1])
                        b = tf.placeholder(tf.float32, shape=[None, 1])
                        mytraloss = sess.run([mccc(a, b)], feed_dict={a: a10, b: Train_L[:, curAnno:curAnno + 1]})
                        myvalloss = sess.run([mccc(a, b)], feed_dict={a: a20, b: Devel_L[:, curAnno:curAnno + 1]})
                        mytesloss = sess.run([mccc(a, b)], feed_dict={a: a30, b: Test_L[:, curAnno:curAnno + 1]})
                        mytramse2 = sess.run([mymse2(b, a)], feed_dict={a: a10, b: Train_L[:, curAnno:curAnno + 1]})
                        myvalmse2 = sess.run([mymse2(b, a)], feed_dict={a: a20, b: Devel_L[:, curAnno:curAnno + 1]})
                        mytesmse2 = sess.run([mymse2(b, a)], feed_dict={a: a30, b: Test_L[:, curAnno:curAnno + 1]})
                        mytramse1 = sess.run([mymse1(b, a)], feed_dict={a: a10, b: Train_L[:, curAnno:curAnno + 1]})
                        myvalmse1 = sess.run([mymse1(b, a)], feed_dict={a: a20, b: Devel_L[:, curAnno:curAnno + 1]})
                        mytesmse1 = sess.run([mymse1(b, a)], feed_dict={a: a30, b: Test_L[:, curAnno:curAnno + 1]})
                        print(mytraloss, myvalloss, mytesloss)
                        print(mytramse2, myvalmse2, mytesmse2)
                    '''

                    predD1 = modffn.predict(Devel); predD1 = predD1.flatten()
                    predT1 = modffn.predict(Test); predT1 = predT1.flatten()
                    '''
                    intermediate_layer_model = Model(inputs=new_model.input,
                                                     outputs=new_model.get_layer('absolute_output').output)
                    # print(modffn.summary(),new_model.summary(),intermediate_layer_model.summary())
                    plt.figure()
                    plt.plot(range(len(predT1)),predT1,'b')
                    plt.plot(range(len(predT1)),ScaleIt(predT1, scal, offs),'g')
                    plt.plot(intermediate_layer_model.predict(Test).flatten(), 'b+-')
                    plt.plot(range(len(predT1)),new_model.predict(Test).flatten(),'g+-')
                    plt.show()
                    '''
                    '''    
                    predD = modffn.predict(Devel); predD = predD.flatten()
                    predT = modffn.predict(Test ); predT = predT.flatten()
                    '''

                    print(str(Emotion[curAnno]) + str(curGen).zfill(2) + '_DevelNoP',
                          calc_scores(np.expand_dims(Devel_L[:, curAnno], axis=1), np.expand_dims(predD1, axis=1)))
                    print(str(Emotion[curAnno]) + str(curGen).zfill(2) + '_TestNoP',
                          calc_scores(np.expand_dims(Test_L[:, curAnno], axis=1), np.expand_dims(predT1, axis=1)))
                    '''
                    if PP:
                        predD = movingaverage(predD, 11, 'same')
                        predD = ScaleIt(predD, scal, offs)
                        predT = movingaverage(predT, 11, 'same')
                        predT = ScaleIt(predT, scal, offs)
                    '''
                    '''
                    Former LRP
                    # ffnLayersList=[layers for layers in modffn.layers]
                    ffnLayersList = [layers for layers in modffn.layers if isinstance(layers, keras.layers.core.Dense)]
                    # Wlist=[layers.get_weights() for layers in ffnLayersList]
                    # print(len(Wlist))
                    # print([ item.shape for sublist in Wlist for item in sublist])
                    myWList = [layers.get_weights()[0] for layers in ffnLayersList]
                    myBList = [layers.get_weights()[1] for layers in ffnLayersList]
                    LayersList = []  # [modules.Linear(curCnt, NumNodes[0])]+NextAct
                    curCnt = ipFeatCount
                    for nid, numN in enumerate(NumNodes):
                        NextAct = [ActDict[activations[nid]]]
                        curLlayer = modules.Linear(curCnt, NumNodes[nid])  # Linear from curCnt to NumNodes[nid]
                        curLlayer.assign(myWList[nid], myBList[nid])
                        LayersList = LayersList + [curLlayer] + NextAct  # Apply activation
                        curCnt = numN
                    curLlayer = modules.Linear(curCnt, 1)  # linear NumNodes[last] to 1 output
                    curLlayer.assign(myWList[len(NumNodes)], myBList[len(NumNodes)])
                    LayersList = LayersList + [curLlayer] + [ActDict[activations[len(NumNodes)]]]
                    if PP:
                        curLlayer = modules.Linear(1, 1)  # Linear from 1 output to 1 output for scaling
                        curLlayer.assign(np.ones((1, 1)) / scal, -np.ones(1) * offs / scal)
                        LayersList = LayersList + [curLlayer] + [ActDict['linear']]
                    nn = modules.Sequential(LayersList)
                    # nn.forward(Devel)
                    # plt.figure()
                    # plt.plot(nn.forward(Devel), 'r')
                    # plt.plot(ScaleIt(modffn.predict(Devel).flatten(), scal, offs), 'g')
                    # plt.plot(Devel_L[:,curAnno],'k')
                    # plt.show()
                    # if PP:
                    #     scal1, offs1 = FindScaler(Train_L[:Zero, curAnno:curAnno + 1], 2)
                    #     scal2, offs2 = FindScaler(movingaverage(nn.forward(TrainZero).flatten(), 11, 'same'), 2)
                    #     scal, offs = [scal2 / scal1, offs2 - offs1 * scal2 / scal1]
                    # else:
                    #     scal, offs = [1,0]
                    predD = nn.forward(Devel);
                    predD = predD.flatten()
                    # plt.plot(predD, 'k')
                    '''
                    # plt.plot(range(len(predT1)),Test_L[:,curAnno], 'orange')
                    # plt.show()
                    # predT = nn.forward(Test);
                    # predT = predT.flatten()
                    predD = new_model.predict(Devel)
                    predT = new_model.predict(Test)
                    print(str(Emotion[curAnno]) + str(curGen).zfill(2) + '_Devel',
                          # calc_scores(np.expand_dims(Devel_L[:, curAnno], axis=1), np.expand_dims(predD, axis=1)))
                          calc_scores(np.expand_dims(Devel_L[:, curAnno], axis=1), predD))
                    print(str(Emotion[curAnno]) + str(curGen).zfill(2) + '_Test',
                          # calc_scores(np.expand_dims(Test_L[:, curAnno], axis=1), np.expand_dims(predT, axis=1)))
                          calc_scores(np.expand_dims(Test_L[:, curAnno], axis=1), predT))
                    # if PP:
                    #     # predD = movingaverage(predD, 11, 'same')
                    #     predD = ScaleIt(predD, scal, offs)
                    #     # predT = movingaverage(predT, 11, 'same')
                    #     predT = ScaleIt(predT, scal, offs)

                    scores_develF[curAnno, curGen, :3] = calc_scores(np.expand_dims(Devel_L[:, curAnno], axis=1),
                                                                     # np.expand_dims(predD, axis=1))
                                                                         predD)

                    scores_develF[curAnno, curGen, 3:5] = [scal,
                                                           offs]  # modffn.evaluate(Devel,Devel_L[:,curAnno],verbose=verbose)
                    SuccessRun[curGen, curAnno] = scores_develF[curAnno, curGen, 0] >= SuccessThresh[curAnno, 0]
                    print(SuccessRun[curGen, curAnno])
                    scores_testtF[curAnno, curGen, :3] = calc_scores(np.expand_dims(Test_L[:, curAnno], axis=1),
                                                                     # np.expand_dims(predT, axis=1))
                                                                    predT)
                    scores_testtF[curAnno, curGen, 3:5] = [scal,
                                                           offs]  # modffn.evaluate(Test,Test_L[:,curAnno],verbose=verbose)
                    SuccessRun[curGen, curAnno] = SuccessRun[curGen, curAnno] and scores_testtF[curAnno, curGen, 0] >= \
                                                  SuccessThresh[curAnno, 1]
                    print(SuccessRun[curGen, curAnno])
                    # print(str(Emotion[curAnno]) + str(curGen).zfill(2) + '_Devel',
                    #       scores_develF[curAnno, curGen, :])
                    # print(str(Emotion[curAnno]) + str(curGen).zfill(2) + '_Test',
                    #       scores_testtF[curAnno, curGen, :])
                    '''
                    print(predT.shape)
                    with tf.Session() as sess:
                        a = tf.placeholder(tf.float32, shape=[None])
                        b = tf.placeholder(tf.float32, shape=[None])
                        mulab=sess.run([tf.multiply(a, b)], feed_dict={a:[1,2,3],b:[4,5,6]})
                        print(mulab)
                        ghk
                        f0 = mccc(a, b)
                        f1 = mycov1(a, b)
                        f2 = mycov2(a, b)
                        f3 = mymse1(a, b)
                        f4 = mymse2(a, b)
                        # sess.run(tf.global_variables_initializer())
                        f0_result,f1_result, f2_result, f3_result, f4_result  = sess.run([f0,f1, f2, f3,f4], feed_dict={a: Test_L[:, curAnno], b:predT})
                        print(f0_result,f1_result, f2_result,f3_result,f4_result)
                    '''
                    if SuccessRun[curGen, curAnno]:
                        # plt.figure()
                        # plt.plot(range(len(predT1)), new_model.predict(Test).flatten(), 'g+-')
                        # plt.plot(range(len(predT1)), Test_L[:, curAnno], 'orange')
                        analyzers = []
                        print(new_model.summary())
                        for method in methods:
                            print(method)
                            if method=='lrp.alpha_beta':
                                analyzer = innvestigate.create_analyzer(method, new_model,
                                                                        neuron_selection_mode="index",
                                                                        alpha=2, beta=1)
                            elif method=='deep_lift.wrapper':
                                paramdict={"reference_inputs": np.zeros([1,Train.shape[1]])}
                                analyzer = innvestigate.create_analyzer(method, new_model,
                                                                        neuron_selection_mode="index",
                                                                        **paramdict
                                                                        )
                                analyzer.fit(Train, batch_size=batch_size, verbose=1)
                            else:
                                analyzer = innvestigate.create_analyzer(method, new_model,
                                                                        neuron_selection_mode="index")

                            analyzers.append(analyzer)
                        Xcomb = np.zeros([0, Train.shape[1]])
                        Ycomb = np.zeros([0, Train_L.shape[1]])
                        for splitId, splits in enumerate([[Train, Train_L], [Devel, Devel_L], [Test, Test_L]]):
                            if splitId in ExplainWhat:
                                Xcomb = np.vstack((Xcomb, splits[0]))
                                Ycomb = np.vstack((Ycomb, splits[1]))
                        Pcomb = np.squeeze(new_model.predict(Xcomb))  # Prediction
                        Ecomb = np.abs(Ycomb[:, curAnno] - Pcomb)  # Error
                        Mcomb = np.sqrt(np.mean(np.square(Ecomb)))
                        assert Mcomb.size==1
                        MaxAllowedError = np.mean(Ecomb)   #MaxErrorAllowed for good samples= [mean-std](Error)
                        test_sample_indices = np.where((np.abs(Ecomb)) < MaxAllowedError)[0]  # Select samples for weight
                        print(MaxAllowedError)
                        # test_sample_indices2 = GoodRegions(test_sample_indices,10)
                        # contlist=ContRegions(test_sample_indices,10,60)
                        # # print(test_sample_indices)
                        # print(MaxAllowedError, contlist)
                        # test_sample_indices3=[]
                        # test_sample_indices3 = np.array([ citem for a,b in contlist for citem in range(a,b)])
                        test_sample_indices3 = test_sample_indices
                        # print(test_sample_indices3)
                        # print(test_sample_indices ,contlist)
                        # for stend in contlist:
                        #     # print(stend[1],stend[0])
                        #     mpatches.Rectangle((stend[0],-0.2),stend[1]-stend[0],1.2,
                        #                       linewidth=1,edgecolor='r',facecolor='r',alpha=0.5)
                        # print(MaxAllowedError, test_sample_indices)
                        # plt.show()
                        test_sample_preds = [None] * len(test_sample_indices)
                        MAX_SEQ_LENGTH=len(Vocab)
                        # analysis = np.zeros([len(test_sample_indices), len(analyzers), 1, MAX_SEQ_LENGTH])
                        for aidx, analyzer in enumerate(analyzers):
                            curTest = Test[test_sample_indices, :]
                            # curTest=np.expand_dims(curTest,axis=-1)
                            print(curTest.shape)
                            # curTest = curTest[None,:,:]
                            print(curTest.shape)
                            myanalysis=analyzer.analyze(curTest,0)
                            myanalysis=myanalysis.mean(axis=0)
                            smyanalysis = np.sort(myanalysis)
                            amyanalysis = np.array(np.argsort(myanalysis))

                            myanalysis2o = analyzer.analyze(Test[test_sample_indices3,:], 0)
                            print(myanalysis2o.shape)
                            myanalysis2 = myanalysis2o.mean(axis=0)
                            smyanalysis2 = np.sort(myanalysis2)
                            amyanalysis2 = np.argsort(myanalysis2)

                            ############################################################
                            ########     Double Column Plot    #########################
                            ############################################################
                            N = 6
                            cols = 2
                            rows = int(math.ceil(N / cols))
                            gs = gridspec.GridSpec(rows, cols)
                            print([_ for _ in gs])
                            fig = plt.figure()
                            ax = fig.add_subplot(gs[0])
                            ax.plot(range(len(predT)), Test_L[:, curAnno],
                                    label='Gold Standard')
                            ax.plot(range(len(predT)), new_model.predict(Test).flatten(),
                                    label='Prediction')
                            sns.despine(ax=ax)
                            ax.legend(loc = 'upper right')
                            for j in range(1, rows):
                                ax1 = fig.add_subplot(gs[j*2])
                                ax1._get_lines.prop_cycler = ax._get_lines.prop_cycler
                                TestFWithNan = np.empty(Test.shape[0]) * np.nan
                                ScoreWithNan = np.empty(Test.shape[0]) * np.nan
                                TestFWithNan[test_sample_indices3] = Test[test_sample_indices3, amyanalysis2[-1 * j]]
                                ScoreWithNan[test_sample_indices3] = myanalysis2o[:, amyanalysis2[-1 * j]]
                                # ax1.plot(range(Test.shape[0]), TestFWithNan, label=r"Input Feature="r"$log_{10}t_f$")
                                ax1.plot(range(Test.shape[0]), TestFWithNan,
                                         label=r"$log_{10}tf$" + "('" + Vocab[amyanalysis2[-j]] + "')")
                                ax1a = ax1#.twinx()
                                ax1a._get_lines.prop_cycler = ax1._get_lines.prop_cycler
                                # ax1a._get_lines.color_cycle.next()
                                ax1a.plot(range(Test.shape[0]), ScoreWithNan,
                                          label="Relevance('" + Vocab[amyanalysis2[-j]] + "')")
                                ax1.legend(loc='upper right')
                                # ax1.set_xlabel('Sample Number = time(s)*10')
                                # ax1.set_ylabel('Feature: '
                                # ax1a.legend()
                                # ax1.legend([Vocab[amyanalysis2[-1 * j]]])
                                ax1.get_shared_x_axes().join(ax1, ax)
                                sns.despine(ax=ax1)
                                # sns.despine(ax=ax)
                            ax2 = fig.add_subplot(gs[1])
                            ax2.plot(range(len(predT)), Test_L[:, curAnno],
                                    label = 'Gold Standard')
                            ax2.plot(range(len(predT)), new_model.predict(Test).flatten(),
                                     label='Prediction')
                            ax2.legend(loc='upper right')
                            sns.despine(ax=ax2)
                            ax2.get_shared_x_axes().join(ax2, ax)
                            ax2.get_shared_y_axes().join(ax2, ax)
                            for j in range(0, rows-1):
                                ax3 = fig.add_subplot(gs[2*j+3])
                                ax3._get_lines.prop_cycler = ax2._get_lines.prop_cycler
                                TestFWithNan = np.empty(Test.shape[0]) * np.nan
                                ScoreWithNan = np.empty(Test.shape[0]) * np.nan
                                TestFWithNan[test_sample_indices3] = Test[test_sample_indices3, amyanalysis2[j]]
                                ScoreWithNan[test_sample_indices3] = myanalysis2o[:, amyanalysis2[j]]
                                ax3.plot(range(Test.shape[0]), TestFWithNan,
                                         label=r"$log_{10}tf$" + "('" + Vocab[amyanalysis2[j]] + "')")
                                ax3a = ax3#.twinx()
                                # ax3a._get_lines.color_cycle.next()
                                ax3a._get_lines.prop_cycler = ax3._get_lines.prop_cycler
                                ax3a.plot(range(Test.shape[0]), 20*ScoreWithNan,
                                          label="Relevance('" + Vocab[amyanalysis2[j]] + "')*20")
                                # ax3.set_xlabel('Sample Number = time(s)*10')
                                ax3.legend(loc='upper right')
                                # ax3a.legend()
                                # ax3.legend([Vocab[amyanalysis2[j]]])
                                ax3.get_shared_x_axes().join(ax3, ax2)
                                sns.despine(ax=ax3)
                            fig.suptitle(methods[aidx]+" scores for "+Emotion[curAnno])
                            fig.tight_layout()
                            plt.tight_layout()
                            if curAnno==0:
                                plt.xlim(13900,17200)
                            ############################################################
                            ########     Single Column Plot    #########################
                            ############################################################
                            fig = plt.figure()
                            ax = fig.add_subplot(111)
                            ax.plot(range(len(predT)), Test_L[:, curAnno],
                                    label='Gold Standard')
                            ax.plot(range(len(predT)), new_model.predict(Test).flatten(),
                                    label='Prediction')
                            sns.despine(ax=ax)
                            ax.legend(loc = 'upper right')
                            for j in [-1,-2,0]:     #Most +ve1, most +ve2, most -ve1
                                n = len(fig.axes)
                                for i in range(n):
                                    fig.axes[i].change_geometry(n + 1, 1, i + 1)
                                ax1 = fig.add_subplot(n + 1, 1, n + 1)
                                ax1._get_lines.prop_cycler = ax._get_lines.prop_cycler
                                TestFWithNan = np.empty(Test.shape[0]) * np.nan
                                ScoreWithNan = np.empty(Test.shape[0]) * np.nan
                                TestFWithNan[test_sample_indices3] = Test[test_sample_indices3, amyanalysis2[j]]
                                ScoreWithNan[test_sample_indices3] = myanalysis2o[:, amyanalysis2[j]]
                                ax1.plot(range(Test.shape[0]), TestFWithNan,
                                         label=r"$log_{10}tf$" + "('" + Vocab[amyanalysis2[j]] + "')")
                                ax1a = ax1#.twinx()
                                ax1a._get_lines.prop_cycler = ax1._get_lines.prop_cycler
                                if j>=0:
                                    ax1a.plot(range(Test.shape[0]), ScoreWithNan*3,
                                              label="Relevance('" + Vocab[amyanalysis2[j]] + "')*3")
                                else:
                                    ax1a.plot(range(Test.shape[0]), ScoreWithNan*3,
                                              label="Relevance('" + Vocab[amyanalysis2[j]] + "')*3")
                                ax1.legend(loc='upper right')
                                ax1.get_shared_x_axes().join(ax1, ax)
                                sns.despine(ax=ax1)
                            fig.suptitle(methods[aidx]+" scores for "+Emotion[curAnno])
                            fig.tight_layout()
                            plt.tight_layout()
                            if curAnno==0:
                                ax.set_xlim(13900,17200)
                                ax.set_ylim(-0.2,0.625)
                            ############################################################
                            ########     Single Column Plot  Selected words  ###########
                            ############################################################
                            selectwords = ['gut', 'ganz', 'langweilig']
                            wid1, wid2, wid3 = [list(Vocab[amyanalysis2]).index(word) for word in selectwords]

                            fig = plt.figure()
                            ax = fig.add_subplot(111)
                            ax.plot(range(len(predT)), Test_L[:, curAnno],
                                    label='Gold Standard')
                            ax.plot(range(len(predT)), new_model.predict(Test).flatten(),
                                    label='Prediction')
                            sns.despine(ax=ax)
                            ax.legend(loc = 'upper right')
                            for j in [-1,0,1]:#[wid1,wid2,wid3]:     #Most +ve1, most +ve2, most -ve1
                                n = len(fig.axes)
                                for i in range(n):
                                    fig.axes[i].change_geometry(n + 1, 1, i + 1)
                                ax1 = fig.add_subplot(n + 1, 1, n + 1)
                                ax1._get_lines.prop_cycler = ax._get_lines.prop_cycler
                                TestFWithNan = np.empty(Test.shape[0]) * np.nan
                                ScoreWithNan = np.empty(Test.shape[0]) * np.nan
                                TestFWithNan[test_sample_indices3] = Test[test_sample_indices3, amyanalysis2[j]]
                                ScoreWithNan[test_sample_indices3] = myanalysis2o[:, amyanalysis2[j]]
                                ax1.plot(range(Test.shape[0]), TestFWithNan,
                                         label=r"$log_{10}tf$" + "('" + Vocab[amyanalysis2[j]] + "')")
                                ax1a = ax1#.twinx()
                                ax1a._get_lines.prop_cycler = ax1._get_lines.prop_cycler
                                if j >= 0:
                                    ax1a.plot(range(Test.shape[0]), ScoreWithNan*3,
                                              label="Relevance('" + Vocab[amyanalysis2[j]] + "')*3")
                                else:
                                    ax1a.plot(range(Test.shape[0]), ScoreWithNan*3,
                                              label="Relevance('" + Vocab[amyanalysis2[j]] + "')*3")
                                # ax1a.plot(range(Test.shape[0]), ScoreWithNan,
                                #               label="Relevance('" + Vocab[amyanalysis2[j]] + "')")
                                ax1.legend(loc='upper right')
                                ax1.get_shared_x_axes().join(ax1, ax)
                                sns.despine(ax=ax1)
                            fig.suptitle(methods[aidx]+" scores for "+Emotion[curAnno])
                            fig.tight_layout()
                            plt.tight_layout()
                            if curAnno==0:
                                ax.set_xlim(13900,17200)
                                ax.set_ylim(-0.2,0.625)
                            '''
                            fig = plt.figure()
                            ax = fig.add_subplot(121)
                            ax.plot(range(len(predT1)), new_model.predict(Test).flatten(), 'g+-')
                            ax.plot(range(len(predT1)), Test_L[:, curAnno], 'orange')
                            for j in range(1,4):
                                print(fig.axes)
                                n = len(fig.axes)
                                for i in range(n):
                                    fig.axes[i].change_geometry(n + 1, 1, i + 1)
                                ax1 = fig.add_subplot(n + 1, 1, n + 1)
                                TestFWithNan = np.empty(Test.shape[0]) * np.nan
                                ScoreWithNan = np.empty(Test.shape[0]) * np.nan
                                TestFWithNan[test_sample_indices3] = Test[test_sample_indices3, amyanalysis2[-1 * j]]
                                ScoreWithNan[test_sample_indices3] = myanalysis2o[:,amyanalysis2[-1*j]]
                                # ax1.scatter(test_sample_indices3,Test[test_sample_indices3,amyanalysis2[-1*j]])
                                # ax1.scatter(test_sample_indices3, myanalysis2o[:,amyanalysis2[-1*j]])
                                # ax1.plot(test_sample_indices3, Test[test_sample_indices3, amyanalysis2[-1 * j]])
                                # ax1.plot(test_sample_indices3, myanalysis2o[:, amyanalysis2[-1 * j]])
                                ax1.plot(range(Test.shape[0]), TestFWithNan)
                                ax1.plot(range(Test.shape[0]), ScoreWithNan)
                                ax1.legend([Vocab[amyanalysis2[-1*j]]])
                                ax1.get_shared_x_axes().join(ax1, ax)
                            ax2 = fig.add_subplot(122)
                            ax2.plot(range(len(predT1)), new_model.predict(Test).flatten(), 'g+-')
                            ax2.plot(range(len(predT1)), Test_L[:, curAnno], 'orange')
                            for j in range(0, 3):
                                n = len(fig.axes)
                                for i in range(n):
                                    fig.axes[i].change_geometry(n + 1, 2, i + 1)
                                ax3 = fig.add_subplot(n + 1, 2, n + 1)
                                TestFWithNan = np.empty(Test.shape[0]) * np.nan
                                ScoreWithNan = np.empty(Test.shape[0]) * np.nan
                                TestFWithNan[test_sample_indices3] = Test[test_sample_indices3, amyanalysis2[j]]
                                ScoreWithNan[test_sample_indices3] = myanalysis2o[:, amyanalysis2[j]]
                                # ax1.scatter(test_sample_indices3,Test[test_sample_indices3,amyanalysis2[-1*j]])
                                # ax1.scatter(test_sample_indices3, myanalysis2o[:,amyanalysis2[-1*j]])
                                # ax1.plot(test_sample_indices3, Test[test_sample_indices3, amyanalysis2[-1 * j]])
                                # ax1.plot(test_sample_indices3, myanalysis2o[:, amyanalysis2[-1 * j]])
                                ax3.plot(range(Test.shape[0]), TestFWithNan)
                                ax3.plot(range(Test.shape[0]), ScoreWithNan)
                                ax3.legend([Vocab[amyanalysis2[j]]])
                                ax3.get_shared_x_axes().join(ax3, ax2)
                            fig.tight_layout()
                            '''

                            '''
                            plt.figure()
                            # print(amyanalysis,amyanalysis.shape,)
                            plt.plot(np.arange(len(Vocab)), smyanalysis)
                            plt.xticks(np.arange(len(Vocab)), Vocab[amyanalysis],rotation='vertical')
                            plt.title(methods[aidx])
                            plt.tight_layout()
                            '''
                            plt.figure()
                            plt.plot(np.arange(len(Vocab)), smyanalysis2)
                            plt.xticks(np.arange(len(Vocab)), Vocab[amyanalysis2], rotation='vertical')
                            plt.title(methods[aidx])
                            plt.tight_layout()
                            # plt.show()

                            CumWeights[curGen,curAnno, aidx, :] = myanalysis2

                            print("\n" +
                                  str(Emotion[curAnno]) + ':' + str(aidx) + ':\t' +
                                  str([Vocab[_] for _ in selFeats[(-CumWeights[curGen, curAnno, aidx, :]).argsort()]]))
                            # print("\n" +
                            #       str(Emotion[curAnno]) + ':' + str(aidx) + ':\t' +
                            #       str(
                            #           CumWeights[curGen, curAnno, aidx, (-CumWeights[curGen, curAnno, aidx, :]).argsort()]
                            #       ))

                            writefile.write("\n1Words0_" +
                                            str(Emotion[curAnno]) + ':' + str(aidx) + ':\t' +
                                            str([Vocab[_] for _ in
                                                 selFeats[(-CumWeights[curGen, curAnno, aidx, :]).argsort()]]))
                            writefile.write("\n2Value0_" +
                                            str(Emotion[curAnno]) + ':' + str(aidx) + ':\t')
                            np.savetxt(writefile, CumWeights[
                                curGen, curAnno, aidx, (-CumWeights[curGen, curAnno, aidx, :]).argsort()],
                                       newline=',', delimiter=',')

                            writefile.write("\n1Words1_" +
                                            str(Emotion[curAnno]) + ':' + str(aidx) + ':\t' +
                                            str([_ for _ in Vocab]))
                            writefile.write("\n2Value1_" +
                                            str(Emotion[curAnno]) + ':' + str(aidx) + ':\t')
                            np.savetxt(writefile, CumWeights[curGen, curAnno, aidx, :],
                                       newline=',', delimiter=',')

                            # print(str([Vocab[_] for _ in selFeats[(-CumWeights[curAnno, aidx, :]).argsort()]]))
                            # print(myanalysis2.shape, CumWeights[curAnno, aidx, :], curAnno, aidx)
                        plt.show()
                        '''
                        for i, ridx in enumerate(test_sample_indices):
                            # x, y = DATASETS['testing']['x4d'][ridx], DATASETS['testing']['y'][ridx]
                            x, y = Test[ridx,:], Test_L[ridx, curAnno]
                            t_start = time.time()
                            # x = x.reshape((1, MAX_SEQ_LENGTH))
                            # presm = new_model.predict_on_batch(x)#[0]  # forward pass without softmax
                            # prob = new_model.predict_on_batch(x)#[0]  # forward pass with softmax
                            # y_hat = prob.argmax()
                            # test_sample_preds[i] = y_hat
                            myanalysis = analyzer.analyze(Test[test_sample_indices], 0)
                            myanalysis = myanalysis.sum(axis=0)
                            plt.figure()
                            smyanalysis = np.sort(myanalysis)
                            amyanalysis = np.argsort(myanalysis).tolist()
                            print('amyanalysis', amyanalysis)
                            plt.plot(np.arange(len(Vocab)), smyanalysis)
                            plt.xticks(np.arange(len(Vocab)), Vocab[amyanalysis], rotation='vertical')
                            plt.title(methods[aidx])
                            plt.tight_layout()
                            plt.show()
                            for aidx, analyzer in enumerate(analyzers):
                                myanalysis=analyzer.analyze(Test[test_sample_indices],0)
                                # myanalysis=myanalysis/myanalysis.sum(axis=1).reshape((-1,1))
                                myanalysis=myanalysis.sum(axis=0)
                                print('myanalysis',myanalysis)
                                # plt.figure()
                                # plt.plot(np.arange(len(Vocab)),myanalysis[0])
                                # plt.xticks(np.arange(len(Vocab)),Vocab,rotation='vertical')
                                # plt.show()
                                #
                                plt.figure()
                                smyanalysis = np.sort(myanalysis)
                                amyanalysis = np.argsort(myanalysis).tolist()
                                print('amyanalysis',amyanalysis)
                                plt.plot(np.arange(len(Vocab)), smyanalysis)
                                plt.xticks(np.arange(len(Vocab)), Vocab[amyanalysis],rotation='vertical')
                                plt.title(methods[aidx])
                                plt.tight_layout()
                                plt.show()
                                # a = np.squeeze(analyzer.analyze(x))
                                # a = np.sum(a, axis=1)
                                # analysis[i, aidx] = a
                                # print(a)
                        
                            t_elapsed = time.time() - t_start
                            print('Review %d (%.4fs)' % (ridx, t_elapsed))
                        # This is a utility method visualizing the relevance scores of each word to the network's prediction.
                        # one might skip understanding the function, and see its output first.
                        def plot_text_heatmap(words, scores, title="", width=10, height=0.2, verbose=0,
                                              max_word_per_line=20):
                            fig = plt.figure(figsize=(width, height))

                            ax = plt.gca()

                            ax.set_title(title, loc='left')
                            tokens = words
                            if verbose > 0:
                                print('len words : %d | len scores : %d' % (len(words), len(scores)))

                            cmap = plt.cm.ScalarMappable(cmap=cm.bwr)
                            cmap.set_clim(0, 1)

                            canvas = ax.figure.canvas
                            t = ax.transData

                            # normalize scores to the followings:
                            # - negative scores in [0, 0.5]
                            # - positive scores in (0.5, 1]
                            normalized_scores = 0.5 * scores / np.max(np.abs(scores)) + 0.5

                            if verbose > 1:
                                print('Raw score')
                                print(scores)
                                print('Normalized score')
                                print(normalized_scores)

                            # make sure the heatmap doesn't overlap with the title
                            loc_y = -0.2

                            for i, token in enumerate(tokens):
                                *rgb, _ = cmap.to_rgba(normalized_scores[i], bytes=True)
                                color = '#%02x%02x%02x' % tuple(rgb)

                                text = ax.text(0.0, loc_y, token,
                                               bbox={
                                                   'facecolor': color,
                                                   'pad': 5.0,
                                                   'linewidth': 1,
                                                   'boxstyle': 'round,pad=0.5'
                                               }, transform=t)

                                text.draw(canvas.get_renderer())
                                ex = text.get_window_extent()

                                # create a new line if the line exceeds the length
                                if (i + 1) % max_word_per_line == 0:
                                    loc_y = loc_y - 2.5
                                    t = ax.transData
                                else:
                                    t = transforms.offset_copy(text._transform, x=ex.width + 15, units='dots')

                            if verbose == 0:
                                ax.axis('off')
                        plot_text_heatmap(
                            "I really love this movie but not in the beginning".split(' '),
                            np.array([0.02, 0.2, 0.5, 0.1, 0.1, 0.1, -0.2, 0.05, 0.00, 0.08])
                        )
                        # "love" is shaded with strong red because its relevance score is rather high
                        # "not" is highlighted in light blue because of its negative score.

                        # Traverse over the analysis results and visualize them.
                        for i, idx in enumerate(test_sample_indices):
                            words = [decoder[t] for t in list(DATASETS['testing']['encoded_reviews'][idx])]
                            print('Review(id=%d): %s' % (idx, ' '.join(words)))
                            y_true = DATASETS['testing']['y'][idx]
                            y_pred = test_sample_preds[i]
                            print("Pred class : %s %s" %
                                  (LABEL_IDX_TO_NAME[y_pred],
                                   '✓' if y_pred == y_true else '✗ (%s)' % LABEL_IDX_TO_NAME[y_true])
                                  )
                            for j, method in enumerate(methods):
                                plot_text_heatmap(words, analysis[i, j].reshape(-1), title='Method: %s' % method,
                                                  verbose=0)
                                plt.show()
                        '''
                        '''
                        # -----------------------------------
                        print("\n" + str(Emotion[curAnno]) + str(curGen).zfill(2) + ':' +
                              " FFN (CCC,PCC,RMSE)[loss, metric]: devel " + Lstr(scores_develF[curAnno, curGen, :]))
                        writefile.write("\n0Perfo_" + str(Emotion[curAnno]) + Lstr(curGen).zfill(2) + ': Devel' +
                                        " FFN : " + Lstr(scores_develF[curAnno, curGen, :], ','))
                        Xcomb = np.zeros([0, Train.shape[1]])
                        Ycomb = np.zeros([0, Train_L.shape[1]])
                        for splitId, splits in enumerate([[Train, Train_L], [Devel, Devel_L], [Test, Test_L]]):
                            if splitId in ExplainWhat:
                                Xcomb = np.vstack((Xcomb, splits[0]))
                                Ycomb = np.vstack((Ycomb, splits[1]))
                        Pcomb = np.squeeze(nn.forward(Xcomb))  # Prediction
                        # if PP:
                        # Pcomb = movingaverage(Pcomb, 11, 'same')
                        # Pcomb = ScaleIt(Pcomb, scal, offs)
                        Ecomb = np.abs(Ycomb[:, curAnno] - Pcomb)  # Error
                        MaxAllowedError = np.median(
                            Ecomb)  # -np.std(Ecomb)    #MaxErrorAllowed for good samples= [mean-std](Error)
                        selIndices = np.where((np.abs(Ecomb)) < MaxAllowedError)  # Select samples for weight
                        xs = Xcomb[selIndices]  # [0:5,:]
                        ys = Ycomb[selIndices][:, curAnno]  # [0:5,:]
                        print(xs.shape, ys.shape, MaxAllowedError)
                        scores_testtF[curAnno, curGen, 5:] = [Xcomb.shape[0],
                                                              xs.shape[0],
                                                              MaxAllowedError,
                                                              np.mean(Ecomb),
                                                              np.std(Ecomb),
                                                              np.min(Ecomb),
                                                              np.max(Ecomb),
                                                              np.max(Ecomb) - np.min(Ecomb)]
                        print("\n" + Emotion[curAnno] + " FFN [CCC,PCC,RMSE][loss, metric]: test  " + str(
                            scores_testtF[curAnno, curGen, :]))
                        writefile.write("\n0Perfo_" + str(Emotion[curAnno]) + str(curGen).zfill(2) + ': Test' +
                                        " FFN : " + Lstr(scores_testtF[curAnno, curGen, :], ','))
                        # xs=Test
                        # ys=Test_L[:,curAnno]
                        ypred = nn.forward(
                            xs)  # [np.random.choice(xs.shape[0], np.max(3000, int(0.02 * xs.shape[0])))])
                        # del selIndices, Xcomb,Ycomb,Pcomb,Ecomb, MaxAllowedError,xs,ys
                        # ypred = nn.forward(xs[np.random.choice(xs.shape[0], np.max(3000,int(0.02 * xs.shape[0]))   )])
                        # ypred=nn.forward(xs[np.random.choice(xs.shape[0],int(0.02*xs.shape[0]))])
                        # ypred=snn.predict(xs[np.random.choice(xs.shape[0],int(0.2*xs.shape[0]))])

                        # Rsum = np.zeros((4, ipFeatCount))
                        # print(nn.lrp(ypred, 'alphabeta', 0.5).shape, ypred.shape)
                        # print((nn.lrp(ypred, 'alphabeta', 0.5)/ypred).shape)

                        # Rsum[0, :] = nn.lrp(ypred).sum(axis=0)
                        # Rsum[1, :] = (nn.lrp(ypred) / ypred).sum(axis=0)
                        # Rsum[2, :] = nn.lrp(ypred, 'epsilon', 1.).sum(axis=0)
                        # Rsum[3, :] = (nn.lrp(ypred, 'epsilon', 1.) / ypred).sum(axis=0)
                        # LaughterWeights = (nn.lrp(ypred, 'epsilon', 1.) / ypred)[:, Vocab.index("<LAUGHTER>")]
                        # print(LaughterWeights.shape)
                        '''
                        '''
                        fig = plt.figure()
                        host = fig.add_subplot(111)
                        # par0 = host.twinx()
                        # par1 = host.twinx()
                        # par2 = host.twinx()
                        par3 = host.twinx()
                        host.set_xlim(0, xs.shape[0])
                        host.set_ylim(0, 0.45)
                        # host.set_ylim(np.min(xs[:,463]), np.max(xs[:,463]))
                        # par0.set_ylim(np.min(ys), np.max(ys))
                        # par1.set_ylim(np.min(ypred), np.max(ypred))
                        # par2.set_ylim(np.min(Ecomb[selIndices]), np.max(Ecomb[selIndices]))
                        # par3.set_ylim(np.min(LaughterWeights), np.max(LaughterWeights)*1.1)
                        par3.set_ylim(0, np.max(LaughterWeights) * 1.5)
                        # par3.set_ylim(0, 0.0015)
                        host.set_xlabel("Time")
                        host.set_ylabel("log(1+tf(<laughter>))")
                        # par0.set_ylabel("Labels")
                        # par1.set_ylabel("Prediction")
                        # par2.set_ylabel("Prediction Error")
                        par3.set_ylabel("Laughter Weights")
                        h0, = host.plot(range(xs.shape[0]), xs[:, 463], 'r', label='log(1+tf(<Laughter>))')
                        # p0, = par0.plot(np.arange(xs.shape[0]),ys, 'r', label='Labels')
                        # p1, = par1.plot(np.arange(xs.shape[0]),ypred, 'g', label='Prediction')
                        # p2, = par2.plot(np.arange(xs.shape[0]),Ecomb[selIndices], 'k', label='Prediction Error')
                        p3, = par3.plot(range(xs.shape[0]), np.round(LaughterWeights, 3), 'b',
                                        label='<Laughter> Weights')
                        # lns = [h0,p0,p1, p2, p3]
                        lns = [h0, p3]
                        host.legend(handles=lns, loc='best')
                        host.xaxis.set_ticks([])
                        host.yaxis.label.set_color(h0.get_color())
                        # par0.yaxis.label.set_color(p0.get_color())
                        # par1.yaxis.label.set_color(p1.get_color())
                        # par2.yaxis.label.set_color(p2.get_color())
                        par3.yaxis.label.set_color(p3.get_color())
                        plt.show()
                        '''
                        '''
                        # plt.plot(xs[:,463], 'r')
                        # plt.plot(ys,'r')
                        # plt.plot(ypred,'g')
                        # plt.plot(Ecomb, 'k')
                        # plt.plot(LaughterWeights,'b')
                        # plt.show()
                        del selIndices, Xcomb, Ycomb, Pcomb, Ecomb, MaxAllowedError, xs, ys
                        # Rsum[4, :] = nn.lrp(ypred, 'alphabeta', 2.).sum(axis=0)
                        # Rsum[5, :] = (nn.lrp(ypred, 'alphabeta', 2.)/ ypred).sum(axis=0)
                        # Rsum[3, :] = (nn.lrp(ypred, 'flat') / ypred).sum(axis=0)
                        # Rsum[4, :] = (nn.lrp(ypred, 'ww') / ypred).sum(axis=0)
                    #     for wtId in range(Rsum.shape[0]):
                    #         CurWeights[curGen, curAnno, wtId, :]=Rsum[wtId,:]
                    #         CumWeights[curAnno, wtId,:] = CumWeights[curAnno, wtId,:] + CurWeights[curGen, curAnno, wtId,:]
                    #         print("\n"+
                    #             str(Emotion[curAnno])+str(curGen).zfill(2)+':'+str(wtId)+':\t'+
                    #             str([Vocab[_] for _ in selFeats[(-CurWeights[curGen, curAnno, wtId,:]).argsort()]]))
                    #         writefile.write("\n1Words_"+
                    #             str(Emotion[curAnno]) + str(curGen).zfill(2) + ':' + str(wtId) + ':\t' +
                    #             str([Vocab[_] for _ in selFeats[(-CurWeights[curGen, curAnno, wtId, :]).argsort()]]))
                    #         writefile.write("\n2Value_" +
                    #                         str(Emotion[curAnno]) + str(curGen).zfill(2) + ':' + str(wtId) + ':\t')
                    #                         # + str(CurWeights[curGen, curAnno, wtId, :]))
                    #         np.savetxt(writefile,
                    #                    CurWeights[curGen, curAnno, wtId, (-CurWeights[curGen, curAnno, wtId, :]).argsort()],
                    #                     newline=',',delimiter=',')
                    #
                    #         print("\n" +
                    #               str(Emotion[curAnno])  + ':' + str(wtId) + ':' +
                    #               str([Vocab[_] for _ in selFeats[(-CumWeights[curAnno, wtId, :]).argsort()]]))
                    #         writefile.write("\n1Words0_" +
                    #                         str(Emotion[curAnno])  + ':' + str(wtId) + ':' +
                    #                         str([Vocab[_] for _ in
                    #                              selFeats[(-CumWeights[curAnno, wtId, :]).argsort()]]))
                    #         writefile.write("\n2Value0_" +
                    #                         str(Emotion[curAnno])  + ':' + str(wtId) + ':\t')
                    #         np.savetxt(writefile,
                    #                    CumWeights[curAnno, wtId, (-CumWeights[ curAnno, wtId, :]).argsort()],
                    #                    newline=',', delimiter=',')
                    #
                    #         writefile.write("\n1Words1_" +
                    #                         str(Emotion[curAnno]) + ':' + str(wtId) + ':\t' +
                    #                         str([_ for _ in Vocab]))
                    #         writefile.write("\n2Value1_" +
                    #                         str(Emotion[curAnno]) + ':' + str(wtId) + ':\t')
                    #         np.savetxt(writefile, CumWeights[curAnno, wtId, :],
                    #                    newline=',', delimiter=',')
                    #     del Rsum,  ypred,splitId, splits
                    # del modffn, nn, ModelName, LayersList, myWList,myBList, curCnt,ffnLayersList,NextAct,curLlayer,scal, offs
                    '''
# print(CumWeights)
with open(ResultsFile, 'a') as writefile:
    writefile.write("\n\n\n")
    for curAnno in range(numAnno):
      for curGen in range(numGens):
        for wtId in range(CumWeights.shape[2]):
            print("\n" +
                  str(Emotion[curAnno]) + ':' + str(wtId) + ':\t' +
                  str([Vocab[_] for _ in selFeats[(-CumWeights[curGen,curAnno, wtId, :]).argsort()]]))
            writefile.write("\n1Words0_" +
                            str(Emotion[curAnno]) + ':' + str(wtId) + ':\t' +
                            str([Vocab[_] for _ in selFeats[(-CumWeights[curGen,curAnno, wtId, :]).argsort()]]))
            writefile.write("\n2Value0_" +
                            str(Emotion[curAnno]) + ':' + str(wtId) + ':\t')
            np.savetxt(writefile, CumWeights[curGen, curAnno, wtId, (-CumWeights[curGen,curAnno, wtId, :]).argsort()],
                       newline=',', delimiter=',')

            writefile.write("\n1Words1_" +
                            str(Emotion[curAnno]) + ':' + str(wtId) + ':\t' +
                            str([_ for _ in Vocab]))
            writefile.write("\n2Value1_" +
                            str(Emotion[curAnno]) + ':' + str(wtId) + ':\t')
            np.savetxt(writefile, CumWeights[curGen,curAnno, wtId, :],
                       newline=',', delimiter=',')
print("Run Complete!")

'''
GroupDict = {
        'at': [7, 8],
        # 'sehen':      [11,12,229,464],
        # 'because':  [18,24,110,186],
        'articles': [21, 493, 517],
        'hallo': [236, 71, 332],
        # 'yes':      [36,90],    #324
        'filler': [118, 136, 171],  # 123,160,
        # 'must':     [175,241,148,272],#166,
        # 'hear':     [17,188],
        'boring': [201, 450],
        # 'ein':      [130,131,132,100,114],
        'in': [83, 84],
        'all': [184, 187, 224],
        'ganz': [299, 346, 502, 138],
        # 'has':      [437,345,95,461],
        'diese': [102, 280, 281, 292, 294],
        # 'weiss':    [113,333],
        'jede': [349, 350],
        'irgend': [105, 116, 164, 177, 501],
        'gut': [364, 520],
        # 'meinst':   [227,325,402],
        'mein': [330, 492],
        # 'komm':     [400,206],
        'komisch': [486, 60],
        # 'mach':     [413,127,153,319],
        'andere': [234, 257],
        'beispiel': [194, 301, 406],
        'best': [79, 363],
        # 'no':       [91,379], #remove ne,na 128, 129,
        'schlecht': [69, 159],
        'gleich': [232, 465],
        'hand': [14, 275, 16],
        'lang': [19, 137],
        'letzte': [399, 246],
        'situation': [419, 23],
        'viel': [307, 209],
        'verschiedener': [221, 225],
        'tap': [318, 488, 336],
        'advertisement': [142, 143, 167, 375, 156]
    }
'''

'''
                                        % 
                                       / \___,      __
                                    ../       \  __/  \
                                               \/           
graph input1
graph input1*gradient1 or scaore1
graph input2
graph input2*gradient2 or scaore2
transcription           
'''