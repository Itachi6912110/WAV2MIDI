import torch
from torch import nn
from torch.autograd import Variable
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import torchvision.models as models
import torch.utils.data as data_utils
import matplotlib.pyplot as plt
import numpy as np
import sys
from argparse import ArgumentParser
import mir_eval
from sklearn.externals import joblib
from sklearn.mixture import GaussianMixture
from statistics import median
from onoffset_modules import *

#----------------------------
# Smoothing Process
#----------------------------
def Smooth_prediction(predict_notes, threshold):
    Filter = np.ndarray(shape=(5,), dtype=float, buffer=np.array([0.25, 0.5, 1.0, 0.5, 0.25]))
    onset_times = []
    prob_seq = []
    for num in range(predict_notes.shape[1]):
        if num > 1 and num < predict_notes.shape[1]-2:
            prob_seq.append(np.dot(predict_notes[0,num-2:num+3], Filter) / 2.5)
        else:
            prob_seq.append(predict_notes[0][num])

    # find local min, mark 
    if prob_seq[0] > prob_seq[1] and prob_seq[0] > prob_seq[2] and prob_seq[0] > threshold:
        onset_times.append(0.01)
    if prob_seq[1] > prob_seq[0] and prob_seq[1] > prob_seq[2] and prob_seq[1] > prob_seq[3] and prob_seq[1] > threshold:
        onset_times.append(0.03)
    for num in range(len(prob_seq)):
        if num > 1 and num < len(prob_seq)-2:
            if prob_seq[num] > prob_seq[num-1] and prob_seq[num] > prob_seq[num-2] and prob_seq[num] > prob_seq[num+1] and prob_seq[num] > prob_seq[num+2] and prob_seq[num] > threshold:
                onset_times.append(0.02*num+0.01)
    if prob_seq[len(prob_seq)-1] > prob_seq[len(prob_seq)-2] and prob_seq[len(prob_seq)-1] > prob_seq[len(prob_seq)-3] and prob_seq[len(prob_seq)-1] > threshold:
        onset_times.append(0.02*(len(prob_seq)-1)+0.01)
    if prob_seq[len(prob_seq)-2] > prob_seq[len(prob_seq)-1] and prob_seq[len(prob_seq)-2] > prob_seq[len(prob_seq)-3] and prob_seq[len(prob_seq)-2] > prob_seq[len(prob_seq)-4] and prob_seq[len(prob_seq)-2] > threshold:
        onset_times.append(0.02*(len(prob_seq)-2)+0.01)


    prob_seq_np = np.ndarray(shape=(len(prob_seq),), dtype=float, buffer=np.array(prob_seq))

    return np.ndarray(shape=(len(onset_times),), dtype=float, buffer=np.array(onset_times)), prob_seq_np

def find_first_bellow_thres(aSeq):
    activate = False
    first_bellow_frame = 0
    for i in range(len(aSeq)):
        if aSeq[i] > 0.5:
            activate = True
        if activate and aSeq[i] < 0.5:
            first_bellow_frame = i
            break
    return first_bellow_frame

def Smooth_sdt6(predict_sdt, threshold=0.5):
    # predict shape: (time step, 3)
    Filter = np.ndarray(shape=(5,), dtype=float, buffer=np.array([0.25, 0.5, 1.0, 0.5, 0.25]))
    #Filter = np.ndarray(shape=(5,), dtype=float, buffer=np.array([1.0, 1.0, 1.0, 1.0, 1.0]))
    sSeq = []
    dSeq = []
    onSeq = []
    offSeq = []
    
    for num in range(predict_sdt.shape[0]):
        if num > 1 and num < predict_sdt.shape[0]-2:
            sSeq.append(predict_sdt[num][0])
            dSeq.append(predict_sdt[num][1])
            onSeq.append(np.dot(predict_sdt[num-2:num+3, 3], Filter) / 2.5)
            offSeq.append(np.dot(predict_sdt[num-2:num+3, 5], Filter) / 2.5)

        else:
            sSeq.append(predict_sdt[num][0])
            dSeq.append(predict_sdt[num][1])
            onSeq.append(predict_sdt[num][3])
            offSeq.append(predict_sdt[num][5])
    
    ##############################
    # Peak strategy
    ##############################
    
    # find peak of transition
    # peak time = frame*0.02+0.01
    onpeaks = []
    if onSeq[0] > onSeq[1] and onSeq[0] > onSeq[2] and onSeq[0] > threshold:
        onpeaks.append(0)
    if onSeq[1] > onSeq[0] and onSeq[1] > onSeq[2] and onSeq[1] > onSeq[3] and onSeq[1] > threshold:
        onpeaks.append(1)
    for num in range(len(onSeq)):
        if num > 1 and num < len(onSeq)-2:
            #if onSeq[num] > 0.95:
            #    if onSeq[num-1] < 0.95 or onSeq[num+1] < 0.95:
            #        onpeaks.append(num)  
            #elif onSeq[num] > onSeq[num-1] and onSeq[num] > onSeq[num-2] and onSeq[num] > onSeq[num+1] and onSeq[num] > onSeq[num+2] and onSeq[num] > threshold:
            if onSeq[num] > onSeq[num-1] and onSeq[num] > onSeq[num-2] and onSeq[num] > onSeq[num+1] and onSeq[num] > onSeq[num+2] and onSeq[num] > threshold:
                onpeaks.append(num)

    if onSeq[-1] > onSeq[-2] and onSeq[-1] > onSeq[-3] and onSeq[-1] > threshold:
        onpeaks.append(len(onSeq)-1)
    if onSeq[-2] > onSeq[-1] and onSeq[-2] > onSeq[-3] and onSeq[-2] > onSeq[-4] and onSeq[-2] > threshold:
        onpeaks.append(len(onSeq)-2)


    offpeaks = []
    if offSeq[0] > offSeq[1] and offSeq[0] > offSeq[2] and offSeq[0] > threshold:
        offpeaks.append(0)
    if offSeq[1] > offSeq[0] and offSeq[1] > offSeq[2] and offSeq[1] > offSeq[3] and offSeq[1] > threshold:
        offpeaks.append(1)
    for num in range(len(offSeq)):
        if num > 1 and num < len(offSeq)-2:
            #if offSeq[num] > 0.95:
            #    if offSeq[num-1] < 0.95 or offSeq[num+1] < 0.95:
            #        offpeaks.append(num)  
            #elif offSeq[num] > offSeq[num-1] and offSeq[num] > offSeq[num-2] and offSeq[num] > offSeq[num+1] and offSeq[num] > offSeq[num+2] and offSeq[num] > threshold:
            if offSeq[num] > offSeq[num-1] and offSeq[num] > offSeq[num-2] and offSeq[num] > offSeq[num+1] and offSeq[num] > offSeq[num+2] and offSeq[num] > threshold:
                offpeaks.append(num)

    if offSeq[-1] > offSeq[-2] and offSeq[-1] > offSeq[-3] and offSeq[-1] > threshold:
        offpeaks.append(len(offSeq)-1)
    if offSeq[-2] > offSeq[-1] and offSeq[-2] > offSeq[-3] and offSeq[-2] > offSeq[-4] and offSeq[-2] > threshold:
        offpeaks.append(len(offSeq)-2)

    # determine onset/offset by silence, duration
    # intervalSD = [0,1,0,1,...], 0:silence, 1:duration
    if len(onpeaks) == 0 or len(offpeaks) == 0:
        return None

    
    Tpeaks = onpeaks + offpeaks
    Tpeaks.sort()

    intervalSD = [0]

    for i in range(len(Tpeaks)-1):
        current_sd = 0 if sum(sSeq[Tpeaks[i]:Tpeaks[i+1]]) > sum(dSeq[Tpeaks[i]:Tpeaks[i+1]]) else 1
        intervalSD.append(current_sd)
    intervalSD.append(0)


    MissingT= 0
    AddingT = 0
    est_intervals = []
    t_idx = 0
    while t_idx < len(Tpeaks):
        if t_idx == len(Tpeaks)-1:
            break
        if t_idx == 0 and Tpeaks[t_idx] not in onpeaks:
            if intervalSD[0] == 1 and intervalSD[1] == 0:
                onset_inserted = find_first_bellow_thres(sSeq[0:Tpeaks[0]])
                if onset_inserted != Tpeaks[0] and Tpeaks > onset_inserted + 1:
                    est_intervals.append([0.02*onset_inserted+0.01, 0.02*Tpeaks[0]+0.01])
                    #MissingT += 1
                    AddingT += 1
                else:
                    MissingT += 1
            t_idx += 1

        if Tpeaks[t_idx] in onpeaks and Tpeaks[t_idx+1] in offpeaks:
            if Tpeaks[t_idx] == Tpeaks[t_idx+1]:
                t_idx += 1
                continue
            if Tpeaks[t_idx+1] > Tpeaks[t_idx]+1: 
                est_intervals.append([0.02*Tpeaks[t_idx]+0.01, 0.02*Tpeaks[t_idx+1]+0.01])
            assert(Tpeaks[t_idx] < Tpeaks[t_idx+1])
            t_idx += 2
        elif Tpeaks[t_idx] in onpeaks and Tpeaks[t_idx+1] in onpeaks:
            offset_inserted = find_first_bellow_thres(dSeq[Tpeaks[t_idx]:Tpeaks[t_idx+1]]) + Tpeaks[t_idx]
            if offset_inserted != Tpeaks[t_idx] and offset_inserted > Tpeaks[t_idx]+1:
                est_intervals.append([0.02*Tpeaks[t_idx]+0.01, 0.02*offset_inserted+0.01])
                # MissingT += 1
                AddingT += 1
                assert(Tpeaks[t_idx] < offset_inserted)
            else:
                MissingT += 1
            #elif Tpeaks[t_idx+1] > Tpeaks[t_idx] + 1:
            #    MissingT += 1
            #elif Tpeaks[t_idx+1] > Tpeaks[t_idx]+1:
            #	est_intervals.append([0.02*Tpeaks[t_idx]+0.01, 0.02*Tpeaks[t_idx+1]+0.01])
            #	MissingT += 1
            t_idx += 1
        elif Tpeaks[t_idx] in offpeaks:
            if intervalSD[t_idx] == 1 and intervalSD[t_idx+1] == 0:
                onset_inserted = find_first_bellow_thres(sSeq[Tpeaks[t_idx-1]:Tpeaks[t_idx]]) + Tpeaks[t_idx-1]
                if onset_inserted != Tpeaks[t_idx-1] and Tpeaks[t_idx]>onset_inserted+1:
                    est_intervals.append([0.02*onset_inserted+0.01, 0.02*Tpeaks[t_idx]+0.01])
                    #MissingT += 1
                    AddingT += 1
                    assert(onset_inserted < Tpeaks[t_idx])
                else:
                	MissingT += 1
                #elif Tpeaks[t_idx] > Tpeaks[t_idx-1]+1:
                #    MissingT += 1
                #elif Tpeaks[t_idx] > Tpeaks[t_idx-1]+1:
                #    est_intervals.append([0.02*Tpeaks[t_idx-1]+0.01, 0.02*Tpeaks[t_idx]+0.01])
                #    MissingT += 1
            t_idx += 1
    
    #print("Missing ratio: ", MissingT/len(est_intervals))
    print("Note Segmentation: CFR = %.4f" %(MissingT/(len(Tpeaks)+AddingT)))

    #if len(onset_times) != len(offset_times):
    #    print("Onset times %d and offset times %d mismatch..." %(len(onset_times), len(offset_times)))
    
    #onpeaks = [x* 0.02 +0.01 for x in onpeaks]
    #offpeaks = [x* 0.02 +0.01 for x in offpeaks]
    #onpeaks_np = np.ndarray(shape=(len(onpeaks),1), dtype=float, buffer=np.array(onpeaks))
    #offpeaks_np = np.ndarray(shape=(len(offpeaks),1), dtype=float, buffer=np.array(offpeaks))
    #est_intervals = Naive_match(onpeaks_np, offpeaks_np)

    # Modify 1
    sSeq_np = np.ndarray(shape=(len(sSeq),), dtype=float, buffer=np.array(sSeq))
    dSeq_np = np.ndarray(shape=(len(dSeq),), dtype=float, buffer=np.array(dSeq))
    onSeq_np = np.ndarray(shape=(len(onSeq),), dtype=float, buffer=np.array(onSeq))
    offSeq_np = np.ndarray(shape=(len(offSeq),), dtype=float, buffer=np.array(offSeq))

    #return np.ndarray(shape=(len(onset_times),), dtype=float, buffer=np.array(onset_times)), np.ndarray(shape=(len(offset_times),), dtype=float, buffer=np.array(offset_times)), sSeq_np, dSeq_np, tSeq_np
    return np.ndarray(shape=(len(est_intervals),2), dtype=float, buffer=np.array(est_intervals)),  sSeq_np, dSeq_np, onSeq_np, offSeq_np, MissingT/(len(Tpeaks)+AddingT)
    #return est_intervals,  sSeq_np, dSeq_np, onSeq_np, offSeq_np


def Naive_match(onset_times ,offset_times):
    est_intervals = np.zeros((onset_times.shape[0], 2))
    fit_offset_idx = 0
    all_checked = False
    last_onset_matched = 0

    #print(onset_times)
    #print(offset_times)
    #input()

    double_onset_count = 0
    onset_offset_pair_count = 0

    onset_times_sort = np.sort(onset_times, axis=0)
    offset_times_sort = np.sort(offset_times, axis=0)

    for i in range(onset_times.shape[0]):
        # Match for i-th onset time
        if i == onset_times.shape[0] - 1:
            while onset_times_sort[i] >= offset_times_sort[fit_offset_idx]:
                if fit_offset_idx == offset_times.shape[0]-1:
                    all_checked = True
                    break
                else:
                    fit_offset_idx += 1
            if all_checked:
                last_onset_matched = i
                break
            else:
                est_intervals[i][0] = onset_times_sort[i]
                est_intervals[i][1] = offset_times_sort[fit_offset_idx]
        
        else:
            est_intervals[i][0] = onset_times_sort[i]
            if onset_times_sort[i+1] < offset_times_sort[fit_offset_idx]:
                est_intervals[i][1] = onset_times_sort[i+1]
                double_onset_count += 1
            else:
                while onset_times_sort[i] >= offset_times_sort[fit_offset_idx]:
                    if fit_offset_idx == offset_times.shape[0]-1:
                        all_checked = True
                        break
                    else:
                        fit_offset_idx += 1

                if all_checked:
                    last_onset_matched = i
                    break
                else:
                    est_intervals[i][1] = offset_times_sort[fit_offset_idx]
                    fit_offset_idx = fit_offset_idx + 1 if fit_offset_idx < offset_times.shape[0]-1 else offset_times.shape[0]-1
                    onset_offset_pair_count += 1

    if all_checked:
        est_intervals = np.delete(est_intervals, np.s_[last_onset_matched:onset_times.shape[0]], axis=0)

    print('Pair ratio = %.4f' %(onset_offset_pair_count/(onset_offset_pair_count+double_onset_count)))
    #print(est_intervals)

    return est_intervals

def Naive_pitch(pitch_step, pitch_intervals):
    interval_idx = 0
    pitch_buf = []
    pitch_est = np.zeros((pitch_intervals.shape[0],))
    onset_flag = False

    for i in range(pitch_intervals.shape[0]):
        start_frame = int((pitch_intervals[i][0]-0.01) / 0.02)
        end_frame = int((pitch_intervals[i][1]-0.01) / 0.02)
        if end_frame == start_frame + 1 or end_frame == start_frame:
            pitch_est[i] = pitch_step[start_frame] if pitch_step[start_frame] != 0 else 1.0
        else:
            pitch_est[i] = np.median(pitch_step[start_frame:end_frame]) if np.median(pitch_step[start_frame:end_frame]) != 0 else 1.0

    return pitch_est

def nearest_match(on, off):
    buf_on = []
    buf_off = []
    buf_on_wait = 0
    on_in = False
    on_idx = 0
    off_idx = 0

    while on_idx < len(on) and off_idx < len(off):
        if on[on_idx] < off[off_idx]:
            on_in = True
            buf_on_wait = on[on_idx]
            on_idx += 1
            continue
        elif off[off_idx] < on[on_idx]:
            if on_in:
                on_in = False
                if buf_on_wait >= off[off_idx]:
                    print("Error occurs!!")
                    input()
                buf_on.append(buf_on_wait)
                buf_off.append(off[off_idx])
            
            off_idx += 1
            continue
        else:
            print(on[on_idx])
            print(off[off_idx])
            print("Infinite Loop!!")
            input()
    onset_np = np.ndarray(shape=(len(buf_on),1), dtype=float, buffer=np.array(buf_on))
    offset_np = np.ndarray(shape=(len(buf_off),1), dtype=float, buffer=np.array(buf_off))
    return np.hstack((onset_np, offset_np))


def pitch2freq(pitch_np):
    freq_l = [ (2**((pitch_np[i]-69)/12))*440 for i in range(pitch_np.shape[0]) ]
    return np.ndarray(shape=(len(freq_l),), dtype=float, buffer=np.array(freq_l))

def freq2pitch(freq_np):
    pitch_np = 69+12*np.log2(freq_np/440)
    return pitch_np

#----------------------------
# Parser
#----------------------------
parser = ArgumentParser()
parser.add_argument("-d", help="data file position", dest="dfile", default="data.npy", type=str)
parser.add_argument("-p", help="pitch file position", dest="pfile", default="p.npy", type=str)
parser.add_argument("-of", help="output est file position", dest="offile", default="o.npy", type=str)
parser.add_argument("-sm", help="output smooth file position", dest="smfile", default="o.npy", type=str)
parser.add_argument("-m", help="decoder model file position", dest="dmfile1", default="model/onset_v3_model", type=str)
parser.add_argument("--feat", help="feature cascaded", dest="feat_num", default=1, type=int)
parser.add_argument("--threshold", help="post-processing threshold", dest="threshold", default=0.5, type=float)

args = parser.parse_args()

#----------------------------
# Parameters
#----------------------------
#START_LIST = [1, 3, 5, 7, 9]
START_LIST = [1]
data_file = args.dfile # Z file
p_file = args.pfile  # marked onset/offset/pitch matrix file
of_file = args.offile  # marked onset/offset/pitch matrix file
sm_file = args.smfile  # marked onset/offset/pitch matrix file
on_dec_model_file = args.dmfile1
INPUT_SIZE = 174*args.feat_num
OUTPUT_SIZE = 6
THRESHOlD = args.threshold
PATIENCE = 700
BATCH_SIZE = 10

#----------------------------
# Data Collection
#----------------------------
print("Note Segmentation: Loading data ...")
with open(data_file, 'r') as fd:
    data_np = np.loadtxt(fd)
    data_np = np.transpose(data_np)
    #data_np = data_np.reshape((1,-1))
    data_np = data_np.reshape((1, -1, int(args.feat_num//3), 174*3)).transpose((0,2,3,1))
    data = torch.from_numpy(data_np).type(torch.FloatTensor).cuda()

with open(p_file, 'r') as fp:
    p_np = np.loadtxt(fp)
    p_np = np.delete(p_np, 0, axis=1)
    p_np = p_np.reshape((p_np.shape[0],))

#input & target data
input_loader = data_utils.DataLoader(
    ConcatDataset(data), 
    batch_size=BATCH_SIZE,
    shuffle=False)

#----------------------------
# Model Initialize
#----------------------------
#####################################
# ATTENTION MODEL
#####################################
# load resnet50
print("Note Segmentation: Loading model ...")
resnet18 = models.resnet18(pretrained=False)
num_ftrs = resnet18.fc.in_features
resnet18.fc = nn.Linear(num_ftrs, OUTPUT_SIZE)
num_fout = resnet18.conv1.out_channels
resnet18.conv1 = nn.Conv2d(int(args.feat_num//3), num_fout, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
resnet18.avgpool = nn.AvgPool2d(kernel_size=(17,1), stride=1, padding=0)

onDec = resnet18

onDec.load_state_dict(torch.load(on_dec_model_file))

onDec.cuda()

#----------------------------
# Evaluation
#----------------------------
print("Note Segmentation: Estimating Note Interval ...")
for step, xys in enumerate(input_loader):                 # gives batch data
    b_x = xys[0].contiguous() # reshape x to (batch, time_step, input_size)

    predict_on_notes = []

    input_time_step = b_x.shape[3]
    k = 9
    window_size = 2*k+1
    
    nn_softmax = nn.Softmax(dim=1)
    
    for step in range((input_time_step//BATCH_SIZE)+1):
        if BATCH_SIZE*step > k and BATCH_SIZE*step < input_time_step - (k+1) - BATCH_SIZE:

            input_Var = Variable(torch.stack([ b_x[0, :, :, BATCH_SIZE*step+i-k:BATCH_SIZE*step+i-k+window_size]\
                           for i in range(BATCH_SIZE)], dim=0))

            onDecOut6 = onDec(input_Var)
            onDecOut1 = nn_softmax(onDecOut6[:, :2])
            onDecOut2 = nn_softmax(onDecOut6[:, 2:4])
            onDecOut3 = nn_softmax(onDecOut6[:, 4:])
            
            for i in range(BATCH_SIZE):
                predict_on_note = [ onDecOut1.view(BATCH_SIZE, 1, 2).data[i][0][j] for j in range(2) ]
                predict_on_note += [ onDecOut2.view(BATCH_SIZE, 1, 2).data[i][0][j] for j in range(2) ]
                predict_on_note += [ onDecOut3.view(BATCH_SIZE, 1, 2).data[i][0][j] for j in range(2) ]
                predict_on_notes.append(predict_on_note)

            #####################################

        elif BATCH_SIZE*step <= k:
            for i in range(BATCH_SIZE):
                predict_on_notes.append([0.0 for j in range(OUTPUT_SIZE)])
        elif BATCH_SIZE*step >= input_time_step - (k+1) - BATCH_SIZE:
            for i in range(BATCH_SIZE):
                predict_on_notes.append([0.0 for j in range(OUTPUT_SIZE)])
        else:
            predict_on_notes.append([0.0 for j in range(OUTPUT_SIZE)])

    predict_on_notes_np = np.ndarray(shape=(len(predict_on_notes), OUTPUT_SIZE), dtype=float, buffer=np.array(predict_on_notes))

    pitch_intervals, sSeq_np, dSeq_np, onSeq_np, offSeq_np, conflict_ratio = Smooth_sdt6(predict_on_notes_np, THRESHOlD) # list of onset secs, ndarray
    freq_est = Naive_pitch(p_np, pitch_intervals)

    out_est = np.hstack((pitch_intervals, freq_est.reshape((-1,1))))
    out_smooth = np.hstack((sSeq_np.reshape((-1,1)), dSeq_np.reshape((-1,1)), onSeq_np.reshape((-1,1)), offSeq_np.reshape((-1,1))))
    np.savetxt(of_file, out_est)
    np.savetxt(sm_file, out_smooth)
