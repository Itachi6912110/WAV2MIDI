# -*- coding: utf-8 -*-
"""
Created on Fri Aug 25 21:54:18 2017

@author: lisu
"""
import soundfile as sf
import numpy as np
import scipy
import scipy.signal
import argparse
from keras.models import load_model

def STFT(x, fr, fs, Hop, h):        
    t = np.arange(Hop, np.ceil(len(x)/float(Hop))*Hop, Hop)
    N = int(fs/float(fr))
    window_size = len(h)
    f = fs*np.linspace(0, 0.5, np.round(N/2), endpoint=True)
    Lh = int(np.floor(float(window_size-1) / 2))
    tfr = np.zeros((int(N), len(t)), dtype=np.float)     
        
    for icol in range(0, len(t)):
        ti = int(t[icol])           
        tau = np.arange(int(-min([round(N/2.0)-1, Lh, ti-1])), \
                        int(min([round(N/2.0)-1, Lh, len(x)-ti])))
        indices = np.mod(N + tau, N) + 1                                             
        tfr[indices-1, icol] = x[ti+tau-1] * h[Lh+tau-1] \
                                /np.linalg.norm(h[Lh+tau-1])           
                            
    tfr = abs(scipy.fftpack.fft(tfr, n=N, axis=0))  
    return tfr, f, t, N

def nonlinear_func(X, g, cutoff):
    cutoff = int(cutoff)
    if g!=0:
        X[X<0] = 0
        X[:cutoff, :] = 0
        X[-cutoff:, :] = 0
        X = np.power(X, g)
    else:
        X = np.log(X)
        X[:cutoff, :] = 0
        X[-cutoff:, :] = 0
    return X

def Freq2LogFreqMapping(tfr, f, fr, fc, tc, NumPerOct):
    StartFreq = fc
    StopFreq = 1/tc
    Nest = int(np.ceil(np.log2(StopFreq/StartFreq))*NumPerOct)
    central_freq = []

    for i in range(0, Nest):
        CenFreq = StartFreq*pow(2, float(i)/NumPerOct)
        if CenFreq < StopFreq:
            central_freq.append(CenFreq)
        else:
            break

    Nest = len(central_freq)
    freq_band_transformation = np.zeros((Nest-1, len(f)), dtype=np.float)
    for i in range(1, Nest-1):
        l = int(round(central_freq[i-1]/fr))
        r = int(round(central_freq[i+1]/fr)+1)
        #rounding1
        if l >= r-1:
            freq_band_transformation[i, l] = 1
        else:
            for j in range(l, r):
                if f[j] > central_freq[i-1] and f[j] < central_freq[i]:
                    freq_band_transformation[i, j] = (f[j] - central_freq[i-1]) / (central_freq[i] - central_freq[i-1])
                elif f[j] > central_freq[i] and f[j] < central_freq[i+1]:
                    freq_band_transformation[i, j] = (central_freq[i + 1] - f[j]) / (central_freq[i + 1] - central_freq[i])
    tfrL = np.dot(freq_band_transformation, tfr)
    return tfrL, central_freq

def Quef2LogFreqMapping(ceps, q, fs, fc, tc, NumPerOct):
    StartFreq = fc
    StopFreq = 1/tc
    Nest = int(np.ceil(np.log2(StopFreq/StartFreq))*NumPerOct)
    central_freq = []

    for i in range(0, Nest):
        CenFreq = StartFreq*pow(2, float(i)/NumPerOct)
        if CenFreq < StopFreq:
            central_freq.append(CenFreq)
        else:
            break
    f = 1/q
    Nest = len(central_freq)
    freq_band_transformation = np.zeros((Nest-1, len(f)), dtype=np.float)
    for i in range(1, Nest-1):
        for j in range(int(round(fs/central_freq[i+1])), int(round(fs/central_freq[i-1])+1)):
            if f[j] > central_freq[i-1] and f[j] < central_freq[i]:
                freq_band_transformation[i, j] = (f[j] - central_freq[i-1])/(central_freq[i] - central_freq[i-1])
            elif f[j] > central_freq[i] and f[j] < central_freq[i+1]:
                freq_band_transformation[i, j] = (central_freq[i + 1] - f[j]) / (central_freq[i + 1] - central_freq[i])

    tfrL = np.dot(freq_band_transformation, ceps)
    return tfrL, central_freq

def CFP_filterbank(x, fr, fs, Hop, h, fc, tc, g, NumPerOctave):
    NumofLayer = np.size(g)

    [tfr, f, t, N] = STFT(x, fr, fs, Hop, h)
    tfr = np.power(abs(tfr), g[0])
    tfr0 = tfr # original STFT
    ceps = np.zeros(tfr.shape)


    if NumofLayer >= 2:
        for gc in range(1, NumofLayer):
            if np.remainder(gc, 2) == 1:
                tc_idx = round(fs*tc)
                ceps = np.real(np.fft.fft(tfr, axis=0))/np.sqrt(N)
                ceps = nonlinear_func(ceps, g[gc], tc_idx)
            else:
                fc_idx = round(fc/fr)
                tfr = np.real(np.fft.fft(ceps, axis=0))/np.sqrt(N)
                tfr = nonlinear_func(tfr, g[gc], fc_idx)

    tfr0 = tfr0[:int(round(N/2)),:]
    tfr = tfr[:int(round(N/2)),:]
    ceps = ceps[:int(round(N/2)),:]

    HighFreqIdx = int(round((1/tc)/fr)+1)
    f = f[:HighFreqIdx]
    tfr0 = tfr0[:HighFreqIdx,:]
    tfr = tfr[:HighFreqIdx,:]
    HighQuefIdx = int(round(fs/fc)+1)
    q = np.arange(HighQuefIdx)/float(fs)
    ceps = ceps[:HighQuefIdx,:]
    
    tfrL0, central_frequencies = Freq2LogFreqMapping(tfr0, f, fr, fc, tc, NumPerOctave)
    tfrLF, central_frequencies = Freq2LogFreqMapping(tfr, f, fr, fc, tc, NumPerOctave)
    tfrLQ, central_frequencies = Quef2LogFreqMapping(ceps, q, fs, fc, tc, NumPerOctave)

    return tfrL0, tfrLF, tfrLQ, f, q, t, central_frequencies 

def full_feature_extraction(filename):
    x, fs = sf.read(filename)
#    if x.shape[1]>1:
#        x = np.mean(x, axis = 1)
    x = scipy.signal.resample_poly(x, 16000, fs)
    fs = 16000.0 # sampling frequency
    x = x.astype('float32')
    Hop = 320 # hop size (in sample)
    #Hop = 160 # hop size (in sample)
    h3 = scipy.signal.blackmanharris(743) # window size - 2048
    h2 = scipy.signal.blackmanharris(372) # window size - 1024
    h1 = scipy.signal.blackmanharris(186) # window size - 512
    fr = 2.0 # frequency resolution
    fc = 80.0 # the frequency of the lowest pitch
    tc = 1/1000.0 # the period of the highest pitch
    g = np.array([0.24, 0.6, 1])
    NumPerOctave = 48 # Number of bins per octave
    
    tfrL01, tfrLF1, tfrLQ1, f1, q1, t1, CenFreq1 = CFP_filterbank(x, fr, fs, Hop, h1, fc, tc, g, NumPerOctave)
    tfrL02, tfrLF2, tfrLQ2, f2, q2, t2, CenFreq2 = CFP_filterbank(x, fr, fs, Hop, h2, fc, tc, g, NumPerOctave)
    tfrL03, tfrLF3, tfrLQ3, f3, q3, t3, CenFreq3 = CFP_filterbank(x, fr, fs, Hop, h3, fc, tc, g, NumPerOctave)
    Z1 = tfrLF1 * tfrLQ1
    ZN1 = (Z1 - np.mean(Z1)) / np.std(Z1)
    Z2 = tfrLF2 * tfrLQ2
    ZN2 = (Z2 - np.mean(Z2)) / np.std(Z2)
    Z3 = tfrLF3 * tfrLQ3
    ZN3 = (Z3 - np.mean(Z3)) / np.std(Z3)
    SN1 = gen_spectral_flux(tfrL01, invert=False, norm=True)
    SN2 = gen_spectral_flux(tfrL02, invert=False, norm=True)
    SN3 = gen_spectral_flux(tfrL03, invert=False, norm=True)
    SIN1 = gen_spectral_flux(tfrL01, invert=True, norm=True)
    SIN2 = gen_spectral_flux(tfrL02, invert=True, norm=True)
    SIN3 = gen_spectral_flux(tfrL03, invert=True, norm=True)
    #print(Z1.shape)
    #print(SN1.shape)
    #print(SIN1.shape)
    SN = np.concatenate((SN1, SN2, SN3), axis=0)
    SIN = np.concatenate((SIN1, SIN2, SIN3), axis=0)
    ZN = np.concatenate((ZN1, ZN2, ZN3), axis=0)
    SN_SIN_ZN = np.concatenate((SN, SIN, ZN), axis=0)
    #print(SN_SIN_ZN.shape)
    #input("check ...")

    #return Z, CenFreq, tfrL0, tfrLF, tfrLQ
    return SN_SIN_ZN, Z1, CenFreq1

def gen_spectral_flux(S, invert=False, norm=True):
    flux = np.diff(S)
    first_col = np.zeros((S.shape[0],1))
    flux = np.hstack((first_col, flux))
    
    if invert:
        flux = flux * (-1.0)

    flux = np.where(flux < 0, 0.0, flux)

    if norm:
        flux = (flux - np.mean(flux)) / np.std(flux)

    return flux

def feature_extraction(filename):
    x, fs = sf.read(filename)
    #    if x.shape[1]>1:
    #        x = np.mean(x, axis = 1)
    x = scipy.signal.resample_poly(x, 16000, fs)
    fs = 16000.0 # sampling frequency
    x = x.astype('float32')
    Hop = 320 # hop size (in sample)
    h = scipy.signal.blackmanharris(2049) # window size
    fr = 2.0 # frequency resolution
    fc = 80.0 # the frequency of the lowest pitch
    tc = 1/1000.0 # the period of the highest pitch
    g = np.array([0.24, 0.6, 1])
    NumPerOctave = 48 # Number of bins per octave
    
    tfrL0, tfrLF, tfrLQ, f, q, t, CenFreq = CFP_filterbank(x, fr, fs, Hop, h, fc, tc, g, NumPerOctave)
    Z = tfrLF * tfrLQ
    return Z, t, CenFreq, tfrL0, tfrLF, tfrLQ


def patch_extraction(Z, patch_size, th):
    # Z is the input spectrogram or any kind of time-frequency representation
    M, N = np.shape(Z)    
    half_ps = int(np.floor(float(patch_size)/2))

    Z = np.append(np.zeros([M, half_ps]), Z, axis = 1)
    Z = np.append(Z, np.zeros([M, half_ps]), axis = 1)
    Z = np.append(Z, np.zeros([half_ps, N+2*half_ps]), axis = 0)

    M, N = np.shape(Z)
    
#    data = np.zeros([1, patch_size, patch_size])
#    mapping = np.zeros([1, 2])
    data = np.zeros([300000, patch_size, patch_size])
    mapping = np.zeros([300000, 2])
    counter = 0
    for t_idx in range(half_ps, N-half_ps):
        PKS, LOCS = findpeaks(Z[:,t_idx], th)
#        print('time at: ', t_idx)
        for mm in range(0, len(LOCS)):
            if LOCS[mm] >= half_ps and LOCS[mm] < M - half_ps and counter<300000:# and PKS[mm]> 0.5*max(Z[:,t_idx]):
                patch = Z[np.ix_(range(LOCS[mm]-half_ps, LOCS[mm]+half_ps+1), range(t_idx-half_ps, t_idx+half_ps+1))]
                patch = patch.reshape(1, patch_size, patch_size)
#                data = np.append(data, patch, axis=0)
#                mapping = np.append(mapping, np.array([[LOCS[mm], t_idx]]), axis=0)
                data[counter,:,:] = patch
                mapping[counter,:] = np.array([[LOCS[mm], t_idx]])
                counter = counter + 1
            elif LOCS[mm] >= half_ps and LOCS[mm] < M - half_ps and counter>=300000:
                print('Out of the biggest size. Please shorten the input audio.')
                
    data = data[:counter-1,:,:]
    mapping = mapping[:counter-1,:]
    Z = Z[:M-half_ps,:]
#    print(data.shape)
#    print(mapping.shape)
    return data, mapping, half_ps, N, Z

def patch_prediction(modelname, data, patch_size):
    data = data.reshape(data.shape[0], patch_size, patch_size, 1)
    model = load_model(modelname)
    pred  = model.predict(data)
    return pred

def contour_prediction(mapping, pred, N, half_ps, Z, t, CenFreq, max_method):
    PredContour = np.zeros(N)

    pred = pred[:,1]
    pred_idx = np.where(pred>0.5)
    MM = mapping[pred_idx[0],:]
#    print(MM.shape)
    pred_prob = pred[pred_idx[0]]
#    print(pred_prob.shape)
    MM = np.append(MM, np.reshape(pred_prob, [len(pred_prob),1]), axis=1)
    MM = MM[MM[:,1].argsort()]    
    
    for t_idx in range(half_ps, N-half_ps):
        Candidate = MM[np.where(MM[:,1]==t_idx)[0],:]
#        print(Candidate[:,2])
        if Candidate.shape[0] >= 2:
            if max_method == 'posterior':
                fi = np.where(Candidate[:,2]==np.max(Candidate[:,2]))
                fi = fi[0]
            elif max_method == 'prior':
                fi = Z[Candidate[:,0].astype('int'),t_idx].argmax(axis=0)
            fi = fi.astype('int')
#            print(fi)
            PredContour[Candidate[fi,1].astype('int')] = Candidate[fi,0] 
        elif Candidate.shape[0] == 1:
            PredContour[Candidate[0,1].astype('int')] = Candidate[0,0] 
    
    # clip the padding of time
    PredContour = PredContour[range(half_ps, N-half_ps)]
    
    for k in range(len(PredContour)):
        if PredContour[k]>1:
            PredContour[k] = CenFreq[PredContour[k].astype('int')]
    
    Z = Z[:, range(half_ps, N-half_ps)]
#    print(t.shape)
#    print(PredContour.shape)    
    result = np.zeros([t.shape[0],2])
    result[:,0] = t/16000.0
    result[:,1] = PredContour
    return result

def contour_pred_from_raw(Z, t, CenFreq):
    PredContour = Z.argmax(axis=0)
    for k in range(len(PredContour)):
        if PredContour[k]>1:
            PredContour[k] = CenFreq[PredContour[k].astype('int')]
    result = np.zeros([t.shape[0],2])
    result[:,0] = t/16000.0
    result[:,1] = PredContour
    return result

def show_prediction(mapping, pred, N, half_ps, Z, t):
    postgram = np.zeros(Z.shape)
    pred = pred[:,1]
    for i in range(pred.shape[0]):
        postgram[mapping[i,0].astype('int'), mapping[i,1].astype('int')] = pred[i]
    return postgram

def findpeaks(x, th):
    # x is an input column vector
    M = x.shape[0]
    pre = x[1:M - 1] - x[0:M - 2]
    pre[pre < 0] = 0
    pre[pre > 0] = 1

    post = x[1:M - 1] - x[2:]
    post[post < 0] = 0
    post[post > 0] = 1

    mask = pre * post
    ext_mask = np.append([0], mask, axis=0)
    ext_mask = np.append(ext_mask, [0], axis=0)
    
    pdata = x * ext_mask
    pdata = pdata-np.tile(th*np.amax(pdata, axis=0),(M,1))
    pks = np.where(pdata>0)
    pks = pks[0]
    
    locs = np.where(ext_mask==1)
    locs = locs[0]
    return pks, locs

def melody_extraction(infile, outfile):
    # melody_extraction(‘path1/input.wav’, ‘path2/output.txt’)
    patch_size = 25
    th = 0.5
    modelname = 'model/model3_patch25'
    max_method = 'posterior'
    #print('Feature extraction of ' + infile)
    print('Feature Extraction: Extracting Pitch Contour ...')
    Z, t, CenFreq, tfrL0, tfrLF, tfrLQ = feature_extraction(infile)
    if max_method == 'raw':
        result = contour_pred_from_raw(Z, t, CenFreq)
        postgram = Z
    else:
        print('Patch extraction from %d frames' % (Z.shape[1]))
        data, mapping, half_ps, N, Z = patch_extraction(Z, patch_size, th)
        print('Predictions from %d patches' % (data.shape[0]))
        pred = patch_prediction(modelname, data, patch_size)
        result = contour_prediction(mapping, pred, N, half_ps, Z, t,\
                                    CenFreq, max_method)
        postgram = show_prediction(mapping, pred, N, half_ps, Z, t)
    #print(result.shape)
    np.savetxt(outfile, result)
    return result, postgram, Z, tfrL0, tfrLF, tfrLQ, t, CenFreq

#def output_feature_extraction(infile, outfile_z, outfile_t, outfile_f, outfile_s):
def output_feature_extraction(infile, outfile_feat, outfile_z, outfile_cf):
    print('Feature Extraction: Extracting Spectral Difference and CFP ...')
    #Z, t, f, CenFreq, tfrL0, tfrLF, tfrLQ = full_feature_extraction(infile)
    SN_SIN_ZN, Z1, CenFreq1 = full_feature_extraction(infile)
    #print(CenFreq.shape)
    #print(CenFreq)
    #input()
    #print(Z.shape)
    np.savetxt(outfile_feat, SN_SIN_ZN)
    np.savetxt(outfile_z, Z1)
    np.savetxt(outfile_cf, CenFreq1)
    #np.savetxt(outfile_t, t)
    #np.savetxt(outfile_f, f)
    #np.savetxt(outfile_s, tfrL0)
    return SN_SIN_ZN, Z1, CenFreq1

    
if __name__== "__main__":
#    melody_extraction(infile, outfile)

#    infile = 'opera_fem4.wav'
#    outfile = 'opera_fem4'
#    result = melody_extraction(infile, outfile)
#    plt.figure(1)
#    plt.plot(result[:,0],result[:,1])
    
    parser = argparse.ArgumentParser(
        description="Melody extraction")
    parser.add_argument("InFile", type=str)
    #parser.add_argument("OutFile", type=str)
    parser.add_argument("OutFile_FEAT", type=str)
    parser.add_argument("OutFile_Z", type=str)
    parser.add_argument("OutFile_CF", type=str)
    parser.add_argument("OutFile_P", type=str)
    #parser.add_argument("OutFile_T", type=str)
    #parser.add_argument("OutFile_F", type=str)
    #parser.add_argument("OutFile_S", type=str)
    
    args = parser.parse_args()

    melody_extraction(args.InFile, args.OutFile_P)
    output_feature_extraction(args.InFile, args.OutFile_FEAT, args.OutFile_Z, args.OutFile_CF)
