import torch
from torch import nn
from torch.autograd import Variable
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import torch.utils.data as data_utils
import numpy as np
import random

use_cuda = torch.cuda.is_available()

class ConcatDataset(torch.utils.data.Dataset):
    def __init__(self, *datasets):
        self.datasets = datasets

    def __getitem__(self, i):
        return tuple(d[i] for d in self.datasets)

    def __len__(self):
        return min(len(d) for d in self.datasets)

class Encoder(nn.Module):
    def __init__(self, INPUT_SIZE, HIDDEN_SIZE, HIDDEN_LAYER, MINI_BATCH, WINDOW_SIZE, BIDIR):
        super(Encoder, self).__init__()
        self.input_size = INPUT_SIZE
        self.hidden_size = HIDDEN_SIZE
        self.hidden_layer = HIDDEN_LAYER
        self.mini_batch  = MINI_BATCH
        self.window_size = WINDOW_SIZE
        self.bidir = BIDIR

        self.enc_lstm = nn.LSTM(
            input_size=INPUT_SIZE,
            hidden_size=HIDDEN_SIZE,         # rnn hidden unit
            num_layers=HIDDEN_LAYER,            # number of LSTM layer
            batch_first=True,        # (batch, time_step, input_size)
            bidirectional=BIDIR
        )

    def forward(self, input, hidden):
        output, hidden = self.enc_lstm(input, hidden)
        return output, hidden

    def initHidden(self, MINI_BATCH):
        result1 = Variable(torch.zeros(self.hidden_layer*2, MINI_BATCH, self.hidden_size)) if self.bidir else Variable(torch.zeros(self.hidden_layer, MINI_BATCH, self.hidden_size))
        result2 = Variable(torch.zeros(self.hidden_layer*2, MINI_BATCH, self.hidden_size)) if self.bidir else Variable(torch.zeros(self.hidden_layer, MINI_BATCH, self.hidden_size))
        if use_cuda:
            return (result1.cuda(), result2.cuda())
        else:
            return (result1, result2)

class LNEncoder(nn.Module):
    def __init__(self, INPUT_SIZE, HIDDEN_SIZE, HIDDEN_LAYER, MINI_BATCH, WINDOW_SIZE, BIDIR):
        super(LNEncoder, self).__init__()
        self.input_size = INPUT_SIZE
        self.hidden_size = HIDDEN_SIZE
        self.hidden_layer = HIDDEN_LAYER
        self.mini_batch  = MINI_BATCH
        self.window_size = WINDOW_SIZE
        self.bidir = BIDIR

        self.enc_lstm = nn.LSTM(
            input_size=INPUT_SIZE,
            hidden_size=HIDDEN_SIZE,         # rnn hidden unit
            num_layers=HIDDEN_LAYER,            # number of LSTM layer
            batch_first=True,        # (batch, time_step, input_size)
            bidirectional=BIDIR
        )
        self.enc_relu = nn.ReLU()
        self.enc_ln = nn.LayerNorm(2*HIDDEN_SIZE) if self.bidir else nn.LayerNorm(HIDDEN_SIZE)

    def forward(self, input, hidden):
        output, hidden = self.enc_lstm(input, hidden)
        output = self.enc_relu(output)
        output = self.enc_ln(output)
        return output, hidden

    def initHidden(self, MINI_BATCH):
        result1 = Variable(torch.zeros(self.hidden_layer*2, MINI_BATCH, self.hidden_size)) if self.bidir else Variable(torch.zeros(self.hidden_layer, MINI_BATCH, self.hidden_size))
        result2 = Variable(torch.zeros(self.hidden_layer*2, MINI_BATCH, self.hidden_size)) if self.bidir else Variable(torch.zeros(self.hidden_layer, MINI_BATCH, self.hidden_size))
        if use_cuda:
            return (result1.cuda(), result2.cuda())
        else:
            return (result1, result2)

class BNEncoder(nn.Module):
    def __init__(self, INPUT_SIZE, HIDDEN_SIZE, HIDDEN_LAYER, MINI_BATCH, WINDOW_SIZE, BIDIR):
        super(BNEncoder, self).__init__()
        self.input_size = INPUT_SIZE
        self.hidden_size = HIDDEN_SIZE
        self.hidden_layer = HIDDEN_LAYER
        self.mini_batch  = MINI_BATCH
        self.window_size = WINDOW_SIZE
        self.bidir = BIDIR

        self.enc_lstm = nn.LSTM(
            input_size=INPUT_SIZE,
            hidden_size=HIDDEN_SIZE,         # rnn hidden unit
            num_layers=HIDDEN_LAYER,            # number of LSTM layer
            batch_first=True,        # (batch, time_step, input_size)
            bidirectional=BIDIR
        )
        self.enc_relu = nn.ReLU()
        self.enc_bn = nn.BatchNorm1d(2*HIDDEN_SIZE) if self.bidir else nn.BatchNorm1d(HIDDEN_SIZE)

    def forward(self, input, hidden):
        output, hidden = self.enc_lstm(input, hidden)
        output = self.enc_relu(output)
        output = self.enc_bn(output.squeeze(1))
        return output.unsqueeze(1), hidden

    def initHidden(self, MINI_BATCH):
        result1 = Variable(torch.zeros(self.hidden_layer*2, MINI_BATCH, self.hidden_size)) if self.bidir else Variable(torch.zeros(self.hidden_layer, MINI_BATCH, self.hidden_size))
        result2 = Variable(torch.zeros(self.hidden_layer*2, MINI_BATCH, self.hidden_size)) if self.bidir else Variable(torch.zeros(self.hidden_layer, MINI_BATCH, self.hidden_size))
        if use_cuda:
            return (result1.cuda(), result2.cuda())
        else:
            return (result1, result2)

class AttentionDecoder(nn.Module):
    def __init__(self, INPUT_SIZE, OUTPUT_SIZE, HIDDEN_SIZE, HIDDEN_LAYER, MINI_BATCH, WINDOW_SIZE, BIDIR):
        super(AttentionDecoder, self).__init__()
        self.input_size = INPUT_SIZE
        self.output_size = OUTPUT_SIZE
        self.hidden_size = HIDDEN_SIZE
        self.hidden_layer = HIDDEN_LAYER
        self.mini_batch  = MINI_BATCH
        self.window_size = WINDOW_SIZE
        self.bidir = BIDIR

        self.in2hid = nn.Linear(INPUT_SIZE, HIDDEN_SIZE)
        self.attn = nn.Linear(3*HIDDEN_SIZE, WINDOW_SIZE) if BIDIR else nn.Linear(2*HIDDEN_SIZE, WINDOW_SIZE)
        self.attn_softmax = nn.Softmax(dim=1)
        self.attn_combine = nn.Linear(3*HIDDEN_SIZE, HIDDEN_SIZE) if BIDIR else nn.Linear(2*HIDDEN_SIZE, HIDDEN_SIZE)
        self.attn_relu = nn.ReLU()
        self.dec_lstm = nn.LSTM(
            input_size=HIDDEN_SIZE,
            hidden_size=HIDDEN_SIZE,         # rnn hidden unit
            num_layers=HIDDEN_LAYER,            # number of LSTM layer
            batch_first=True,        # (batch, time_step, input_size)
            bidirectional=BIDIR
        )
        self.dec_linear = nn.Linear(2*HIDDEN_SIZE, OUTPUT_SIZE) if BIDIR else nn.Linear(HIDDEN_SIZE, OUTPUT_SIZE)
        self.dec_softmax = nn.Softmax(dim=2)

    def forward(self, input, attn_hidden, hidden, enc_outs):
        embed = self.in2hid(input)
        attn_weights = self.attn(torch.cat((embed[0], attn_hidden), 1))
        attn_weights = self.attn_softmax(attn_weights)
        attn_applied = torch.bmm(attn_weights.unsqueeze(0), enc_outs.unsqueeze(0))

        output = torch.cat((embed[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)
        output = self.attn_relu(output)
        output, hidden = self.dec_lstm(output, hidden)
        output = self.dec_linear(output)
        output = self.dec_softmax(output)
        return output, hidden, attn_weights

    def initHidden(self, MINI_BATCH):
        result1 = Variable(torch.zeros(self.hidden_layer*2, MINI_BATCH, self.hidden_size)) if self.bidir else Variable(torch.zeros(self.hidden_layer, MINI_BATCH, self.hidden_size))
        result2 = Variable(torch.zeros(self.hidden_layer*2, MINI_BATCH, self.hidden_size)) if self.bidir else Variable(torch.zeros(self.hidden_layer, MINI_BATCH, self.hidden_size))
        if use_cuda:
            return (result1.cuda(), result2.cuda())
        else:
            return (result1, result2)

class AttentionClassifier(nn.Module):
    def __init__(self, INPUT_SIZE, OUTPUT_SIZE, HIDDEN_SIZE, HIDDEN_LAYER, MINI_BATCH, WINDOW_SIZE, BIDIR):
        super(AttentionClassifier, self).__init__()
        self.input_size = INPUT_SIZE
        self.output_size = OUTPUT_SIZE
        self.hidden_size = HIDDEN_SIZE
        self.hidden_layer = HIDDEN_LAYER
        self.mini_batch  = MINI_BATCH
        self.window_size = WINDOW_SIZE
        self.bidir = BIDIR

        self.attn = nn.Linear(2*HIDDEN_SIZE, WINDOW_SIZE) if BIDIR else nn.Linear(HIDDEN_SIZE, WINDOW_SIZE)
        self.attn_softmax = nn.Softmax(dim=1)
        self.attn_combine = nn.Linear(2*HIDDEN_SIZE, HIDDEN_SIZE) if BIDIR else nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE)
        self.attn_relu = nn.ReLU()
        self.dec_linear = nn.Linear(HIDDEN_SIZE, OUTPUT_SIZE)
        self.dec_softmax = nn.Softmax(dim=2)

    def forward(self, attn_hidden, enc_outs):
        attn_weights = self.attn(attn_hidden)
        attn_weights = self.attn_softmax(attn_weights)
        attn_applied = torch.bmm(attn_weights.unsqueeze(1), enc_outs)

        output = self.attn_combine(attn_applied)
        output = self.attn_relu(output)
        output = self.dec_linear(output)
        output = self.dec_softmax(output)
        return output, attn_weights

class AttentionBNClassifier(nn.Module):
    def __init__(self, INPUT_SIZE, OUTPUT_SIZE, HIDDEN_SIZE, HIDDEN_LAYER, MINI_BATCH, WINDOW_SIZE, BIDIR):
        super(AttentionBNClassifier, self).__init__()
        self.input_size = INPUT_SIZE
        self.output_size = OUTPUT_SIZE
        self.hidden_size = HIDDEN_SIZE
        self.hidden_layer = HIDDEN_LAYER
        self.mini_batch  = MINI_BATCH
        self.window_size = WINDOW_SIZE
        self.bidir = BIDIR

        self.attn = nn.Linear(2*HIDDEN_SIZE, WINDOW_SIZE) if BIDIR else nn.Linear(HIDDEN_SIZE, WINDOW_SIZE)
        self.attn_softmax = nn.Softmax(dim=1)
        self.attn_combine = nn.Linear(2*HIDDEN_SIZE, HIDDEN_SIZE) if BIDIR else nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE)
        self.attn_relu = nn.ReLU()
        self.dec_bn = nn.BatchNorm1d(HIDDEN_SIZE)
        self.dec_linear = nn.Linear(HIDDEN_SIZE, OUTPUT_SIZE)
        self.dec_relu = nn.ReLU()
        self.dec_softmax = nn.Softmax(dim=2)

    def forward(self, attn_hidden, enc_outs):
        attn_weights = self.attn(attn_hidden)
        attn_weights = self.attn_softmax(attn_weights)
        attn_applied = torch.bmm(attn_weights.unsqueeze(1), enc_outs)

        output = self.attn_combine(attn_applied)
        output = self.attn_relu(output)
        output = self.dec_bn(output.squeeze(1))
        output = self.dec_linear(output)
        output = self.dec_relu(output)
        output = self.dec_softmax(output.unsqueeze(1))
        return output, attn_weights

class AttentionLNClassifier(nn.Module):
    def __init__(self, INPUT_SIZE, OUTPUT_SIZE, HIDDEN_SIZE, HIDDEN_LAYER, MINI_BATCH, WINDOW_SIZE, BIDIR):
        super(AttentionLNClassifier, self).__init__()
        self.input_size = INPUT_SIZE
        self.output_size = OUTPUT_SIZE
        self.hidden_size = HIDDEN_SIZE
        self.hidden_layer = HIDDEN_LAYER
        self.mini_batch  = MINI_BATCH
        self.window_size = WINDOW_SIZE
        self.bidir = BIDIR

        self.attn = nn.Linear(2*HIDDEN_SIZE, WINDOW_SIZE) if BIDIR else nn.Linear(HIDDEN_SIZE, WINDOW_SIZE)
        self.attn_softmax = nn.Softmax(dim=1)
        self.attn_combine = nn.Linear(2*HIDDEN_SIZE, HIDDEN_SIZE) if BIDIR else nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE)
        self.attn_relu = nn.ReLU()
        self.dec_ln = nn.LayerNorm(HIDDEN_SIZE)
        self.dec_linear = nn.Linear(HIDDEN_SIZE, OUTPUT_SIZE)
        self.dec_relu = nn.ReLU()
        self.dec_softmax = nn.Softmax(dim=2)

    def forward(self, attn_hidden, enc_outs):
        attn_weights = self.attn(attn_hidden)
        attn_weights = self.attn_softmax(attn_weights)
        attn_applied = torch.bmm(attn_weights.unsqueeze(1), enc_outs)

        output = self.attn_combine(attn_applied)
        output = self.attn_relu(output)
        output = self.dec_ln(output.squeeze(1))
        output = self.dec_linear(output)
        output = self.dec_relu(output)
        output = self.dec_softmax(output.unsqueeze(1))
        return output, attn_weights

class AttentionLNPrevClassifier(nn.Module):
    def __init__(self, INPUT_SIZE, OUTPUT_SIZE, HIDDEN_SIZE, HIDDEN_LAYER, MINI_BATCH, WINDOW_SIZE, BIDIR):
        super(AttentionLNPrevClassifier, self).__init__()
        self.input_size = INPUT_SIZE
        self.output_size = OUTPUT_SIZE
        self.hidden_size = HIDDEN_SIZE
        self.hidden_layer = HIDDEN_LAYER
        self.mini_batch  = MINI_BATCH
        self.window_size = WINDOW_SIZE
        self.bidir = BIDIR

        self.attn = nn.Linear(2*HIDDEN_SIZE, WINDOW_SIZE) if BIDIR else nn.Linear(HIDDEN_SIZE, WINDOW_SIZE)
        self.attn_softmax = nn.Softmax(dim=1)
        self.attn_combine = nn.Linear(2*HIDDEN_SIZE+2, HIDDEN_SIZE) if BIDIR else nn.Linear(HIDDEN_SIZE+2, HIDDEN_SIZE)
        self.attn_relu = nn.ReLU()
        self.dec_ln = nn.LayerNorm(HIDDEN_SIZE)
        self.dec_linear = nn.Linear(HIDDEN_SIZE, OUTPUT_SIZE)
        self.dec_relu = nn.ReLU()
        self.dec_softmax = nn.Softmax(dim=2)

    def forward(self, attn_hidden, enc_outs, prev_info):
        attn_weights = self.attn(attn_hidden)
        attn_weights = self.attn_softmax(attn_weights)
        attn_applied = torch.bmm(attn_weights.unsqueeze(1), enc_outs)

        attn_applied = torch.cat((attn_applied, prev_info), 2)

        output = self.attn_combine(attn_applied)
        output = self.attn_relu(output)
        output = self.dec_ln(output.squeeze(1))
        output = self.dec_linear(output)
        output = self.dec_relu(output)
        output = self.dec_softmax(output.unsqueeze(1))
        return output, attn_weights

def train_multitask(input_Var, target_Var1, target_Var2, encoders, decoders, enc_opts, dec_opts, 
    loss_funcs, INPUT_SIZE, OUTPUT_SIZE, BATCH_SIZE, k=3):

    # encoder: Encoder
    # decoder: AttentionClassifier
    onEnc       = encoders[0]
    offEnc      = encoders[1]
    onDec       = decoders[0]
    offDec      = decoders[1]
    onEncOpt    = enc_opts[0]
    offEncOpt   = enc_opts[1]
    onDecOpt    = dec_opts[0]
    offDecOpt   = dec_opts[1]
    onLossFunc  = loss_funcs[0] 
    offLossFunc = loss_funcs[1] 
    
    input_batch = input_Var.size()[0]
    input_time_step = input_Var.size()[1]

    onEncOpt.zero_grad()
    onDecOpt.zero_grad()
    offEncOpt.zero_grad()
    offDecOpt.zero_grad()

    onLoss  = 0
    offLoss = 0

    window_size = 2*k+1
    
    for step in range((input_time_step//BATCH_SIZE)+1):
        if BATCH_SIZE*step > k and BATCH_SIZE*step < input_time_step - (k+1) - BATCH_SIZE:

            onEncHidden = onEnc.initHidden(BATCH_SIZE)
            offEncHidden = offEnc.initHidden(BATCH_SIZE)
            
            onEncOuts = torch.zeros(2*k+1, BATCH_SIZE, onEnc.hidden_size*2) if onEnc.bidir else torch.zeros(2*k+1, BATCH_SIZE, onEnc.hidden_size)
            #offEncOuts = torch.zeros(2*k+1, BATCH_SIZE, offEnc.hidden_size*2) if offEnc.bidir else torch.zeros(2*k+1, BATCH_SIZE, offEnc.hidden_size)

            # Onset Encode Step
            for ei in range(window_size):
                enc_out, onEncHidden = onEnc(input_Var[:, BATCH_SIZE*step-k+ei:BATCH_SIZE*step-k+ei+BATCH_SIZE, :].contiguous().view(BATCH_SIZE, 1, INPUT_SIZE), onEncHidden)
                onEncOuts[ei] = enc_out.squeeze(1).data

            # To Onset Decoder
            onEncOuts = onEncOuts.transpose(0, 1)

            onDecAttnHidden = torch.cat((onEncHidden[0][2*onEnc.hidden_layer-1], onEncHidden[0][2*onEnc.hidden_layer-2]), 1) if onEnc.bidir else onEncHidden[0][onEnc.hidden_layer-1]
            onEncOuts = Variable(onEncOuts).cuda()

            # 1 step input (cause target only 1 time step)
            onDecOut, onDecAttn = onDec(onDecAttnHidden, onEncOuts)

            # To Offset Encoder
            offEncOuts, offEncHidden = offEnc(onEncOuts, offEncHidden)

            # To Offset Decoder
            #offEncOuts = offEncOuts.transpose(0, 1)
            offDecAttnHidden = torch.cat((offEncHidden[0][2*offEnc.hidden_layer-1], offEncHidden[0][2*offEnc.hidden_layer-2]), 1) if offEnc.bidir else offEncHidden[0][offEnc.hidden_layer-1]
            offEncOuts = Variable(offEncOuts).cuda()

            offDecOut, offDecAttn = offDec(offDecAttnHidden, offEncOuts, onDecOut)
            
            for i in range(BATCH_SIZE):
                onLoss += onLossFunc(onDecOut[i].view(1, OUTPUT_SIZE), torch.max(target_Var1[:,BATCH_SIZE*step+i].contiguous().view(input_batch, 1, OUTPUT_SIZE), dim=2)[1].view(input_batch))
            
            for i in range(BATCH_SIZE):
                offLoss += offLossFunc(offDecOut[i].view(1, OUTPUT_SIZE), torch.max(target_Var2[:,BATCH_SIZE*step+i].contiguous().view(input_batch, 1, OUTPUT_SIZE), dim=2)[1].view(input_batch))
            
    #for i in range(input_batch):
    #    loss += loss_func(note_out_prob[i], torch.max(target_Var.contiguous().view(input_batch, -1, OUTPUT_SIZE), dim=2)[1].view(note_out_prob.shape[1]))
    
    Loss = onLoss + offLoss
    Loss.backward()

    onEncOpt.step()
    onDecOpt.step()
    offEncOpt.step()
    offDecOpt.step()

    return onLoss.item() / input_time_step , offLoss.item() / input_time_step
