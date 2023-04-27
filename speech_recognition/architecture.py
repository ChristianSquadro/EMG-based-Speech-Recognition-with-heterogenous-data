import random

import torch
from torch import nn
import torch.nn.functional as F

from transformer import TransformerEncoderLayer, TransformerDecoderLayer, PositionalEncoding
from data_utils import decollate_tensor
from absl import flags
FLAGS = flags.FLAGS
flags.DEFINE_integer('model_size', 768, 'number of hidden dimensions')
flags.DEFINE_integer('num_layers', 6, 'number of layers')
flags.DEFINE_float('dropout', .2, 'dropout')
flags.DEFINE_integer('pad', 42, 'Padding value according to the position on phoneme inventory')

class ResBlock(nn.Module):
    def __init__(self, num_ins, num_outs, stride=1):
        super().__init__()

        self.conv1 = nn.Conv1d(num_ins, num_outs, 3, padding=1, stride=stride)
        self.bn1 = nn.BatchNorm1d(num_outs)
        self.conv2 = nn.Conv1d(num_outs, num_outs, 3, padding=1)
        self.bn2 = nn.BatchNorm1d(num_outs)

        if stride != 1 or num_ins != num_outs:
            self.residual_path = nn.Conv1d(num_ins, num_outs, 1, stride=stride)
            self.res_norm = nn.BatchNorm1d(num_outs)
        else:
            self.residual_path = None

    def forward(self, x):
        input_value = x

        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))

        if self.residual_path is not None:
            res = self.res_norm(self.residual_path(input_value))
        else:
            res = input_value

        return F.relu(x + res)

class Model(nn.Module):
    def __init__(self, num_features, num_outs_enc, num_outs_dec, device):
        super().__init__()

        self.conv_blocks = nn.Sequential(
            ResBlock(8, FLAGS.model_size, 2),
            ResBlock(FLAGS.model_size, FLAGS.model_size, 2),
            ResBlock(FLAGS.model_size, FLAGS.model_size, 2),
        )
        self.w_raw_in = nn.Linear(FLAGS.model_size, FLAGS.model_size)

        self.embedding_tgt = nn.Embedding(num_outs_dec + 2 , FLAGS.model_size, padding_idx=FLAGS.pad) # We need to take in consideration the embedding of <S> and <PAD> without predicting them
        self.pos_encoder = PositionalEncoding(FLAGS.model_size)

        encoder_layer = TransformerEncoderLayer(d_model=FLAGS.model_size, nhead=4, relative_positional=True, relative_positional_distance=100, dim_feedforward=3072, dropout=FLAGS.dropout)
        decoder_layer = TransformerDecoderLayer(d_model=FLAGS.model_size, nhead=4, relative_positional=False, relative_positional_distance=100, dim_feedforward=3072, dropout=FLAGS.dropout)
        self.transformerEncoder = nn.TransformerEncoder(encoder_layer, FLAGS.num_layers)
        self.transformerDecoder = nn.TransformerDecoder(decoder_layer, FLAGS.num_layers)
        self.w_out = nn.Linear(FLAGS.model_size, num_outs_dec)
        self.w_aux = nn.Linear(FLAGS.model_size, num_outs_enc)
        
        self.device=device
        
        self.tgt_key_padding_mask=None
        self.src_key_padding_mask=None
        self.memory_key_padding_mask=None
        self.tgt_mask=None

    def create_tgt_padding_mask(self, tgt):
        # input tgt of shape ()
        tgt_padding_mask = tgt == FLAGS.pad
        return tgt_padding_mask
    
    def create_src_padding_mask(self, src):
        # input tgt of shape ()
        src_padding_mask = src == FLAGS.pad
        return src_padding_mask
    
    def forward(self, x_raw, y, length_raw_signal):
        # x shape is (batch, time, electrode)
        # y shape is (batch, sequence_length)

        if self.training:
            r = random.randrange(8)
            if r > 0:
                x_raw_clone = x_raw.clone()
                x_raw_clone[:,:-r,:] = x_raw[:,r:,:] # shift left r
                x_raw_clone[:,-r:,:] = 0 #TODO should i change it with padding value???
                x_raw = x_raw_clone

        x_raw = x_raw.transpose(1,2) # put channel before time for conv
        x_raw = self.conv_blocks(x_raw)
        x_raw = x_raw.transpose(1,2)
        x_raw = self.w_raw_in(x_raw)
        x = x_raw

        #Momentary solution to handle the padding problem of VRAM overusage
        x=decollate_tensor(x, length_raw_signal)
        x=nn.utils.rnn.pad_sequence(x, batch_first=True, padding_value=FLAGS.pad)

        #Padding Target Mask and attention mask
        self.tgt_key_padding_mask = self.create_tgt_padding_mask(y).to(self.device)
        self.src_key_padding_mask = self.create_src_padding_mask(x[:,:,0]).to(self.device)
        self.memory_key_padding_mask = self.src_key_padding_mask
        self.tgt_mask = nn.Transformer.generate_square_subsequent_mask(self, y.shape[1]).to(self.device)

        #Embedding and positional encoding of tgt
        tgt=self.embedding_tgt(y)
        tgt=self.pos_encoder(tgt)
        
        x = x.transpose(0,1) # put time first
        tgt = tgt.transpose(0,1) # put sequence_length first
        x_encoder = self.transformerEncoder(x, src_key_padding_mask=self.src_key_padding_mask)
        x_decoder = self.transformerDecoder(tgt, x_encoder, memory_key_padding_mask=self.memory_key_padding_mask, tgt_key_padding_mask=self.tgt_key_padding_mask, tgt_mask=self.tgt_mask)

        x_encoder = x_encoder.transpose(0,1)
        x_decoder = x_decoder.transpose(0,1)

        
        return self.w_aux(x_encoder), self.w_out(x_decoder)
        
    def forward_separate(self, mode , x_raw=None, y=None, memory=None):
        # x shape is (batch, time, electrode)
        # y shape is (batch, sequence_length)

        if mode == 'encoder':
            if self.training:
                r = random.randrange(8)
                if r > 0:
                    x_raw_clone = x_raw.clone()
                    x_raw_clone[:,:-r,:] = x_raw[:,r:,:] # shift left r
                    x_raw_clone[:,-r:,:] = 0
                    x_raw = x_raw_clone

            x_raw = x_raw.transpose(1,2) # put channel before time for conv
            x_raw = self.conv_blocks(x_raw)
            x_raw = x_raw.transpose(1,2)
            x_raw = self.w_raw_in(x_raw)
            x = x_raw

            
            x = x.transpose(0,1) # put time first
            x_encoder = self.transformerEncoder(x)
            x_encoder = x_encoder.transpose(0,1)
            
            return x_encoder
            
        elif mode == 'decoder':
            #Embedding and positional encoding of tgt
            tgt=self.embedding_tgt(y)
            tgt=self.pos_encoder(tgt)
            
            tgt = tgt.transpose(0,1) # put sequence_length first
            memory = memory.transpose(0,1) # put sequence_length first
            x_decoder = self.transformerDecoder(tgt, memory)
            x_decoder = x_decoder.transpose(0,1)
            
            return self.w_out(x_decoder)

