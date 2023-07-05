import torch
from data_utils import PhoneTransform
from absl import flags
import numpy as np
FLAGS = flags.FLAGS

def run_greedy(model, X_raw, tgt, vocab_size, device):
  batch_len=tgt.shape[0]
  total_logits= torch.zeros(batch_len,1,vocab_size,dtype=torch.float32)
  total_logits[:,0,40] = 1. #adding probability of 1 to the start token
  phones_seq = [['<S>'] for _ in range(batch_len)]
  start_tok = vocab_size - 3
  max_seq_length= tgt.shape[1]  
  dec_input = torch.full([batch_len, 1], start_tok).to(device)
  phone_transform = PhoneTransform()

  # forward pass, attention is applied to data_encoded as trained
  memory, out_enc = model(mode= 'greedy_search', part='encoder', x_raw= X_raw)

  with torch.no_grad():
    while True:
      torch.cuda.empty_cache()
      #Decoder
      step_logits = model(mode='greedy_search', part='decoder', y=dec_input, memory=memory)
      probs = torch.nn.functional.softmax(step_logits, dim=2)
      predicted_idx = torch.argmax(probs, dim=2)[:,-1]

      #Adding the new character
      for i in range(predicted_idx.shape[0]):
        if len(phones_seq[i]) == 0:
          phones_seq[i].append(phone_transform.int_to_phone([predicted_idx[i]]))
        elif not(phones_seq[i][-1] == '</S>'):
          phones_seq[i].append(phone_transform.int_to_phone([predicted_idx[i]]))

      #Concatenate the decoder input sequence
      predicted_idx=predicted_idx.reshape(batch_len, 1)
      dec_input=torch.cat((dec_input, predicted_idx), dim=1)
      total_logits=torch.cat((total_logits, step_logits[:,-1,:].reshape(batch_len, 1, vocab_size).to('cpu')), dim=1)

      #Stopping Criteria
      if all([any(phone == '</S>' for phone in item ) for item in phones_seq]) or dec_input.shape[1] == max_seq_length:
        break
    
    #Formatting phones
    phones_seq = [ ' '.join(item) for item in phones_seq ]
    
    #Padding for losses
    new_logits_seq= torch.zeros(batch_len,max_seq_length,vocab_size,dtype=torch.float32)
    new_logits_seq[:, :, 42] = 1.
    new_logits_seq[:, :total_logits.shape[1], :] = total_logits

  return out_enc, new_logits_seq, phones_seq
