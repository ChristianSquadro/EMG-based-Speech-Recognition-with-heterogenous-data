import torch
from data_utils import PhoneTransform
from absl import flags
import numpy as np
FLAGS = flags.FLAGS

def run_greedy(model, X_raw, tgt, vocab_size, device):
  model.eval()
  batch_len=tgt.shape[0]
  total_logits= torch.zeros(batch_len,0,vocab_size,dtype=torch.float32,device=device)
  phones_seq = [[] for _ in range(batch_len)]
  start_tok = vocab_size - 2
  max_seq_length= tgt.shape[1] + 1 
  dec_input = torch.full([batch_len, 1], start_tok).to(device)
  phone_transform = PhoneTransform()

  # forward pass, attention is applied to data_encoded as trained
  memory = model(mode= 'greedy_search', part='encoder', x_raw= X_raw)

  while True:
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
    total_logits=torch.cat((total_logits, step_logits[:,-1,:].reshape(batch_len, 1, vocab_size)), dim=1)

    #Formatting phones
    new_phones_seq = [ ' '.join(item) for item in phones_seq ]

    #Stopping Criteria
    if all([item[-1] == '</S>' for item in phones_seq]) or dec_input.shape[1] == max_seq_length:
      break    

  return memory, total_logits, new_phones_seq
