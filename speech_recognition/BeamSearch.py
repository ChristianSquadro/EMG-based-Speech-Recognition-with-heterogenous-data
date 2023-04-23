import torch
import torch.nn as nn
import torch.optim
import numpy as np
import collections
import matplotlib.pyplot as plt
import PrefixTree

PRINT_DUP = False
PRINT_HYP = False
PRINT_FIN = False

from absl import flags
FLAGS = flags.FLAGS
flags.DEFINE_integer('BeamWidth', 100, 'width for pruning the prefix_tree')
flags.DEFINE_boolean('Constrained', True, 'flag to enable language model and vocaboulary')
flags.DEFINE_float('LM_Weight', 0.9 , 'importance for language model scoring')
flags.DEFINE_float('LMPenalty', -1.0, 'penalty to penalize short words insertion')
    
# Helpers
def replicate(l,t):
    # replicate each element of l t times
    return [x for x in l for _ in range(t)]


def decode(l,dct,start_tok,end_tok):
    # decode into phones, skip start token, finish at end token
    result = []
    for p in l:
        if p == start_tok:
            continue
        if p == end_tok:
            break
        result.append(dct.lookupPhoneByIndex(p).name)
    return result

HypoHolder = collections.namedtuple('HypoHolder',['histories','probs','memory','words','nodes'])

def run_single_bs(model,data,target,vocab_size,tree,language_model,device):
    def check_hypos_are_consistent(h,step):
        assert h.histories.ndim == 2 and h.histories.shape[1] == (step+1)
        assert len(h.nodes) == h.histories.shape[0]
        assert h.probs.shape[0] == h.histories.shape[0]
        assert h.probs.shape[1] == step
        assert len(h.words) == h.histories.shape[0]

        #assert len(h.memory[0]) == 1 # no idea why we have an extra list with only one element here
        # note: state is always (hidden_state,att,annotation) (some of these might be unused)
        # the hidden state is a LIST of one element, that element is a tuple (for the two-part hidden state),
        # each tuple element has the number of current histories as first dimension
        #real_state = h.state[0][0]
        #assert real_state[0].shape[0] == h.histories.shape[0]
        #assert real_state[1].shape[0] == h.histories.shape[0]

    def update_hypos(old_hypos,filter_list,this_step_probs,unfiltered_new_state,dct):
        assert type(old_hypos) == HypoHolder
        assert filter_list.shape[1] == 2 # 
        # note: we have to update the elements of old_hypos: histories, probs, state, aligns, words, nodes
        # the inputs for this step are this_step_probbs and unfiltered_new_state

        # create new histories by merging filtered token histories and new tokens
        pre_histories = old_hypos.histories[filter_list[:,0]]
        best_tokens = filter_list[:,1][:,None]
        new_histories = torch.cat([pre_histories, best_tokens],1)

        # create new probs by adding filtered probs and new recognition probs
        pre_probs = old_hypos.probs[filter_list[:,0]]
        flt_probs = this_step_probs[filter_list[:,0],filter_list[:,1]][:,None]
        new_probs = torch.cat([pre_probs,flt_probs],1)
    
        new_state = filter_state(unfiltered_new_state,filter_list[:,0])
        flt_att = new_state[1]

        # filter and extend alignments
        pre_align = [ old_hypos.aligns[p] for p in filter_list[:,0] ]
        assert len(pre_align) == flt_att[0].shape[0]
        head_count = len(flt_att)
        new_att = []
        for pos in range(len(pre_align)):
            this_new_att = [ flt_att[h][pos] for h in range(head_count) ]
            new_att.append(this_new_att)

        new_aligns = [ (pr + n) for (pr,n) in zip(pre_align,new_att) ]

        if FLAGS.Constrained:
            # filter words, but do not add new words
            new_words = [old_hypos.words[index] for index in filter_list[:,0]]
            new_nodes = PrefixTree.node_step(old_hypos.nodes,filter_list,dct)
            # Get current nodes
        else:
            raise Exception('This is going to crash because of the nodes. Do I need to keep track of them at all?')
            state = (flt_state,flt_align,flt_annotation)
            accu_probs = torch.sum(flt_probs,1) 
            probs = flt_probs

        new_hypos = HypoHolder(histories=new_histories,probs=new_probs,state=new_state,aligns=new_aligns,words=new_words,nodes=new_nodes)
        return new_hypos


    ### MAIN PART
    dct = tree._dictionary
    pr = False # print outputs?
    pr2 = False
   
    # forward pass, attention is applied to data_encoded as trained
    memory = model.forward_separate('encoder', x_raw= data)

    # prepare some constants
    start_tok = vocab_size - 1
    end_tok = vocab_size - 2
    max_len = torch.sum(target != end_tok) + 20
    
    # initialize

    # create the initial hypo
    hypos = HypoHolder(
            histories = torch.tensor([[ start_tok ]],device=device) ,
            probs = torch.zeros(1,0,dtype=torch.float32,device=device),
            memory= memory,
            words = [[]], 
            nodes = [tree._root] 
        )

    finished_hypos = {}
        # holds all the FINISHED hypos (end token reached) - the only thing which counts at the end

    for step in range(max_len):
        check_hypos_are_consistent(hypos,step)
        if pr2:
            print('--- BEGIN STEP %d ---' % step)

        # start here
        last_frame_hypo = hypos.histories[:,-1] # hypos is always the MOST RECENT hypo in the history
        
        # decode_step treats the different hypos as though they were different elements of a batch
        step_logits, new_state = model.forward_separate('decoder',y=last_frame_hypo,memory=hypos.memory)

        # step_logits and step_probs have the shape (hypos * tokens)
        step_probs = torch.nn.functional.log_softmax(step_logits,1)
        
        if step == 0:
            full_probs = step_probs
        else:
            full_probs = step_probs + torch.sum(hypos.probs,1,keepdim=True)
        
        if FLAGS.Constrained:
            old_full_probs = torch.clone(full_probs) # not needed
            full_probs = PrefixTree.filter_valid_cont(hypos.nodes, full_probs,device) 
            # this step sets all possible combinations of hypo and new phone to zero which do NOT correspond to valid words
            
        # Compute the best hypos (requires some shape juggling), make sure that hypos with probability -inf are never taken
        pre_top_hypos = torch.topk(full_probs.flatten(),min(FLAGS.BeamWidth,torch.sum(torch.isfinite(full_probs)))).indices 
        top_hypos = torch.stack([pre_top_hypos // full_probs.shape[1],pre_top_hypos % full_probs.shape[1]],1) # rescaling
        assert top_hypos.shape[1] == 2 # numerical indexes, each row has the form (first idx, second idx) to index a two-dimensional matrix
        
        # now update the relevant variables which carry hypo information and must be aligned: histories, probs, state, align, words, nodes (the latter two only for constrained search)

        # normal propagation
        new_hypos = update_hypos(hypos,top_hypos,step_probs,new_state,dct)

        # save and remove finished hypos
        unfinished_hypos = save_finished_hypos(new_hypos,finished_hypos,end_tok, language_model, target.shape[1]) 
        assert not torch.any(unfinished_hypos.histories[:,-1] == end_tok)

        # propagate across word boundaries
        expanded_hypos = check_words(tree,unfinished_hypos,language_model, device)
        if pr2:
            print('check_words for step %d: %d -> %d' % (step,unfinished_hypos.histories.shape[0],expanded_hypos.histories.shape[0]))

        # end of step
        hypos = expanded_hypos
        if pr2:
            print('Step %d, four top hypos:' % step)
            for pos in range(4):
                print(torch.sum(hypos.probs[pos]).item(),[x.name for x in hypos.words[pos]],[ (tree._dictionary.lookupPhoneByIndex(i.item()).name if i != 44 else '##' ) for i in hypos.histories[pos] ])
        if PRINT_HYP:
            print('END OF STEP',step)
            for i in range(hypos.histories.shape[0]):
                node = hypos.nodes[i]       
                print('    hypo %d, prob %.2f, node %d,words %s' % (i,torch.sum(hypos.probs[i]).item(),node._id,[x.name for x in hypos.words[i]]))       
        if PRINT_FIN:
            print('FINISHED HYPOS AFTER STEP %d (total %d)' % (step,len(finished_hypos)))
            f_keys = list(sorted(finished_hypos.keys(),reverse=True))
            for k in f_keys[:5]:
                print('   %f: %s, %s' % (k,finished_hypos[k][0].tolist(),finished_hypos[k][1]))



    # NEW
    save_finished_hypos(hypos, finished_hypos, end_tok, language_model, target.shape[1])
    keys = [x for x in finished_hypos.keys()]
    max_prob = np.max(keys)    
    
    return finished_hypos[max_prob][0],max_prob,finished_hypos[max_prob][1] # return the first because it is the most likely


# Save all finished hypos from the hypos (to be recognized by end_tok as the last token in the history).
# finished_hypos is where finished hypos are permanently collected, we return a hypo holder with the remaining active hypos
def save_finished_hypos(hypos,finished_hypos,end_tok, language_model, len_target):
    assert type(hypos) is HypoHolder

    end_reached = (hypos.histories[:,-1] == end_tok)
    end_reached_pos = torch.where(end_reached)[0]
    active = torch.logical_not(end_reached)
    active_pos = torch.where(active)[0]

    # let's make this simple, will see later how it goes best
    remaining_histories = hypos.histories[active]
    remaining_probs = hypos.probs[active]
    remaining_state = filter_state(hypos.state,active_pos)
    remaining_aligns = [ hypos.aligns[p] for p in active_pos ]
    remaining_words = [ hypos.words[p] for p in active_pos ]
    remaining_nodes = [ hypos.nodes[p] for p in active_pos ]
    
    for p in end_reached_pos:
        save_finished_hypo(finished_hypos,hypos.histories[p],hypos.probs[p],None,None,hypos.words[p],None,language_model,len_target)

    remaining_hypos = HypoHolder(
            histories = remaining_histories,
            probs = remaining_probs,
            state = remaining_state,
            aligns = remaining_aligns,
            words = remaining_words,
            nodes = remaining_nodes
            )
    return remaining_hypos

def save_finished_hypo(finished_hypos,history, probs, state, align, words, nodes, language_model, len_target):
    score = 0

    # logP(Y|X) where X is the source, Y is the current target ??
        # coverage_penalty(len(finished_histories[i]), probs[i], max_len)
#     n = normalize_length(len(history))
    
    score = language_model.getLogProb('</s>',tuple(words), maxN=3)
# #     cov_pen = PrefixTree.coverage_penalty()
#     prefinal_prob = torch.mean(probs)
#     final_prob = prefinal_prob + (score * LMWeight) + LMPenalty
    final_prob = torch.clone(probs)
    final_prob[-1] += (score * FLAGS.LMWeight) + FLAGS.LMPenalty

    if FLAGS.constrained:
        tup = (history,[x.name for x in words])
    else:
        tup = (history, [])
    
    finished_hypos[torch.mean(final_prob).item()] = tup

# filter the state tuple of a hypo holder according to an array of positions (as integers)
def filter_state(state,pos_list):
    hidden,att,annotation = state
    if type(hidden[0]) is tuple: # depends on the kind of RNN cell
        flt_hidden = [ (x[pos_list],y[pos_list]) for (x,y) in hidden ]
    else:
        flt_hidden = [ x[pos_list] for x in hidden ]
    
    flt_att = [na[pos_list] for na in att] # highly advanced indexing, no idea what this does 
    flt_annotation = annotation[pos_list]

    new_state = (flt_hidden,flt_att,flt_annotation)
    return new_state

# Propagate hypos across word boundaries. This works by duplicating each hypo which is at a word boundary to the tree root, adding the language model probability
# and the new word; everything else remains unchanged. Should be done as a final step after pruning. The existing hypos will remain as they are
def check_words(tree, hypos, language_model,device):
    hypo_count = hypos.histories.shape[0]
 
    # the parts of the hypo which do NOT change can simply be filtered
    filter_positions = list(range(hypo_count))
    
    # this is what's going to change, only save the parts which will be concatenated
    new_probs = []
    new_words = []
    new_nodes = []

    added_word_count = 0
    for hypo_pos in range(hypo_count):
        if PRINT_DUP:
            if len(hypos.nodes[hypo_pos].words) == 0:
                print('    dupword: hypo %d, prob %.2f, node %d, word -----' % (hypo_pos,torch.mean(hypos.probs[hypo_pos]).item(),hypos.nodes[hypo_pos]._id))
        for wd in hypos.nodes[hypo_pos].words:
            if PRINT_DUP:
                print('    dupword: hypo %d, prob %.2f, node %d, word %s' % (hypo_pos,torch.mean(hypos.probs[hypo_pos]).item(),hypos.nodes[hypo_pos]._id,wd))

            # compute lm probability
            logprob_lm = PrefixTree.check_language_model(language_model, [ x.name for x in hypos.words[hypo_pos]] + [ wd.name ])

            # collect info for new hypo
            cp_probs = torch.clone(hypos.probs[hypo_pos])
            cp_probs[-1] += (logprob_lm * FLAGS.LMWeight) + FLAGS.LMPenalty
            new_probs.append(cp_probs)
            new_words.append(hypos.words[hypo_pos] + [ wd ])
            new_nodes.append(tree._root)

            filter_positions.append(hypo_pos)
            added_word_count += 1

    # special case
    if added_word_count == 0:
        return hypos # unchanged!

    # collect and return
    filter_positions = torch.tensor(filter_positions,device=device)
    joint_histories = hypos.histories[filter_positions]
    joint_state = filter_state(hypos.state,filter_positions)
    joint_probs = torch.cat([hypos.probs,torch.stack(new_probs,0)],0)
    joint_aligns = [ hypos.aligns[p] for p in filter_positions ]
    joint_words = hypos.words + new_words
    joint_nodes = hypos.nodes + new_nodes
#     print('ADDED %d WORDS, %d hypos -> %d hypos' % (added_word_count,hypos.histories.shape[0],joint_histories.shape[0]))
            
    joint_hypos = HypoHolder(
            histories = joint_histories,
            probs = joint_probs,
            state = joint_state,
            aligns = joint_aligns,
            words = joint_words,
            nodes = joint_nodes
            )
    return joint_hypos