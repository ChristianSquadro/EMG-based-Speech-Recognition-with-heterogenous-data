import numpy as np
import torch
import kenlm
import Words
import Phones
import Dictionary

class Node:
    _id = 0 

    def __init__(self,phone,words,phone_count):
        # phone is the phone associated with this node, words is a list of words
        # which correspond to this node
        # probs is the probability associated with this node
        super().__init__()
        assert type(words) is list

        self.phone = phone # may be None for start node
        self.words = [] if words is None else words
        self._phone_count = phone_count
        self.probs = np.full(phone_count+1,-np.inf) # +1 for end token
        self.children = {} # all child elements indexed by PHONE 
#         self.cont = False # does a word have a continuation? 
        self._id = Node._id
        Node._id += 1
        
    def __str__(self):
        return 'Node for phone %s, words %s, id %f' % (self.phone,[x.name for x in self.words], self._id)
        # return 'Node for phone %s' % (self.phone)

    def isWord(self):
        return len(self.words) > 0

class PrefixTree:
    def __init__(self,dictionary,phone_count):
        super().__init__()
        self._phone_count = phone_count
        root_phone = Phones.Phone(self._phone_count + 2,'<S>')
        self._root = Node(root_phone,[],self._phone_count)
        self._dictionary = dictionary

    def addPronunciation(self,pron,word):
        # pron is a list of Phones, word is a Word object
        node = self._root

        for pos,phone in enumerate(pron):
            if phone not in node.children:
                node.children[phone] = Node(phone,[],self._phone_count)
            node = node.children[phone]

        # now we are at the last node
        node.words.append(word)

    def addWord(self, word):
        assert type(word) is Words.Word
        for pron in self._dictionary.lookupProns(word):
            self.addPronunciation(pron,word)
                
    def addWords(self, words):
        for w in words:
            self.addWord(w)
                
    def getNode(self, phones):
        # phones must be a list of Phones
        node = self._root
        for p in phones:
            if p in node.children:
                node = node.children[p]
            else:
                return None
        return node
    
    def isWord(self, phones):
        node = self.getNode(phones)
        if node:
            return node.isWord()
        else:
            return False
    
    def getSuccessorPhones(self, phones):
        result = []
        node = self.getNode(phones)
        if node:
            for c in node.children.values():
                result.append(c.phone)

        return result
    
    def getSuccessorPhonesFromId(self, id):
        result = []
        node = self.getNodeFromId(id)
        if node:
            for c in node.children.values():
                result.append(c.phone)

        return result

    def getWordsForNode(self, node):
        assert node is not None
        words = []
        for ch in node.children.values():
            words.extend(self.getWordsForNode(ch))
        words.extend(node.words)
        return words

    def getWordsForPrefix(self, phones):
        words = []
        node = self.getNode(phones)
        if node is not None:
            return self.getWordsForNode(node)
        return words

    def traverseNodes(self,fun,start=None,childrenFirst=False):
        if start is None:
            start = self._root

        if not childrenFirst:
            fun(self,start)

        for chld in start.children.values():
            self.traverseNodes(fun,start=chld,childrenFirst=childrenFirst)

        if childrenFirst:
            fun(self,start)
    
    def dump(self):
        counter = 0

        def travFun(self,node):
            nonlocal counter
            counter += 1
            print(node)

        print('Before traversal counter is',counter)
        self.traverseNodes(travFun)
        print('After traversal counter is',counter)

    def getNodeFromPhone(self, phone, start=None):
        result = None
        
        if start is None:
            start = [self._root]
        
        for p in start:
            for i in p.children.values():
                if phone.idx == i.phone.idx:
                    result = i
            if result is None:
                result = self.getNodeFromPhone(phone, p.children.values())
        
        return result
    
    def getNodeFromId(self, idx, start=None):
        result = None
        
        if start is None:
            start = [self._root]
        
        if self._root._id == idx:
            return self._root

        for p in start:
            for i in p.children.values():
                if idx == i._id:
                    result = i
            if result is None:
                result = self.getNodeFromId(idx, p.children.values())
        
        return result


# probs must have shape len(nodes) * token_count
# probs are filtered so that for each row, corresponding to the node at the respective position,
# those probs which do NOT correspond to a valid continuation of a word from this node are set to -infinity
def filter_valid_cont(nodes, probs,device):
    flt_probs = torch.clone(probs)
    
    for n in range(len(nodes)):
        node = nodes[n]
        flt_probs[n] = flt_probs[n] + torch.from_numpy(node.probs).to(device)

    return flt_probs

    
# for each node, yield the child node which corresponds to child_list at the respective position
# dct handles the mapping between phones and phone ids
def node_step(old_nodes, filter_list, dct):

    new_nodes = []
    for pos in range(filter_list.shape[0]):
        this_node = old_nodes[filter_list[pos,0]]
        end_tok = this_node._phone_count
        if filter_list[pos,1] == end_tok:
            # special case: this is a finished hypo, will soon be saved, do not propagate
            assert this_node.phone.name == '<S>'
            this_child_node = this_node
        else:
            this_phone = dct.lookupPhoneByIndex(filter_list[pos,1].item())
            this_child_node = this_node.children[this_phone]
        new_nodes.append(this_child_node)

    return new_nodes
    
# MOVED TO MAIN SCIRIPT!
def old_check_words(tree, histories, flt_align, flt_state, flt_annotation, nodes, words, flt_probs, language_model, LMWeight, LMPenalty, pr, len_target, all_att_weights):
    
    for i in range(len(nodes)):
        node = nodes[i]
        
        for w in node.words:
            # resize to get element to duplicate
            dup_h = histories[i][None]
            dup_p = flt_probs[i][None]
            dup_nann = flt_annotation[i][None]
            
            # add the duplicates
            flt_probs = torch.cat((flt_probs, dup_p))
            histories = torch.cat((histories, dup_h))
            flt_annotation = torch.cat((flt_annotation, dup_nann))

            for a in range(len(flt_align)):
                dup_na = flt_align[a][i][None]
                flt_align[a] = torch.cat((flt_align[a], dup_na))
    
            for n in range(len(flt_state)):
                for l in range(len(flt_state[n])):
                    flt_state[n] = list(flt_state[n])
                    dup_ns = flt_state[n][l][i][None]
                    flt_state[n][l] = torch.cat((flt_state[n][l], dup_ns))
                    flt_state[n] = tuple(flt_state[n])
            
            new_idx = len(histories) - 1 
            # at new_idx I go back to the root for a new word (as if there was no continuation) so i save this prefix as word a
            # and save root id
            nodes.append(tree._root)

            words.append(words[i][:])
            if pr:
                print("Found the word : ")
                print(w.name)
                tmp1 = flt_probs[new_idx][len(flt_probs[i])-1]
                print("Prob before LM: ", tmp1)
            
            words[new_idx].append(w.name) # Since at new_idx i start a new word, i save the word that I have in the node
            logprob_lm = check_language_model(language_model, words[new_idx], w.name)
            cov_pen = len(w.name)
            # Update probs
            
            flt_probs[new_idx][len(flt_probs[i])-1] += (logprob_lm * LMWeight) - LMPenalty # *cov_pen
            
            if pr:
               print("Logprob LM: ", logprob_lm) 
               print("Coverage Penalty: ", cov_pen)
               print("Coverage Penalty with Penalty: ", cov_pen*LMPenalty)
               tmp2 = flt_probs[new_idx][len(flt_probs[i])-1] 
               print("Prob after LM: ", tmp2)
            
            # flt_probs[new_idx][len(flt_probs)-1] = flt_probs[new_idx][len(flt_probs)-1] + logprob_lm

    state = (flt_state,flt_align,flt_annotation)
#     hypos = histories[:,-1]
    probs = flt_probs
    accu_probs = torch.sum(probs,1)
    return histories, nodes, words, accu_probs, state, probs

def coverage_penalty():
    return 0

def check_language_model(lm, sentence):
    # tranform from list of word into a string (lm accepts only string)
    sentence= ' '.join(sentence)
    logprob = lm.score(sentence, bos = False, eos = False)
    return logprob


def init_tree(phones, voc, words):
    
    with open(phones) as f:
        phones = f.read().split()
    # phones.append('</S>') # how to add end of sentence in the tree? 
    with open(voc) as f:
        test_words = f.read().split()

    # Make sure that words appear only once
    t = set(test_words)
    test_words = list(t)
    
    d= { line.split()[0] : line.split()[1:] for line in open('descriptions/dgaddy-lexicon.txt') if line.split() != [] }

    test_dct = Dictionary.Dictionary()
    for p in phones:
        test_dct.addPhone(p)
    for w in d.keys():
        test_dct.addWord(w)
    for w,pr in d.items():
        wo = test_dct.lookupWordByName(w)
        pro = [ test_dct.lookupPhoneByName(p) for p in pr ]
        test_dct.addPronunciation(wo,pro)
    
    phone_count = len(test_dct._phonesByName)
    tree = PrefixTree(test_dct,phone_count) 
    tree.addWords([test_dct.lookupWordByName(wn) for wn in d.keys()])
    # Fill the probabilities
    tree.traverseNodes(fill_probs,start=None,childrenFirst=False)
    
    
    return tree

def get_filtered_lm(newLm, voc):
    vocabulary = open(voc).readline().split()

    voc_dict = { pos: Words.Word(pos,val) for (pos,val) in enumerate(vocabulary) }
    
    voc_dict[len(voc_dict)] = Words.Word(len(voc_dict), '<UNK>')

    all_words = [w[1].name for w in voc_dict.items()]
    all_words.append('<s>')
    all_words.append('</s>')

    unogram = open('Descriptions/1gram', 'r')
    dugram = open('Descriptions/2gram', 'r')
    trigram = open('Descriptions/3gram', 'r')
    with open(newLm, 'w') as newfile:
        
        newfile.write('1-grams:')
        for u in unogram:
            ul = u.split()
            if ul[1] in all_words:
                newfile.write(u)
        print("done with 1 grams")
        newfile.write('2-grams:')
        for d in dugram:
            dl = d.split()
            if (dl[1,self._phone_count] in all_words) and (dl[2] in all_words):
                newfile.write(d)
        print("done with 2 grams")
        newfile.write('3-grams:')
        for t in trigram:
            tl = t.split()
            if (tl[1] in all_words) and (tl[2] in all_words) and (tl[3] in all_words):
                newfile.write(t)
    
    print("done with 3 grams")

    
    



def init_language_model(lmFile):
    lm = kenlm.Model(lmFile)
    return lm


def fill_probs(tree,node):
    end_tok = tree._phone_count 
    child_id = []
    tree._root.probs[end_tok] = 0
    
    for c in node.children.keys():
        child_id.append(c.idx)
    
    for i in child_id:
        node.probs[i] = 0
    
