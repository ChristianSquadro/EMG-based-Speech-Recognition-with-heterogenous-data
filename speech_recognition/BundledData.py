import torch
import os
import io 
import pdb
import numpy as np

BundPath = '/home/silvia/projects/EMG/End2EndSystem/TreesData/FeatSt.LMPena4.0.LMWeig20.0'
phonesSetFile = '/home/mwand/projects/EMG/Descriptions/PhonesSetBiokit-ReducedPhones'


class Node:
    
    def __init__(self, name, question, neg, pos, unk, model):
        self.name = name
        self.info = [question, neg, pos, unk] if question is not None else None 
        self.neg = None # negative answer node
        self.pos = None # positive answer node
        self.unk = None # unknown answer node
        self.model = model # leaf 
    
    def __str__(self):
        return self.name

class Tree:
    def __init__(self):
        self._root = None

    def addNode(self, node):
        if self._root is None:
            self._root = node
        else:
            self._addNode(self._root, node)

    def _addNode(self, node, value):
        if node.info is not None:
            if node.neg is None:
                if value.name == node.info[1]:
                    node.neg = value
            else: 
                self._addNode(node.neg, value)
            
            if node.pos is None:
                if value.name == node.info[2]:
                    node.pos = value
            else:
                self._addNode(node.pos, value)
    

    def getNodeFromPhone(self, name, start):
        result = None

        if name == start.name:
            result = start
        else:
            n1 = self.getNodeFromPhone(name, start.pos)
            n2 = self.getNodeFromPhone(name, start.neg)
            print(n1)
            print(n2)
            print("..")
        return result
    

    def printFromRoot(self,node):
        if node.info is not None:
            if node.pos is not None:
                print(node.pos.name)
            if node.neg is not None:
                print(node.neg.name)
        
        self.printFromRoot(node.pos)
        self.printFromRoot(node.neg)

def getTargetFromTree(spkses,feat,original_target):
    target = []
    phones = phonesSet()
    if spkses.split("_")[0] == '901':
        bundledTreeDescTemplate = str(BundPath + '/' + 'bundled.%s.tree')
    else:
        bundledTreeDescTemplate = str(BundPath + '/' + spkses + '/' + 'bundled.%s.tree')
    thisTree = bundledTreeDescTemplate % feat
    tree = init_tree(thisTree)
    # Pattern of each line is NODE - QUESTION - NEGATIVE ANSWER - POSITIVE ANSWER - DON'T KNOW - LEAF NAME
    # print("*************")
    # print("Phones: ", phones)
    # print("Original target: ", original_target)
    for i in range(len(original_target)):
        model = consult_tree(tree, i, original_target, phones)
        target.append(model)
    
    return target

def phonesSet():
    phones = dict()
    f = open(phonesSetFile)
    for line in f:
        l = line.split()
        phones[l[0]] = l[1:]
    return phones

def consult_tree(tree, idx, target, phones):
    # for a question i need current (0), previous (-1) and next (+1)
    curr = target[idx]
    if idx == 0:
        if len(target) == 1:
            p = [None, target[idx], None]
        else:
            p = [None, target[idx], target[idx+1]]
    elif idx == len(target)-1:
        p = [target[idx-1], target[idx], None]
    else:
        p = [target[idx-1], target[idx], target[idx+1]]
    # print("p is: ", p)
    t = get_phone_target(tree, phones, p, None)
    
    return t 
    

def init_tree(tree_file):
    tree = Tree()
    t  = open(tree_file)
    
    tree_dct = {}
    for n in t:
        line = n.split()
        if line[0] != 'ROOT' and line[0] != 'hook-SIL': # for now ignore these 2 cases
            if line[-1] == '-':
                name = line[0]
                ques = line[2]
                no,yes,unk,node = line[4:]
                n = Node(name, ques, no, yes, unk, node)
            else:
                name = line[0]
                node = line[-1]
                n = Node(name, None, None, None, None, node)
            tree.addNode(n)
    
    return tree
            
def get_phone_target(tree, phones, curr, start):
    # print("Getting phone target for ", curr, "starting from ", start)
    if start is None:
        start = tree._root
    
    if start.info is not None:
        pos, val = start.info[0].split("=")
        pos = int(pos) + 1
       
        # check the "question"
        tmp = []
        for k,v in phones.items():
            if curr[pos] in v:
                # print("checking question ", curr[pos], " is in ", k, v)
                tmp.append(k) # getting all the attributes for that phone (e.g., phones, consonant, unvoiced, etc.)
        
        if val not in tmp: # negative 
            model = get_phone_target(tree, phones, curr, start.neg)   
        else: # positive
            model = get_phone_target(tree, phones, curr, start.pos)
    
    else:
        model = start.model
    # print("Returning the model ", model)
    return model 


