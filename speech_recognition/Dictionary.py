import Phones
import Words

class Dictionary:
    #### Construction and reading of initialization data #### 
    def __init__(self):
        self.clear()

    # add a single phone, raise Exception if phone is present
    def addPhone(self,name):
        if name in self._phonesByName:
            raise Exception('Cannot add already present phone %s' % name)
        phone = Phones.Phone(self._nextPhoneId,name)
        self._phonesByIndex[self._nextPhoneId] = phone
        self._phonesByName[name] = phone

        self._nextPhoneId += 1

        return phone

    # add a single word, raise Exception if already present
    def addWord(self,name):
        if name in self._wordsByName:
            raise Exception('Cannot add already present word %s' % name)
        word = Words.Word(self._nextWordId,name)
        self._wordsByIndex[self._nextWordId] = word
        self._wordsByName[name] = word

        self._nextWordId += 1

        return word

    # add a single pronunciation, where all input must be word/phone objects
    def addPronunciation(self,word,pron):
        assert type(word) is Words.Word
        for p in pron:
            assert type(p) is Phones.Phone

        # check whether pronunciation is present
        if word not in self._dictionary:
            self._dictionary[word] = [ pron ]
        else:
            self._dictionary[word].append(pron)


    # Add phones in the phones set. If skip_existing is False, raises Exception if phone already present
    def readPhonesSet(self,filename,skip_existing=False):
        with open(filename) as fid:
            firstline = fid.readline()
            phones = firstline.split()
            for pn in phones:
                if pn in self._phonesByName and skip_existing:
                    continue # otherwise let Exception rise
                self.addPhone(pn)

    # This function will silently ignore duplicates
    def readDictionary(self,filename,phoneMap):
        nextId = 0

        with open(filename) as fid:
            for line in fid:
                elements = line.split()
                wordName = elements[0]
                if phoneMap is not None:
                    pron = [ self.lookupPhoneByName(phoneMap[p]) for p in elements[1:] ]
                else:
                    pron = [ self.lookupPhoneByName(p) for p in elements[1:] ]

                # need to lookup word, create it if it does not exist
                try:
                    wordObject = self.lookupWordByName(wordName)
                except KeyError:
                    wordObject = self.addWord(wordName)

                self.addPronunciation(wordObject,pron)
                
    #### Lookup Interface #### 
    def getPhoneCount(self):
        return len(self._phonesByName)

    def getWordCount(self):
        return len(self._wordByName)

    def lookupPhoneByIndex(self,idx):
        return self._phonesByIndex[idx]

    def lookupPhoneByName(self,phoneName):
        return self._phonesByName[phoneName]

    def lookupWordByIndex(self,idx):
        return self._wordsByIndex[idx]

    def lookupWordByName(self,wordName):
        return self._wordsByName[wordName]

    def lookupProns(self,wd):
        if type(wd) is Words.Word:
            return self._dictionary[wd]
        else:
            return self._dictionary[self.lookupWordByName(wd)]

    # somewhat for the langauge model
    def getWordsByIndex(self):
        return self._wordsByIndex

    def clear(self):
        self._phonesByIndex = {} # idx -> phone
        self._phonesByName = {} # name -> phone
        self._wordsByIndex = {} # idx -> word
        self._wordsByName = {} # name -> word
        self._dictionary = {} # word object -> list of list of phone objects
        
        self._nextPhoneId = 0 # for growing the object
        self._nextWordId = 0

    def __str__(self):
        return 'Dictionary with %d phones and %d vocabulary items' % (len(self._phonesByName),len(self._wordsByName))

if __name__ == '__main__':
    phonesSetFile = '/home/mwand/projects/AudioVisual/TCDTimit/Description/PhonesSet'
    dctFile = '/home/mwand/projects/AudioVisual/TCDTimit/Description/LipreadingDict.all.01M'

    dct = Dictionary()

    dct.readPhonesSet(phonesSetFile)

    assert len(dct._phonesByName) == 77
    assert len(dct._phonesByIndex) == 77

    print('Phone AE1:',dct.lookupPhoneByName('AE1'))

    dct.readDictionary(dctFile,None)

#     assert len(dct._wordsByName) == 522 # removing duplicates
#     assert len(dct._wordsByIndex) == 522 # removing duplicates

    mightWord = dct.lookupWordByName('MIGHT')
    mightProns = dct.lookupProns(mightWord)
    mightSpelled = [ [ x.name for x in pl ] for pl in mightProns ]
    print('Word %s, entry %s, spelled as %s' % (mightWord,mightProns,mightSpelled))

    print('Printing entire dictionary')
    print(dct)

    print('Test gone ok')

# try these
#    COINCIDED  K OW2 AH0 N S AY1 D AH0 D
#    COINCIDED  K OW2 AH0 N S OY1 D AH0 D
#    COLESLAW  K OW1 L S L AA2


