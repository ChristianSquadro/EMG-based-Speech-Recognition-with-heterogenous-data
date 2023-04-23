class Word:
    def __init__(self,idx,name):
#         # the pronunciations should be a list of lists
#         assert type(pronunciations) is list
#         for x in pronunciations:
#             assert type(x) is list
        self.idx = idx
        self.name = name
#         self.pronunciations = pronunciations


    def __str__(self):
        return '(%d: %s)' % (self.idx,self.name)


