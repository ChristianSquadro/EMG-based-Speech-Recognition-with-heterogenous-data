class Phone:
    def __init__(self,idx,name):
        self.idx = idx # public properties
        self.name = name

    def __str__(self):
        return '(%d: %s)' % (self.idx,self.name)


