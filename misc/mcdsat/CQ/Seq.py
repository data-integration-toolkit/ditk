

class Seq:
    def __init__(self):
        self.i = -1
        
    def __next__(self):
        self.i = self.i+1
        return self.i

    def nuevaVar(self,y):
        # return "_"+str(self.next())
        return "_"+str(self.__next__())

    
