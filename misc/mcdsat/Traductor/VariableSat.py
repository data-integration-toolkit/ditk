class VariableSat:
    def __init__(self, b, nom, ind):
        self.nombreVar = nom 
        self.indices = ind
        self.bit = b #Indica si es true o false la var
        s = self.nombreVar+str(self.indices)
#        self.string = self.nombreVar+str(self.indices)
        self.clave = s.__hash__()
        self.string = {True:'', False:'-'}[self.bit]+s
        

    def negarVar(self):
        return VariableSat(not self.bit, self.nombreVar, self.indices)

    def __str__(self):
        return self.string

    __repr__ = __str__

    def __hash__(self):
        return self.clave

    def __eq__(self, other):
        return self.clave == other.clave

