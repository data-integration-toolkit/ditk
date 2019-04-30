# from sets import set

#from SubObjetivo import *
#from Seq import *
from .SubObjetivo import *
from .Seq import *

class CQ:
    def __init__(self, cab, cuer, comp):
        self.cabeza = cab
        self.cuerpo = cuer
        self.comparacion = comp
        self.vars,self.varsExist,self.varsDist = self.obtVariables();
        self.varsExistL = list(self.varsExist)

    def variables(self):
        return self.vars

    def obtVariables(self):
        vars = set()
        varsExist = set()
        varsDist = set()
        for so in self.cuerpo:
            for x in so.orden:
                vars.add(x)
                if self.esVarDisting(x):
                    varsDist.add(x)
                else:
                    varsExist.add(x)
        return vars, varsExist, varsDist

    def esSeguro(self):
        varsDisting = self.cabeza.argumentos
        for arg in list(varsDisting.keys()):
            if varsDisting[arg] == 0:
                if (not self.estaVarEnCuerpo(arg)):
                    return False
        return True    

    def estaVarEnCuerpo(self, arg_tipo):
        for subObj in self.cuerpo:
            if subObj.esVarArgumento(arg_tipo):
                return True
        return False

    def __str__(self):
        return str(self.cabeza) + " :- " + ", ".join(str(x) for x in self.cuerpo)

    __repr__ = __str__

    def map_variables(self, psi):
        cuer = []
        comp = []
        cab = self.cabeza.map_variables(psi)
        for subob in self.cuerpo:
            cuer.append(subob.map_variables(psi))
        for order in self.comparacion:
            comp.append(order.map_variables(psi))
        return CQ(cab, cuer, comp)

    #igual que map_variables pero si encuentra una variable que no se encuentre en
    #el dominio de psi, la sustituye por una variable nueva
    def map_variables2(self, psi, seq):
        cuer = []
        comp = []
        cab = self.cabeza.map_variables2(psi, seq)
        for subob in self.cuerpo:
            cuer.append(subob.map_variables2(psi, seq))
        for order in self.comparacion: 
            comp.append(order.map_variables(psi))#ojo con esto
        return CQ(cab, cuer, comp)

    def esVarDisting(self, var):
        # return self.cabeza.argumentos.has_key(var)
        return var in self.cabeza.argumentos

    def todasVarDisting(self):
        return len(self.cabeza.argumentos) == len(self.vars)
    
    def imprimir(self, dic):
        # for x in dic.keys():
        #     print x, map(str, dic[x])
        for x in list(dic.keys()):
            print((x, list(map(str, dic[x]))))
        
    def obtSubObXVar(self, var):
        res = set([])
        i = 0
        for subOb in self.cuerpo:
            # if subOb.argumentos.has_key(var):
            if var in subob.argumentos:
                res.add(i)
        return res
                
