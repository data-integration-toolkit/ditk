import pprint

class TransformarFormula:
    variables = {}
    numeros = {}

    def __init__(self, listaVars):
        self.numVar = 0
        for x in listaVars:
            self.varNueva(x)
        self.n = len(self.variables)
        #pprint.pprint(self.variables)

    def varNueva(self, var):
        self.numVar = self.numVar + 1
        snumVar = str(self.numVar)
        self.variables[var] = self.numVar
        #snumVar
        self.numeros[self.numVar] = str(var)
#         return str(snumVar)

    def var2Num(self, var, varsCopia):
        if not var.bit:
            return '-'+varsCopia[var]
        else:
            return varsCopia[var]

    def var2NumSimple(self, var, numCopia):
        if not var.bit:
            neg = '-'
        else:
            neg = ''
        return neg+str(self.n*numCopia+self.variables[var])


    def cl2Num(self, cl, varsCopia):
        ret = ''
        for x in cl:
            ret = ret + str(self.var2Num(x, varsCopia)) + ' ' 
        return ret + '0\n'

    def formula2Num(self, formula, numCopia, arch):
        ret = ''
        varsCopia = {}
        desp = self.n*numCopia
        for x, y in list(self.variables.items()):
            varsCopia[x]=str(desp + y)
            
        for x in formula:
            arch.write(self.cl2Num(x, varsCopia))
        return ret

    # Hace el cambio de vuelta a partir del modelo generado por zchaff
    def num2Var(self, num2):
        if num2[0] == '-':
            num = num2[1:]
            neg = '-'
        else:
            num = num2
            neg = ''
        return neg + self.numeros[num]

    def mod2Var(self, mod):
        ret = []
        for x in mod:
            ret.append(self.num2Var(x))
        ret.sort()
        return ret
 
