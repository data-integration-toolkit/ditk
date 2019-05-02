# from sets import set
class MCD:
    ## hc: head homomorfismo (dict)
    ## phic: mapping vars(Q) en vars(V) (dict)
    ## Gc: subobjetivos cubiertos (set) 
    def __init__(self, phictodo, gc, vista):
        self.vista = vista
        self.phictodo = phictodo
        self.phic, self.hc = self.calcularPhiH()
        self.vistaH = vista.map_variables(self.hc)
        self.gc = gc
        self.clave = "V:"  + self.vista.cabeza.predicado + " tau:" + str(self.phictodo) + " phic:" + str(self.phic) + " hc:" + str(self.hc) + " gc:" + str(list(self.gc))
        self.string = "V:"  + self.vista.cabeza.predicado + " tau:" + str(self.phictodo) + " gc:" + str(list(self.gc))
        self.ec, self.phi_1 = self.calcularEc()

    # OJO CON ESTO PQ EL STRING DE UNA TABLA DE HASH NO ES NECESARIAMENTE EL MISMO SIEMPRE
    def __str__(self):
        return self.string

    __repr__ = __str__

    def __hash__(self):
        return self.clave.__hash__()

    def __eq__(self, other):
        return self.clave == other.clave


#     def calcularPhiH(self):
#         h = {}
#         phi = {}
#         error = False
#         #print "phictodo", self.phictodo
#         for var in self.phictodo:
#             rep = self.phictodo[var][0]
#             #print rep, "rep"
#             if len(self.phictodo[var]) > 1:
#                 for i in xrange(1,len(self.phictodo[var])):
#                     x = self.phictodo[var][i]
#             phi[var] = rep
#         return phi, h


    def calcularPhiH(self):
        h = {}
        phi = {}
        error = False
        phictodo = self.phictodo
        for var in phictodo:
            rep = phictodo[var][0]
            if len(phictodo[var]) > 1:
                for i in range(1,len(phictodo[var])):
                    x = phictodo[var][i]
                    if x in h:
                        h[rep]=h[x]
                        rep = h[x]

                for i in range(1,len(phictodo[var])):
                    x = phictodo[var][i]
                    h[x] = rep

            phi[var] = rep

        self.clausura(h)

        for q,v in list(phi.items()):
            if v in h:
                phi[q] = h[v]

        return phi, h



    def clausura(self, h):    
        for i in h:
            for x,y in list(h.items()):
                if i == y:
                    h[x] = h[i]



    def calcularEc(self):
        phi_1 = {}
        for var in self.phic:
            pvar = self.phic[var]
            if pvar in phi_1:
                phi_1[pvar].append(var)
            else:
                phi_1[pvar] = [var]
        ec = {}
        for var in phi_1:
            rep = phi_1[var][0]
            if len(phi_1[var]) > 1:
                for x in phi_1[var]:
                    ec[x] = rep
            else:
                ec[rep] = rep
        ###print "ec", ec
        return ec, phi_1



    def obtUnificacion(self, ecgeneral):
        ret = {}
        for y in self.phi_1:
            x = self.phi_1[y][0]
            if x in ecgeneral:
                x = ecgeneral[x]
            ret[y] = x
        return ret
    
