import timeit
import os
import resource
import pprint
import random
# import cPickle
import pickle

from Parser import *
from Parser.CQparser import *
from CQ.Argumento import *
from CQ.Predicado import *
from CQ.SubObjetivo import *
from CQ.CQ import *
from CQ.SOComparacion import *
# from TransformarFormula import *
# from VariableSat import *
from .TransformarFormula import *
from .VariableSat import *


archVistas = ''
archCons = ''
varsV = {}
varsG = {}
varsT = {}
varsZ = {}
 
def traducir(exp, archV, archC, archVars, archTiempo, archSalida):
    #tiempo = timeit.Timer('traducir1()', "from __main__ import traducir1; import psyco; psyco.full()").timeit(1)/1
    tiempoi = resource.getrusage(resource.RUSAGE_SELF)[0]
    traducir1(exp, archV, archC, archVars, archSalida)
    tiempof = resource.getrusage(resource.RUSAGE_SELF)[0]
    
    fileobject = open(archTiempo, 'w')
    fileobject.write(str(tiempof-tiempoi))
    fileobject.close()

def traducir1(exp, archVistas, archCons, archVars, archSalida): #(archVistas, archCons):
    vistas = cargarCQ(archVistas)
    consultas = cargarCQ(archCons)
    for q in consultas:
        if exp == 'SatRW':
            transf = traducirConsultaRW(q, vistas, archSalida)
        elif exp == 'Sat':
            transf = traducirConsultaMCD(q, vistas, archSalida)
        guardarVars(transf, archVars)


def guardarVars(transf, archVars):
    fh = open(archVars,'wb')
    # cPickle.dump(transf.numeros, fh)
    pickle.dump(transf.numeros, fh)
    fh.close()


def traducirConsultaMCD(q, vistas, archSalida):
    variables, clausulas = generarTeoriaMCD(q, vistas)
    transf = TransformarFormula(variables)
    n = 1
    imprimirCopias(variables, clausulas, [], n, transf, archSalida)
    return transf


def traducirConsultaRW(q, vistas, archSalida):
    variables, clausulas = generarTeoriaMCD(q, vistas)
    transf = TransformarFormula(variables)
    n = len(q.cuerpo)
    clausulas2 = clausulasCombinarMCD(transf, n)
    imprimirCopias(variables, clausulas, clausulas2, n, transf, archSalida)
    return transf



def clausulasCombinarMCD(transf, n):
    clausulas2 = []
    for numG in range(n):
        varG = VariableSat(False, 'g', [numG])
        varG1 = VariableSat(True, 'g', [numG])
        clsPorLoMenos1 = ''
        for numCopiaX in range(n):
            numVarGX = transf.var2NumSimple(varG,numCopiaX)
            clsPorLoMenos1 = clsPorLoMenos1 + transf.var2NumSimple(varG1,numCopiaX) + ' '
            for numCopiaY in range(numCopiaX+1, n):
                numVarGY = transf.var2NumSimple(varG,numCopiaY)
                clausulas2.append(numVarGX + ' ' + numVarGY + ' 0\n')
            clausulaSimet= numVarGX+' '
            if numG > 0 and numCopiaX > 0:
                for numGmenor in range(numG):
                    varG2 = VariableSat(True, 'g', [numGmenor])                
                    clausulaSimet= clausulaSimet + transf.var2NumSimple(varG2,numCopiaX-1) + ' '
                clausulas2.append(clausulaSimet+'0\n')
            elif numG == 0 and numCopiaX == 0:
                varG2 = VariableSat(True, 'g', [0])                
                clausulas2.append(transf.var2NumSimple(varG2,0) + ' 0\n')                    
        clausulas2.append(clsPorLoMenos1+'0\n')
    return clausulas2
                


def imprimirCopias(variables, clausulas, clausulas2, numCopias, transf, archSalida):
    
    arch = open(archSalida, 'w+')
    numVars = len(variables)*numCopias
    numCl = len(clausulas)*numCopias + len(clausulas2)
    arch.write("p cnf "+ str(numVars) + " " + str(numCl) + "\n")


    for cls in clausulas2:
        arch.write(cls)

    for numCopia in range(numCopias):
        transf.formula2Num(clausulas, numCopia, arch)

    arch.write('%\n')
    arch.close()

 

def generarTeoriaMCD(q, vistas):
    global varsV
    global varsG
    global varsT
    global varsZ
    lv, c1, c2 = variablesV(q, vistas)
    lg, c3 = variablesG(q, lv)
    lt, lz, c6, c7, c8, c9, c14, ltaux = clausulas678(q, vistas)
#     print "V",varsV,lv
#     print "G",varsG,lg
#     print "T",len(varsT),len(lt)
#     print "Z",len(varsZ),len(lz)
    c12 = clausulas12(vistas, lv, lg)
    c4 = clausulas4(q,vistas,ltaux)
    
    c5, c13 = clausulas513(q,vistas,ltaux)
    c10, c11 = clausulas11(lt, lv, ltaux)
    c15 = clausulas15(q, vistas)
#     print "clausulas 1  \/ vi (por lo menos uno)"
#     pprint.pprint(c1) 
#     print "clausulas 2  -vi \/ -vj (maximo uno)"
#     pprint.pprint(c2) 
#     print "clausulas 3  \/ gk (por lo menos uno)"
#     pprint.pprint(c3) 
#     print "clausulas 4  Vm /\ tij => -tik y property 1 C2 "
#     pprint.pprint(c4) 
#     print "clausulas 5  vm => -tij (i Dist y j exist) "
#     pprint.pprint(c5)
#     print "clausulas 6  gi /\ vm => \/ zir (r subob de Vm)"
#     pprint.pprint(c6) 
#     print "clausulas 7  zir => tir"
#     pprint.pprint(c7) 
#     print "clausulas 8  gi /\ vm <= \/ zir (r subob de Vm)"
#     pprint.pprint(c8)
#     print "clausulas 9  maximo una z por vm, gi"
#     pprint.pprint(c9)
#     print "clausulas 10 t explicito"
#     pprint.pprint(c10)
#     print "clausulas 11  tik => \/ vm (si tik entonces alguna vm)"
#     pprint.pprint(c11)
#     print "clausulas 12  v_m & g_j => -g_k "
#     pprint.pprint(c12)
#     print "clausulas 13  t_ij => -t_kj"
#     pprint.pprint(c13)
#     print "clausulas 14  v_i => -gk cuando los preds son diff"
#     pprint.pprint(c14)

    variables = []
    variables = lv+ lg+ list(lt)+ lz

    clausulas =  c1 + c2 + c3 + c4 + c5 + c6 + c7 + c8 + c9 + c10 + c11 + c12 + c13 + c14 + c15
    return variables, clausulas


def clausulas15(query, vistas):
    global varsZ
    n= len(query.cuerpo)
    c15 = []
    z = 0

    for v in vistas:
        ns = len(v.cuerpo)
        for y in range(ns):
            for x1 in range(n):
                for x2 in range(x1+1, n):
                    varz1 = varsZ.get((x1,y,z))
                    varz2 = varsZ.get((x2,y,z))
                    if varz1 and varz2:
                        c15.append([varz1.negarVar(), varz2.negarVar()])                        
        z = z+1
    return c15
            


def clausulas11(lt, lv, ltaux):
    c11 = []
    for t in lt:
        c11temp = [t.negarVar()]
        for v in lv:
            if (t, v) in ltaux:
                c11temp.append(v)
        c11.append(c11temp)
    c10 = []
    # for ((t, v), varzs) in ltaux.iteritems():
    for ((t, v), varzs) in list(ltaux.items()):
        c10.append([t.negarVar(), v.negarVar()] + list(varzs))
    return c10, c11

def clausulas4(q, vistas, lt):
    c4 = []
    nv = 0
    for v in vistas:
        varstodas = v.variables()
        varsexist = v.varsExist
        for varq in q.variables():
            for var in varsexist:
                vista = vistas[nv]
                clausula4(q, varq, nv, var, varstodas, lt, c4)
        nv = nv + 1
    return c4

def clausula4(q, varq, vis, varj, varO, lt, c4temp):
    global varsV
    global varsT
    varv = varsV[vis]
    varqi = int(varq)
    varji = int(varj)
    variableij = varsT.get((varqi, varji))

    if (variableij, varv) in lt:
        clausulag4(q, varq, varv, variableij, c4temp)
        for vark in varO:
            varki = int(vark)
            if vark != varj:
                variableik = varsT.get((varqi, varki))
                if (variableik, varv) in lt:
                    c4temp.append([varv.negarVar(),
                                   variableij.negarVar(),
                                   variableik.negarVar()
                                   ])


def clausulag4(q, varq, varv, variableij,c4temp):
    global varsG
    g = 0
    for subOb in q.cuerpo:
        if varq in subOb.argumentos:
            c4temp.append([varv.negarVar(),
                           variableij.negarVar(),
                           varsG[g]
                           ])
        g = g + 1

def clausulas513(q, vistas, lt):
    global varsV
    global varsT
    c5 = []
    nv = 0
    c13 = []
    varsE = q.varsExistL
    n = len(varsE)
    for v in vistas:
        varvt = varsV[nv]
        for varv in v.variables():
            if not v.esVarDisting(varv):
                for varq in q.varsDist:
                    vart = varsT.get((int(varq), int(varv)))
                    if (vart, varvt) in lt:
                        c5.append([varvt.negarVar(),
                                   vart.negarVar()
                                   ])

            
            for x in range(n):
                vartx = varsT.get((int(varsE[x]), int(varv)))
                if (vartx, varvt) in lt :
                    for y in range(x+1,n):
                        varty = varsT.get((int(varsE[y]), int(varv)))
                        if (varty, varvt) in lt:
                            c13.append([vartx.negarVar(),
                                        varty.negarVar(),
                                        varvt.negarVar()
                                        ])
            
        nv = nv + 1
    return c5, c13

def variablesV(q, vistas):
    global varsV
    vnula = VariableSat(True, 'v', [-1])
    lv =[vnula]
    varsV[-1] = vnula
    c2 = []
    numVista = 0
    for v in vistas:
        var = VariableSat(True, 'v', [numVista])
        lv.append(var)
        varsV[numVista]=var
        numVista = numVista + 1
    n = len(lv)
    for x in range(n):
        for y in range(x+1,n):
            c2.append([lv[x].negarVar(), lv[y].negarVar()])
    return lv, [lv], c2


def variablesG(q, lv):
    global varsG
    lg =[]
    c3 = []
    numSO = 0
    for v in q.cuerpo:
        var = VariableSat(True, 'g', [numSO])
        lg.append(var)
        varsG[numSO]=var
        numSO = numSO + 1
    for g in lg:
        c3temp = [lv[0].negarVar(), g.negarVar()]
        c3.append(c3temp)                
    for v in lv[1:]:
        c3temp=[v.negarVar()] +lg
        c3.append(c3temp)        
    return lg, c3

#v_m & g_j => -g_k
def clausulas12(vistas, lv, lg):
    c12 = []
    n = len(lg)
    for i in range(1, len(lv)):
        vista = vistas[lv[i].indices[0]]
        if vista.todasVarDisting():
            for x in range(n):
                for y in range(x+1,n):
                    #pass
                    c12.append([lv[i].negarVar(), lg[x].negarVar(), lg[y].negarVar()])                    
    return c12


def clausulas678(q, vistas):
    global varsT
    global varsZ
    global varsG
    global varsV
    c6 = []
    c7 = []
    c8 = []
    c9 = []
    c10 = []
    lz = []
    lt = set([])
    ltaux = {}
    c6temp = []
    c14temp = {}
    i = 0
    for subOb in q.cuerpo:
        pred = subOb.predicado
        m = 0
        varg = varsG[i]
        for v in vistas:
            j = 0
            varm = varsV[m]
            c6temp = [varg.negarVar(), varm.negarVar()]
            c9temp = []
            subObCubre = False
            for subObtemp in v.cuerpo:
                predtemp = subObtemp.predicado
                if pred == predtemp:
                    varz = VariableSat(True, 'z', [i,j,m])
                    lz.append(varz)
                    varsZ[(i,j,m)]=varz
                    c6temp.append(varz)
                    c9temp.append(varz)
                    lttemp = clausula78a(varz, varg, varm, subOb, subObtemp, m, ltaux, c7, c8)
                    lt |= lttemp
                    subObCubre = True
                j = j + 1
            c9 = c9 + clausula9(c9temp)
            m = m + 1
            if len(c6temp) > 2:
                c6.append(c6temp)
            if not subObCubre == True:
                c14temp.setdefault(varm.negarVar(),[]).append(varg.negarVar())
        i = i+1
    return lt, lz, c6, c7, c8, c9, clausula14(c14temp), ltaux


def clausula14(c14temp):
    c14 = []
    for (varm, listvarg) in list(c14temp.items()):
        for varg in listvarg:
            c14.append([varg, varm])
    return c14


def clausula9(c9temp):
    c9 = []
    n = len(c9temp)
    for x in range(n):
        for y in range(x+1,n):
           c9.append([c9temp[x].negarVar(), c9temp[y].negarVar()])
    return c9



def clausula78a(varz, varg, varm, subObQ, subObV, vis, ltaux, c7, c8):
    global varsT
    c8temp1 = [varz, varm.negarVar()]
    c8temp2 = [varm]
    lt = set([])
    i = 0
    for x in subObQ.orden:
        y = subObV.orden[i]
        varT = VariableSat(True, 't', [int(x), int(y)])
        lt.add(varT)
        varsT[(int(x), int(y))]=varT
        c7temp = [varm.negarVar(), varz.negarVar(), varT]
        c7.append(c7temp)
        c8temp1.append(varT.negarVar())
        c8temp2.append(varT.negarVar())
        i = i + 1
        ltaux.setdefault((varT,varm),set([])).add(varz)    
    c8.append([varz.negarVar(), varg])
    c8.append([varz.negarVar(), varm])
    return lt

