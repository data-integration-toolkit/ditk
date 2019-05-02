 #!/usr/bin/env python

import tpg
from CQ.CQ import *
from CQ.SubObjetivo import *

class CQparser(tpg.Parser):
    r"""

        separator space '\s+';

        token var      '[A-Z][A-Za-z0-9]*';
        token ctte       '[0-9_]+';
        token nomb      '[a-z][a-z0-9]*';
        token imp       ':[\-]';

        START/e -> EXPERIMENTO/e ;

        EXPERIMENTO/l ->  
                            $ l = []
            CQ/a            $ l.append(a)  
            ( CQ/a      $ l.append(a)  
            )*  
        ;

        CQ/e -> SO/h
                imp
                CUERPO/c    $ e = crearCQ(h, c)
                ;
        
        CUERPO/l ->  
                            $ l = []
            SO/a            $ l.append(a)  
            ( ',' SO/a      $ l.append(a)  
            )*  
        ;

        SO/p -> nomb/n      
                LIST/l      $ p = crearSO(n, l)
                ;

        LIST/l ->  
        '\('                
                            $ l = [] 
            ITEM/a          $ l.append(a)  
            ( ',' ITEM/a    $ l.append(a)  
            )*  
        '\)'  
        ;

        ITEM/a ->
                    var/v   $ a = (v[1:],0)
                |   ctte/c  $ a = (c,1)
                ;

    """

def crearSO(pred, lista):
    ord = []
    arg = {}
    for (x,y) in lista:
        ord.append(x)
        arg[x]=y
    return SubObjetivo(pred, arg, ord)

# falta colocarle los predicados de orden
def crearCQ(cabeza, cuerpo):
    return CQ(cabeza, cuerpo, [])

def cargarCQ(nomArch):
    parser = CQparser()
    in_file = open(nomArch,"r")
    text = in_file.read()
    result = parser(text)
    in_file.close()
    return result

##
##while 1:
##        expr = raw_input('cq: ')
##        if not expr: break
##        #try:
##        print expr
##        result = parser(expr)
##        print expr, "=", result
##        #except Exception, e:
##        #        print expr, ":", e
##        print

