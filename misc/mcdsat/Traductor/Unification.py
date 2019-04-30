# from sets import set

# Algoritmo de unificacion de variables del Lloyd (modificado)
# Se unifican funciones de Vars en Vars. 
# Si fun_i(x) != fun_2(x) entonces se escoge cualquiera de los dos como representante R
# pero se unifican todas las variables fun_i(x) = fun_i(y) = R
def unificar(listFun, listVars):
  mgu = {}
  sust = {}
  for x1 in listVars:
    x = 'X'+x1
    d = disagreementSet(listFun,sust,x) # calcula es conjunto desacuerdo de la variable x
    v = d.pop()                         # escoge a un representante v
    sust_i = sustituciones(d, v)        # calcula las sustituciones del conjunto desacuerdo y el representante
    sustituir(mgu, sust_i)              # realiza estas sustituciones en el mgu
    sust = componer(sust,sust_i)        # compone las susts calculadas hasta el momento con la actual
    mgu[x] = v                          # agrega el map x -> v al mgu
  return mgu

# Hace la composicion de dos sustituciones, de acuerdo a la defincion del Lloyd (pag 18)
def componer(sust1, sust2):
  sustt = {}
  if not sust2:
    return sust1
  
  for (u,s) in list(sust1.items()):
    # if sust2.has_key(s) and s != u:
    if s in sust2 and s!= u:
      sustt[u] = sust2[s]
  sust2.update(sustt)
  return sust2

# Realiza las sustituciones sobre el mgu que se esta calculando
def sustituir(mgu, sust):
  for (x,y) in list(mgu.items()):
    # if sust.has_key(y):
    if y in sust:
      mgu[x] = sust[y]

# Encuentra el conjunto de sustituciones que deben realizarse a partir del conjunto en desacuerdo d
# y el valor que se escogio como representante v
def sustituciones(d, v):
  sust = {}
  for x in d:
    sust[x] = v
  return sust

# Encuentra el conjunto en desacuerdo: el conjuntos de todos los valores de las funciones 
# para los cuales se cumple fun_i(x) != fun_j(x).
# Antes realiza las sustituciones que se hayan hecho previamente
def disagreementSet(listFun, sust, var):
  d = set()
  for f in listFun:
    # if f.has_key(var):
    if var in f:
      vi = f[var]
      # if sust.has_key(vi):
      if vi in sust:
        vi = sust[vi]
      d.add(vi)
  return d
  
def prueba():
  ec1 = {'XX':'A', 'XY':'B', 'XZ':'A'}
  ec2 = {'XX':'A', 'XY':'B', 'XZ':'C'}
  ec3 = {'XX':'A', 'XY':'B', 'XZ':'C'}
  l = [ec2, ec3, ec1]
  mgu = unificar(l, ['Y', 'X', 'Z'])
  print(l)
  print(mgu)


if __name__ == "__main__":
  prueba()
