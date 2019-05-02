# -*- coding: cp949 -*-

import sys
import os


'''

http://www.scipy.org/Cookbook/Matplotlib/Show_colormaps

draw a matrix plot


file format
Name	mub	pnr	roq	decay	msi	Atx2	Tao-1	TER94	how	Dek	CG14438	Sce	Ras85D	cic	mus201	Fmr1	th	Pkn	Pc	Ice	Taf4	pAbp		RpL19	Rac1	Edc3	Akt1	mrn	tws	eff
AD	0.224					0.720273736	0.572284							0.346		0.572284	0.572284													-0.122
HD	0.224					0.339758	0.177				0.572284				-0.122		0.343168				0.177	-0.229116		-0.122			0.598273736			0.346
SCA1	0.720273736					0.572284	-0.122				0.572284					-0.122	0.598273736										0.720273736			0.572284
SCA3	0.346					0.346		0.346						-0.122			0.572284													
SCA7	0.346					0.346											0.491157736													-0.122

if blank --> assigned to 0.0



'''


u_max = None
u_min = None

import random
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import matplotlib.colors as colors


def __load(fname):

    global u_max, u_min
    
    f=open(fname,'r')

    mcol = []
    mrow = []
    data = []
    
    init = True
    for s in f.readlines():
        s = s.replace('\n','')
        if init:
            init=False
            x = s.split('\t')
            del x[0]
            mcol = x
        else:
            if len(s)>0:
                x = s.split('\t')
                rname = x[0]


                
                
                del x[0]
                r=[]
                for v in x:
                    v = v.strip()
                    
                    if v == '':
                        r.append(np.nan)
                    else:
                        
                        vx = eval(v)
                        r.append( vx )

                        if u_max == None or u_min == None:
                            u_max = u_min = vx
                        else:
                            if vx > u_max:
                                u_max = vx
                            elif vx < u_min:
                                u_min = vx


                mrow.append(rname)
                data.append(r)

    f.close()


    return mcol, mrow, data

def __drawBar(user_cmap, min_value, max_value, fname, dpi, fontsize):

    global u_max, u_min



    fontsize = int(fontsize * 1.5)

    mcol = [ '-' ]
    mrow = []
    
    data = []

    if min_value == None or max_value == None:
        min_value = u_min
        max_value = u_max
        
    v=max_value
    step = abs(float(max_value - min_value))/10.0
    


    
    while(v>=min_value):
        data.append([v])
        mrow.append(str(v))
        v=v-step

    if fname == None:
        __drawMatrix(mcol, mrow, data, min_value, max_value, user_cmap, fname, 0, dpi, fontsize)
    else:

        
        __drawMatrix(mcol, mrow, data, min_value, max_value, user_cmap, fname+'_bar.png', 0, dpi, fontsize)


def __draw(mcol, mrow, data, min_value, max_value, user_cmap, fname, y_rotation, dpi, fontsize):


    __drawMatrix(mcol,mrow,data,min_value, max_value, user_cmap, fname, y_rotation, dpi, fontsize)
    __drawBar(user_cmap, min_value, max_value, fname, dpi, fontsize)
    

def __drawMatrix(mcol, mrow, data, min_value, max_value, user_cmap, fname, y_rotation, dpi, fontsize):

    global u_max, u_min

    '''
    http://www.scipy.org/Cookbook/Matplotlib/Show_colormaps

    cmap
        plt.cm.gray: black-white
        plt.cm.gray_r
        plt.cm.jet
        plt.cm.jet_r

        cm.cool
        cm.cool_r
        
        plt.cm.swirly_cmap
        
        
        
    '''
    
    

    nrows, ncols = len(mrow), len(mcol)
    image = np.zeros(nrows*ncols)

    i=0
    for d in data:
        for v in d:
            image[i]=v
            i=i+1

    image = image.reshape((nrows, ncols))

    row_labels = mrow
    col_labels = mcol

    

    if min_value == None or max_value == None:
        min_value = u_min
        max_value = u_max

    plt.matshow(image, norm=colors.Normalize(vmin=min_value, vmax=max_value), cmap=user_cmap)
    

    
    
    
    if y_rotation==None:
        plt.xticks(range(ncols), col_labels, fontsize = fontsize)
        plt.yticks(fontsize=fontsize)
    else:
        plt.xticks(range(ncols), col_labels, rotation=y_rotation, fontsize=fontsize)
        plt.yticks(fontsize=fontsize)
        
    plt.yticks(range(nrows), row_labels)




    if fname != None:
        plt.savefig(fname,dpi=dpi, bbox_inches='tight')
    else:
        plt.show()
        


def run(options):


    [fname, ofname, rotation, min_value, max_value, cmap, blank_color, dpi, fontsize] = options


    y_rotation = 0
    if rotation:
        y_rotation = 90
        


    mcol, mrow, data = __load(fname)
    __draw(mcol, mrow, data, min_value, max_value, cmap, ofname, y_rotation, dpi, fontsize)
    


def getCmaps():

    dct = {}
    
    maps = [m for m in plt.cm.datad if not m.endswith("_r") ]
    for i, m in enumerate(maps):
        cmap = plt.get_cmap(m)
        dct[m] = cmap



    dct = addMyGloomyCmap(dct)
    dct = addMyCartoonCmap(dct)
    dct = addRedWhite(dct)
    dct = addWhiteRed(dct)
    dct = addOrangeWhite(dct)
    dct = addWhiteOrange(dct)
    dct = addOrangeRedWhite(dct)
    

    return dct



def addBlueWhite(dct):
    
    cdict = { 'red': ((0.0, 0.3, 0.3),
                      (0.5, 0.65, 0.65),
                      (1.0, 1.0, 1.0) ),
              'green': ((0.0, 0.3, 0.3),
                        (0.5, 0.65, 0.65),
                        (1.0, 1.0, 1.0) ),
              'blue': ( (0.0, 1.0, 1.0),
                        (0.5, 1.0, 1.0),
                        (1.0, 1.0, 1.0) ) }
    
    my_cmap = matplotlib.colors.LinearSegmentedColormap('OrangeRedWhite', cdict, 256)
    dct['OrangeRedWhite'] = my_cmap
    
    return dct  

def addOrangeRedWhite(dct):
    
    cdict = { 'red': ((0.0, 1.0, 1.0),
                      (0.5, 1.0, 1.0),
                      (1.0, 1.0, 1.0) ),
              'green': ((0.0, 0.3, 0.3),
                        (0.5, 0.65, 0.65),
                        (1.0, 1.0, 1.0) ),
              'blue': ( (0.0, 0.0, 0.0),
                        (0.5, 0.5, 0.5),
                        (1.0, 1.0, 1.0) ) }
    
    my_cmap = matplotlib.colors.LinearSegmentedColormap('OrangeRedWhite', cdict, 256)
    dct['OrangeRedWhite'] = my_cmap
    
    return dct  



def addMyGloomyCmap(dct):


    cdict = { 'red': ((0.0, 0.0, 0.0),
                      (0.5, 0.0, 0.0),
                      (1.0, 1.0, 1.0) ),
              'green': ((0.0, 0.0, 0.0),
                        (0.5, 1.0, 1.0),
                        (1.0, 0.0, 0.0) ),
              'blue': ( (0.0, 1.0, 1.0),
                        (0.5, 0.0, 0.0),
                        (1.0, 0.0, 0.0) ) }

    my_cmap = matplotlib.colors.LinearSegmentedColormap('Gloomy', cdict, 256)
    dct['Gloomy'] = my_cmap
    return dct


def addMyCartoonCmap(dct):


    cdict = { 'red': ((0.0, 0.0, 0.0),
                      (0.5, 1.0, 0.7),
                      (1.0, 1.0, 1.0) ),
              'green': ((0.0, 0.0, 0.0),
                        (0.5, 1.0, 0.0),
                        (1.0, 1.0, 1.0) ),
              'blue': ( (0.0, 0.0, 0.0),
                        (0.5, 1.0, 0.0),
                        (1.0, 0.5, 1.0) ) }

    my_cmap = matplotlib.colors.LinearSegmentedColormap('Cartoon', cdict, 256)
    dct['Cartoon'] = my_cmap
    return dct    

def addRedWhite(dct):
    cdict = { 'red': ((0.0, 1.0, 1.0),
                      (0.5, 1.0, 1.0),
                      (1.0, 1.0, 1.0) ),
              'green': ((0.0, 0.0, 0.0),
                        (0.5, 0.5, 0.5),
                        (1.0, 1.0, 1.0) ),
              'blue': ( (0.0, 0.0, 0.0),
                        (0.5, 0.5, 0.5),
                        (1.0, 1.0, 1.0) ) }

    my_cmap = matplotlib.colors.LinearSegmentedColormap('RedWhite', cdict, 256)
    dct['RedWhite'] = my_cmap
    return dct       


def addWhiteRed(dct):
    cdict = { 'red': ((0.0, 1.0, 1.0),
                      (0.5, 1.0, 1.0),
                      (1.0, 1.0, 1.0) ),
              'green': ((0.0, 1.0, 1.0),
                        (0.5, 0.5, 0.5),
                        (1.0, 0.0, 0.0) ),
              'blue': ( (0.0, 1.0, 1.0),
                        (0.5, 0.5, 0.5),
                        (1.0, 0.0, 0.0) ) }

    my_cmap = matplotlib.colors.LinearSegmentedColormap('WhiteRed', cdict, 256)
    dct['WhiteRed'] = my_cmap
    return dct       

def addOrangeWhite(dct):
    
    cdict = { 'red': ((0.0, 1.0, 1.0),
                      (0.5, 1.0, 1.0),
                      (1.0, 1.0, 1.0) ),
              'green': ((0.0, 0.4, 0.4),
                        (0.5, 0.7, 0.7),
                        (1.0, 1.0, 1.0) ),
              'blue': ( (0.0, 0.3, 0.3),
                        (0.5, 0.6, 0.6),
                        (1.0, 1.0, 1.0) ) }
    
    my_cmap = matplotlib.colors.LinearSegmentedColormap('OrangeWhite', cdict, 256)
    dct['OrangeWhite'] = my_cmap
    
    return dct  


def addWhiteOrange(dct):
    
    cdict = { 'red': ((0.0, 1.0, 1.0),
                      (0.5, 1.0, 1.0),
                      (1.0, 1.0, 1.0) ),
              'green': ((0.0, 1.0, 1.0),
                        (0.5, 0.7, 0.7),
                        (1.0, 0.0, 0.0) ),
              'blue': ( (0.0, 1.0, 1.0),
                        (0.5, 0.6, 0.6),
                        (1.0, 0.3, 0.3) ) }
    
    my_cmap = matplotlib.colors.LinearSegmentedColormap('WhiteOrange', cdict, 256)
    dct['WhiteOrange'] = my_cmap
    
    return dct  

def addMyBinaryCmap(dct):

    # discrete color map with hex color
    # 0 - grey, 1 - pink
    cpool = [ '#BFBFBF', '#FFFFFF', '#FF5050']
    my_cmap = matplotlib.colors.ListedColormap( cpool, 'indexed')
    dct['Binary'] = my_cmap
    return dct  
    

def getOption(argv):

    

    #try:

    fname = argv[1]
    ofname = None
    rotation = False
    min_value = None
    max_value = None
    cmap = plt.cm.RdYlGn
    blank_color = 'w' # white
    dpi = 300
    cmaps = getCmaps()
    fontsize = 30


    if fname == '-m':
        showColormaps()
        sys.exit(1)

    
    for i in range(2, len(argv)):


        
        o = argv[i]

        if o == '-r' or o=='-rotation_y_axis':
            rotation = True
        elif o.find('-max:')>=0:
            x = o.split(':')
            max_value = float( eval(x[1]) )
        elif o.find('-min:')>=0:
            x = o.split(':')
            min_value = float( eval(x[1]) )
        elif o.find('-cmap:')>=0:
            x = o.split(':')
            c = x[1]

            if cmaps.has_key(c):
                cmap = cmaps[c]
            else:
                print 'Unknown Cmap: ', c
                sys.exit(1)

        elif o.find('-blank:')>=0 or o.find('-b:')>=0:
            x = o.split(':')
            blank_color = x[1]
        elif o.find('-dpi:')>=0:
            x = o.split(':')
            dpi = eval(x[1])
            

        elif o.find('-o:')>=0:

            o = o.replace('"','')
            ofname = o [ o.find(':') + 1: ]
            
        elif o.find('-fontsize:')>=0:
            fontsize = eval( o [ o.find(':')+1: ] )
            

    cmap.set_bad(blank_color)
    
    return [fname, ofname, rotation, min_value, max_value, cmap, blank_color, dpi, fontsize]




def showColormaps():
    cmaps = getCmaps()


    keys = cmaps.keys()
    keys.sort()

    a = np.outer(np.arange(0,1,0.01),np.ones(10))

    plt.figure(figsize=(10,5))

    index= 0 
    l = len(cmaps) + 1
    for key in keys:
        index = index + 1
        plt.subplot(1, l, index)
        plt.axis("off")
        plt.imshow(a, aspect='auto', cmap=cmaps[key],origin='lower')
        plt.title(key,rotation=90,fontsize=10)

    plt.savefig("colormaps.png", dpi=100)
    plt.show()
    
    
        
def displayHelp():
    dct = getCmaps()
    s = ','.join(dct.keys()) + '\n\t -> ' + 'http://www.scipy.org/Cookbook/Matplotlib/Show_colormaps\n'
    
    v = '''MatrixPlot tool

python matrix_plot.py [data file] -o:[output file] -rotation_y_axis -max:value -min:value -cmap:name -blank:name

Mendatory:
data file -> matrix file to plot


Optional:
-m -> show color maps (other options will be ignored!)
      http://www.loria.fr/~rougier/teaching/matplotlib/#colormaps

-o:filename -> file to save graph (if not, show graph on screen)
-rotation_y_axis (or -r) -> rotate the fond in y axis
-max:value -> max value of the matrix. When omitted, use the max of the matrix
-min:value -> same as above
-blank:color -> set color for blank (nan) data
    w:white, b:blue,g:green, black:black, gray:gray, red: red
-cmap:name -> color maps:
    XXXXXXXX
-dpi:value -> dpi (300, 600, etc). Applied only with -o option.
-fontsize:number -> change font size(default=30)
    '''


    v = v.replace('XXXXXXXX', s)
    print v
    
if __name__ == '__main__':

    



    
    o = getOption(sys.argv)    
    if o == None:
        displayHelp()

        
    else:
        run(o)
