C++     = g++
CC	= $(C++)
NNF	= ./nnf
CFLAGS	= -Wall -O3 -I$(NNF) -m32 #-DMP
LDFLAGS = -L$(NNF) -lnnf #-lgmp

all: models mincost

mincost:        mincost.o $(NNF)/libnnf.a
		$(C++) -o mincost mincost.o $(CFLAGS) $(LDFLAGS)

mincost.o:	mincost.cc $(NNF)/nnf.h $(NNF)/satlib.h
		$(C++) $(CFLAGS) -c mincost.cc

models:		models.o $(NNF)/libnnf.a
		$(C++) -o models models.o $(CFLAGS) $(LDFLAGS)

models.o:	models.cc $(NNF)/nnf.h $(NNF)/satlib.h
		$(C++) $(CFLAGS) -c models.cc

clean:
		rm -fr *.o *~ core models

