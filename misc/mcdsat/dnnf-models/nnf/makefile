SRCS	= nnf.cc satlib.cc
OBJS	= nnf.o satlib.o
ALLOBJS	= $(OBJS)
GMP     = ../../gmp-4.2

INCLUDE = -I$(GMP)
CFLAGS	= -O3 -Wall -m32 #-DMP
CC	= gcc
C++     = g++

libnnf.a:	$(OBJS)
		ar rv libnnf.a $(OBJS)

.cc.o:
		$(C++) $(CFLAGS) -c $< $(INCLUDE)

.c.o:
		$(CC) $(CFLAGS) -c $< $(INCLUDE)

clean:
		rm -f libnnf.a $(ALLOBJS) *~ core

# DO NOT DELETE

nnf.o: nnf.h hashing.h utils.h
satlib.o: nnf.h hashing.h satlib.h
