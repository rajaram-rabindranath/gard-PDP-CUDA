CC=gcc -std=c99 -g
CFLAGS=-I include/
OBJS=prime
SRC=src/
OBJS= lib.o
all:	primeSEQ

primeSEQ:	$(SRC)primGen.c lib.o
	$(CC) -o primeSEQ $(CFLAGS) $(OBJS) $(SRC)primGen.c -lm
	rm -rf $(OBJS)	
lib.o:	$(SRC)lib.c
	$(CC) $(CFLAGS) -c $(SRC)lib.c
clean:
	rm -rf primeSEQ
	rm -rf $(OBJS)
