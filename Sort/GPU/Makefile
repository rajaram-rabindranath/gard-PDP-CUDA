CUDACC=nvcc 
CC=gcc -std=c99
CFLAGS=-I include/
OBJS=prime
SRC=src/
OBJS= rnd.o ziggurat.o lib.o


all:	cudaBSORT

cudaBSORT:	$(SRC)cudaBSORT.cu
	$(CUDACC) -o cudaBSORT $(CFLAGS) $(SRC)cudaBSORT.cu -lm
	rm -rf $(OBJS)

rnd.o:	$(SRC)rnd.c
	$(CC) $(CFLAGS) -c $(SRC)rnd.c
	
ziggurat.o: $(SRC)ziggurat.c
	$(CC) $(CFLAGS) -c $(SRC)ziggurat.c

lib.o: $(SRC)lib.c
	$(CC) $(CFLAGS) -c $(SRC)lib.c

clean:
	rm -rf cudaBSORT
	rm -rf $(OBJS)
