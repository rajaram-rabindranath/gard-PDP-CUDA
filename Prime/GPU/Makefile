CC=nvcc
CFLAGS=-I include/
SRC=src/

all:	cudaPrime

cudaPrime:	$(SRC)cudaPrime.cu $(OBJS)
	$(CC) -o cudaPrime $(CFLAGS) $(SRC)cudaPrime.cu -lm

clean:
	rm -rf cudaPrime
