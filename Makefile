CC=gcc
CFLAGS=-O2 -Wall -Wextra -lm -I.
LDFLAGS=-lm

DEPS = nnet.h
OBJ = nnet.o funct.o main.o

.PHONY: clean

all: nnet run

run:
	@./nnet

%.o: %.c $(DEPS)
	$(CC) -c $<  $(CFLAGS) -o $@
#	$(CPP) -c $@ $< 

nnet: $(OBJ)
	$(CC) -o $@ $^ $(LDFLAGS)
	
#nnet: main.c
#	@gcc nnet.c main.c -o nnet -lm -O2 -W -Wall

clean:
	rm -f *.o nnet
