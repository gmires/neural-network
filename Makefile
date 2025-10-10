CC=gcc
CFLAGS=-O2 -Wall -Wextra -lm -I.
LDFLAGS=-lm

DEPS = nnet.h
OBJM = nnet.o funct.o main.o
OBJO = nnet.o funct.o or.o
OBJX = nnet.o funct.o xor.o

.PHONY: clean

all: nnet nnet-or nnet-xor

run:
	@./nnet

%.o: %.c $(DEPS)
	@$(CC) -c $<  $(CFLAGS) -o $@
#	$(CPP) -c $@ $< 

nnet: $(OBJM)
	@$(CC) -o $@ $^ $(LDFLAGS)
	
nnet-or: $(OBJM)
	@$(CC) -o $@ $^ $(LDFLAGS)

nnet-xor: $(OBJX)
	@$(CC) -o $@ $^ $(LDFLAGS)

#nnet: main.c
#	@gcc nnet.c main.c -o nnet -lm -O2 -W -Wall

clean:
	rm -f *.o nnet nnet-or nnet-xor
