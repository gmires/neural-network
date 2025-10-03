CPP = gcc -O2 -Wall

.PHONY: clean

all: chip8 run

run:
	@./nnet

chip8: main.c
	@gcc main.c -o nnet -lm -O2 -W -Wall

clean:
	rm -f nnet
