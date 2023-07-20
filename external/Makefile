bindir := ./bin

all: build

build:
	mkdir -p $(bindir) && \
		gcc -Wall -Wextra -pedantic -Werror -O2 -march=native -std=c99 -o ${bindir}/xoshiro256plusplus xoshiro256plusplus.c && \
		gcc -Wall -Wextra -pedantic -Werror -O2 -march=native -std=c99 -o ${bindir}/xoshiro256plus xoshiro256plus.c
