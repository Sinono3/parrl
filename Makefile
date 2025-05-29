all: train test benchmark human
CC=g++
WARNINGS=-Wall -Wextra -Wconversion
CFLAGS=$(WARNINGS) -O2 -march=native -std=c++23 $(shell pkg-config --cflags sfml-all torch box2d)
LDFLAGS=$(shell pkg-config --libs sfml-all torch box2d)

# DEBUG: macOS flags
CFLAGS=$(WARNINGS) -O2 -march=native -std=c++23  $(shell pkg-config --cflags sfml-all) \
	-I/opt/homebrew/include -I/opt/homebrew/opt/box2d/include\
	-I/opt/homebrew/include/torch/csrc/api/include
LDFLAGS=$(shell pkg-config --libs sfml-all) \
	-L/opt/homebrew/opt/box2d/lib -lbox2d -L/opt/homebrew/lib/ \
	-lc10 -ltorch -ltorch_cpu

builddir/%.o: src/%.cpp inc/%.hpp
	mkdir -p builddir
	$(CC) -c $(CFLAGS) $< -Iinc -o $@

builddir/train.o: src/train.cpp
	mkdir -p builddir
	$(CC) -c $(CFLAGS) $< -Iinc -o $@

builddir/test.o: src/test.cpp
	mkdir -p builddir
	$(CC) -c $(CFLAGS) $< -Iinc -o $@

builddir/benchmark.o: src/benchmark.cpp
	mkdir -p builddir
	$(CC) -c $(CFLAGS) $< -Iinc -o $@

builddir/human.o: src/human.cpp
	mkdir -p builddir
	$(CC) -c $(CFLAGS) $< -Iinc -o $@

train: builddir/train.o builddir/util.o
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ $^

test: builddir/test.o builddir/util.o
	$(CC) $(CFLAGS) $(LDFLAGS)  -o $@ $^

benchmark: builddir/benchmark.o builddir/util.o
	$(CC) $(CFLAGS) $(LDFLAGS)  -o $@ $^

human: builddir/human.o builddir/util.o
	$(CC) $(CFLAGS) $(LDFLAGS)  -o $@ $^

clean: 
	rm -R builddir/
