all: train test benchmark human
CC=g++
WARNINGS=-Wall -Wextra -Wconversion
CFLAGS=$(WARNINGS) -O2 -march=native -std=c++23 $(shell pkg-config --cflags sfml-graphics torch box2d)
LDFLAGS=$(shell pkg-config --libs sfml-graphics torch box2d)

# DEBUG: macOS flags
CFLAGS=$(WARNINGS) -O2 -march=native -std=c++23 $(shell pkg-config --cflags sfml-graphics) \
	-I/opt/homebrew/include -I/opt/homebrew/opt/box2d/include\
	-I/opt/homebrew/include/torch/csrc/api/include
LDFLAGS=$(shell pkg-config --libs sfml-graphics) \
	-L/opt/homebrew/opt/box2d/lib -lbox2d -L/opt/homebrew/lib/ \
	-lc10 -ltorch -ltorch_cpu

builddir/%.o: src/%.cpp inc/%.hpp
	mkdir -p builddir
	$(CC) -c $(CFLAGS) $< -Iinc -o $@

builddir/%.exe.o: src/%.cpp
	mkdir -p builddir
	$(CC) -c $(CFLAGS) $< -Iinc -o $@

train: builddir/train.exe.o builddir/Cartpole.o builddir/CartpoleRenderer.o 
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ $^

test: builddir/test.exe.o builddir/CartpoleRenderer.o builddir/Cartpole.o
	$(CC) $(CFLAGS) $(LDFLAGS)  -o $@ $^

benchmark: builddir/benchmark.exe.o builddir/Cartpole.o
	$(CC) $(CFLAGS) $(LDFLAGS)  -o $@ $^

human: builddir/human.exe.o builddir/CartpoleRenderer.o builddir/Cartpole.o
	$(CC) $(CFLAGS) $(LDFLAGS)  -o $@ $^

clean: 
	rm -R builddir/
