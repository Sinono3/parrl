all: train train_new test benchmark human
CC=g++
WARNINGS=-Wall -Wextra -Wconversion
CFLAGS_COMMON=$(WARNINGS) -O2 -std=c++23 -g

CFLAGS_LINUX=$(shell pkg-config --cflags sfml-graphics torch)
LDFLAGS_LINUX=$(shell pkg-config --libs sfml-graphics torch)

CFLAGS_MAC=$(shell pkg-config --cflags sfml-graphics) \
	-I/opt/homebrew/include -I/opt/homebrew/include/torch/csrc/api/include
LDFLAGS_MAC=$(shell pkg-config --libs sfml-graphics) \
	-L/opt/homebrew/lib/ -lc10 -ltorch -ltorch_cpu

# NOTE: Set depending on platform
# CFLAGS=$(CFLAGS_COMMON) $(CFLAGS_LINUX)
# LDFLAGS=$(LDFLAGS_LINUX)
CFLAGS=$(CFLAGS_COMMON) $(CFLAGS_MAC)
LDFLAGS=$(LDFLAGS_MAC)

builddir/%.o: src/%.cpp inc/%.hpp
	mkdir -p builddir
	$(CC) -c $(CFLAGS) $< -Iinc -o $@

builddir/%.exe.o: src/%.cpp
	mkdir -p builddir
	$(CC) -c $(CFLAGS) $< -Iinc -o $@

train: builddir/train.exe.o builddir/Cartpole.o builddir/CartpoleRenderer.o
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ $^

train_new: builddir/train_new.exe.o builddir/Cartpole.o builddir/CartpoleRenderer.o  builddir/MLP.o
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ $^

test: builddir/test.exe.o builddir/CartpoleRenderer.o builddir/Cartpole.o
	$(CC) $(CFLAGS) $(LDFLAGS)  -o $@ $^

benchmark: builddir/benchmark.exe.o builddir/Cartpole.o
	$(CC) $(CFLAGS) $(LDFLAGS)  -o $@ $^

human: builddir/human.exe.o builddir/CartpoleRenderer.o builddir/Cartpole.o
	$(CC) $(CFLAGS) $(LDFLAGS)  -o $@ $^

MLPverify: builddir/MLPverify.exe.o builddir/MLP.o
	$(CC) $(CFLAGS) $(LDFLAGS)  -o $@ $^

clean: 
	rm -R builddir/
