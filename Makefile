# @author Mateusz Machalica

CXX			?= g++ -fmax-errors=5
CXXINCLUDE	+= -I /usr/local/cuda-5.5/include/ -I /opt/cuda/include/
CXXFLAGS	+= $(CXXINCLUDE) -Wall -Wextra -Wpedantic -std=c++0x -O3 -march=native
LD			?= $(CXX)
LDFLAGS		+= -L /usr/lib64/nvidia
LDLIBS		+= -lOpenCL -lstdc++ -lboost_filesystem -lboost_iostreams

HEADERS		:= $(wildcard *.h)

brandes: Main
	ln -sf Main brandes

Main: Main.o
Main.o: $(HEADERS) Makefile

clean:
	-rm -rf *.o
	-rm -rf brandes Main

