# @author Mateusz Machalica

CXX		?= g++ -fmax-errors=5
CXX_include	+= -I /usr/local/cuda-5.5/include/ -I /opt/cuda/include/
CXX_optimize	+= -march=native -O3 -fwhole-program -flto -fuse-linker-plugin -funroll-loops
CXXFLAGS	+= $(CXX_include) -Wall -Wextra -pedantic -std=c++0x $(CXX_optimize)
LD		?= $(CXX)
LDFLAGS		+= -L /usr/lib64/nvidia
LDLIBS		+= -lOpenCL -lstdc++ -lboost_filesystem -lboost_iostreams

HEADERS		:= $(wildcard *.h)

brandes: Main
	ln -sf $< $@
	@du -sh $<

Main: Main.o
Main.o: $(HEADERS) Makefile

clean:
	-rm -rf *.o
	-rm -rf brandes Main

# vim: set ts=8 sts=8:
