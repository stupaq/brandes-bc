# @author Mateusz Machalica

CXXinclude	+= -I /usr/local/cuda-5.5/include/ -I /opt/cuda/include/
CXXoptimize	+= -march=native -O3 -funroll-loops -flto -fwhole-program -fuse-linker-plugin

CXX     	?= g++ -fmax-errors=5
CXXFLAGS	+= $(CXXinclude) -Wall -Wextra -pedantic -std=c++0x $(CXXoptimize) $(CXXarchdep)
LD      	?= $(CXX)
LDFLAGS 	+= -L /usr/lib64/nvidia
LDLIBS  	+= -lOpenCL -lstdc++ -lboost_filesystem -lboost_iostreams

HEADERS 	:= $(wildcard *.h)

brandes: Main
	ln -sf $< $@
	@du -sh $<

Main: Main.o
Main.o: $(HEADERS) Makefile

clean:
	-rm -rf *.o
	-rm -rf brandes Main

# vim: set ts=8 sts=8:
