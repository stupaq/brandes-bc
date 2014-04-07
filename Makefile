# @author Mateusz Machalica

CXXinclude	+= -I /usr/local/cuda-5.5/include/ -I /opt/cuda/include/
CXXoptimize	+= -march=native -O3 -funroll-loops -flto -fwhole-program -fuse-linker-plugin
CXXwarnings	+= -Wall -Wextra -pedantic

CXX		:= g++ -std=c++0x
CXXFLAGS	+= $(CXXinclude) $(CXXwarnings) $(CXXoptimize) $(CXXarchdep)
LDFLAGS		+= -L /usr/lib64/nvidia
LDLIBS		+= -lOpenCL -lstdc++ -lboost_filesystem -lboost_iostreams

HEADERS		:= $(wildcard *.h)
SOURCES		:= $(wildcard *.cpp)
TARGET		:= brandes

$(TARGET): $(SOURCES) $(HEADERS) Makefile
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) $< $(LDFLAGS) $(LDLIBS) -o $@
	@du -sh $@

clean:
	-rm -rf brandes

# vim: set ts=8 sts=8:
