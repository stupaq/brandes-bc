# @author Mateusz Machalica

CXXinclude	+= -I /usr/local/cuda-5.5/include/ -I /opt/cuda/include/ -I /opt/AMDAPP/include/
CXXoptimize	+= -march=native -O3 -funroll-loops -flto -fwhole-program -fuse-linker-plugin -finline-limit=16777216
CXXwarnings	+= -Wall -Wextra -pedantic

ifneq (,$(wildcard /usr/bin/g++-4.8))
CXX		:= g++-4.8 -std=c++11
else
CXX		:= g++ -std=c++11
endif
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
