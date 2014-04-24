# @author Mateusz Machalica

CXXinclude	+= -isystem /usr/local/cuda-5.5/include/ -isystem /opt/cuda/include/ -isystem /opt/AMDAPP/include/
CXXoptimize	+= -march=native -O3 -funroll-loops -flto -fwhole-program -fuse-linker-plugin -finline-limit=16777216
CXXwarnings	+= -Wall -Wextra
CXXlint		?= cpplint --filter=-legal/copyright,-whitespace/braces,-whitespace/newline,-whitespace/parens,-runtime/references

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

#CPPFLAGS	+= -DOPTIMIZE=0
#CPPFLAGS	+= -DDEFAULT_MDEG=8192
#CPPFLAGS	+= -DDEFAULT_WGROUP=32
#CPPFLAGS	+= -DDEFAULT_CPU_JOBS=0
#CPPFLAGS	+= -DDEFAULT_USE_GPU=false
#CPPFLAGS	+= -DNO_DEG1
#CPPFLAGS	+= -DNO_BFS
#CPPFLAGS	+= -DNO_STATS
#CPPFLAGS	+= -DMYCL_QUEUE_PROFILING

$(TARGET): $(SOURCES) $(HEADERS) Makefile
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) $< $(LDFLAGS) $(LDLIBS) -o $@
	@wc -c $@

clean:
	-rm -rf brandes

lint:
	@$(CXXlint) $(SOURCES) $(HEADERS)

todo:
	@grep -nrIe "\(TODO\|FIXME\)" --exclude-dir=.git --exclude=Makefile

# vim: set ts=8 sts=8:
