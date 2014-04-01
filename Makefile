# @author Mateusz Machalica

override CPPFLAGS	+= -Wall -Wextra -I /usr/local/cuda-5.5/include/ -I /opt/cuda/include/ -std=c++0x -O3
override LDFLAGS	+= -L /usr/lib64/nvidia
override LDLIBS		+= -lOpenCL -lstdc++

INCLUDES	= $(wildcard *.h)
SOURCES		= $(wildcard *.cpp)

brandes: $(SOURCES) $(INCLUDES) Makefile
	g++ $(CPPFLAGS) $(LDFLAGS) $< $(LDLIBS) -o $@

clean:
	-rm -rf brandes

