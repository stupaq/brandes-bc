# @author Mateusz Machalica

CPPFLAGS	= -Wall -Wextra -I /usr/local/cuda-5.5/include/ -I /opt/cuda/include/ -std=c++0x -O2
LDFLAGS		= -L /usr/lib64/nvidia
LDLIBS		= -lOpenCL -lstdc++

INCLUDES	= $(wildcard *.h)
SOURCES		= $(wildcard *.cpp)

brandes: $(SOURCES) $(INCLUDES) Makefile
	g++ $(CPPFLAGS) $(LDFLAGS) $< $(LDLIBS) -o $@

clean:
	-rm -rf brandes

