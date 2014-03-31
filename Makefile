# @author Mateusz Machalica

CPPFLAGS	:= -I /usr/local/cuda-5.5/include/ -I /opt/cuda/include/ -std=c++11
CPPOPT		:= -O2
CPPWARN		:= -Wall -Wextra
LDFLAGS		:= -L /usr/lib64/nvidia -l OpenCL

brandes: Main.cpp
	g++ $(CPPFLAGS) $(CPPOPT) $(CPPWARN) $(LDFLAGS) $< -o $@

clean:
	-rm -rf brandes

