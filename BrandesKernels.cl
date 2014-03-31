/** @author Mateusz Machalica */

__kernel void square(__global float* input, __global float* output, const int count) {
  int i = get_global_id(0);
  if(i < count)
    output[i] = input[i] * input[i];
}

