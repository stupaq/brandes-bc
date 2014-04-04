/** @author Mateusz Machalica */

__kernel void square(__global float* input, __global float* output, const int count) {
  int i = get_global_id(0);
  if(i < count)
    output[i] = input[i] * input[i];
}

struct Virtual* {
  int vptr_;
  int vmap_;
} __attribute__((packed));

__kernel void virtual_forward(
    __constant int* current_wave,
    __global bool* proceed,
    __global struct Virtual* vlst,
    __global int* adj,
    __global int* distances,
    __global float* sigma) {
  int i = get_global_id(0);
  int ui = vlst[i].vmap_;
  if (distances[ui] == current_wave) {
    int begin = vlst[virt].vptr_;
    int end = vlst[virt + 1].vptr_;
    while (begin != end) {
      int vj = adj[begin];
      int vj_dist = distances[vj];
      if (vj_dist == -1) {
        distance[vj] = vj_dist = current_wave + 1;
        proceed = true;
      }
      if (vj_dist == current_wave + 1) {
        atomic_add(sigma + vi, sigma + ui);
      }
      begin++;
    }
}

