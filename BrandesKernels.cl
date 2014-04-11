/** @author Mateusz Machalica */

inline int divide_up(
    int value,
    int factor) {
  return (value + (1 << factor) - 1) >> factor;
}

/** Brandes' kernels. */
__kernel void vcsr_init_n(
    const int global_id_range,
    __global float* bc
    ) {
  const int my_i = get_global_id(0);
  if (my_i < global_id_range) {
    bc[my_i] = 0.0f;
  }
}

__kernel void vcsr_init_n1(
    const int global_id_range,
    const int n1,
    __global int* vmap,
    __global int* voff,
    __global int* rmap
    ) {
  const int my_vi = get_global_id(0);
  /* Note that in this kernel global_id_range == n1 + 1, in case you needed n1,
   * there is a separate argument (for compatibility). */
  if (my_vi < global_id_range) {
    if (voff[my_vi] == 0) {
      rmap[vmap[my_vi]] = my_vi;
    }
  }
}

__kernel void vcsr_init_source(
    const int global_id_range,
    const int source,
    __global int* dist,
    __global int* sigma
    ) {
  const int my_i = get_global_id(0);
  if (my_i < global_id_range) {
    dist[my_i] = select(-1, 0, source == my_i);
    sigma[my_i] = select(0, 1, source == my_i);
  }
}

__kernel void vcsr_forward(
    const int global_id_range,
    const int curr_dist,
    const int kMDegLog2,
    __global bool* proceed,
    __global int* vmap,
    __global int* voff,
    __global int* ptr,
    __global int* adj,
    __global float* weight,
    __global int* dist,
    __global int* sigma,
    __global float* delta
    ) {
  const int my_vi = get_global_id(0);
  if (my_vi < global_id_range) {
    const int my_map = vmap[my_vi];
    const int my_dist = dist[my_map];
    if (my_dist == curr_dist) {
      int my_ptr = ptr[my_map];
      const int next_ptr = ptr[my_map + 1];
      const int my_cnt = divide_up(next_ptr - my_ptr, kMDegLog2);
      const int my_off = voff[my_vi];
      my_ptr += my_off;
      const int my_sigma = sigma[my_map];
      for (; my_ptr < next_ptr; my_ptr += my_cnt) {
        const int other_i = adj[my_ptr];
        int other_d = dist[other_i];
        if (other_d == -1) {
          dist[other_i] = other_d = curr_dist + 1;
          *proceed = true;
        }
        if (other_d == curr_dist + 1 && my_sigma != 0.0f) {
          atomic_add(&sigma[other_i], my_sigma);
        }
      }
      if (my_off == 0) {
        delta[my_map] = weight[my_map] / my_sigma;
      }
    }
  }
}

__kernel void vcsr_backward(
    const int global_id_range,
    const int curr_dist,
    const int kMDegLog2,
    __global int* vmap,
    __global int* voff,
    __global int* ptr,
    __global int* adj,
    __global int* dist,
    __global float* delta,
    __global float* red
    ) {
  const int my_vi = get_global_id(0);
  if (my_vi < global_id_range) {
    const int my_map = vmap[my_vi];
    if (dist[my_map] == curr_dist - 1) {
      float sum = 0.0f;
      int my_ptr = ptr[my_map];
      const int next_ptr = ptr[my_map + 1];
      const int my_cnt = divide_up(next_ptr - my_ptr, kMDegLog2);
      my_ptr += voff[my_vi];
      for (; my_ptr < next_ptr; my_ptr += my_cnt) {
        const int other_i = adj[my_ptr];
        if (dist[other_i] == curr_dist) {
          sum += delta[other_i];
        }
      }
      red[my_vi] = sum;
    }
  }
}

__kernel void vcsr_backward_reduce(
    const int global_id_range,
    const int curr_dist,
    __global int* rmap,
    __global int* dist,
    __global float* delta,
    __global float* red
    ) {
  const int my_i = get_global_id(0);
  if (my_i < global_id_range) {
    if (dist[my_i] == curr_dist - 1) {
      float sum = 0.0f;
      int next_i = rmap[my_i];
      const int last_i = rmap[my_i + 1];
      for (; next_i < last_i; next_i++) {
        sum += red[next_i];
      }
      delta[my_i] += sum;
    }
  }
}

__kernel void vcsr_sum(
    const int global_id_range,
    const int source,
    __global float* weight,
    __global int* dist,
    __global int* sigma,
    __global float* delta,
    __global float* bc
    ) {
  const int my_i = get_global_id(0);
  if (my_i < global_id_range && my_i != source && dist[my_i] != -1) {
    bc[my_i] += (delta[my_i] * sigma[my_i] - 1) * weight[source];
  }
}

