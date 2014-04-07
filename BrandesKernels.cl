/** @author Mateusz Machalica */

/** Atomic add implementation for floats. */
typedef union {
  unsigned int int_;
  float float_;
} u_int_float;

inline void atomic_addf(
    __global volatile float* source,
    const float operand) {
  u_int_float curr, prev;
  do {
    prev.float_ = *source;
    curr.float_ = prev.float_ + operand;
  } while (prev.int_ != atomic_cmpxchg(
        (volatile __global unsigned int*) source, prev.int_, curr.int_));
}

/** VCSR Brandes' algorithm. */
__kernel void vcsr_init(
    const int global_id_range,
    __global float* bc) {
  const int my_i = get_global_id(0);
  if (my_i < global_id_range) {
    bc[my_i] = 0.0f;
  }
}

__kernel void vcsr_init_source(
    const int global_id_range,
    const int source,
    __global int* dist,
    __global int* sigma) {
  const int my_i = get_global_id(0);
  if (my_i < global_id_range) {
    dist[my_i] = select(-1, 0, source == my_i);
    sigma[my_i] = select(0, 1, source == my_i);
  }
}

__kernel void vcsr_forward(
    const int global_id_range,
    const int curr_dist,
    __global bool* proceed,
    __global int* vmap,
    __global int* vptr,
    __global int* adj,
    __global int* dist,
    __global int* sigma,
    __global float* delta) {
  const int my_vi = get_global_id(0);
  if (my_vi < global_id_range) {
    const int my_map = vmap[my_vi];
    const int next_map = vmap[my_vi + 1];
    int my_ptr = vptr[my_vi];
    const int next_ptr = vptr[my_vi + 1];
    const int my_dist = dist[my_map];
    if (my_dist == curr_dist) {
      const int my_sigma = sigma[my_map];
      for (; my_ptr != next_ptr; my_ptr++) {
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
      if (my_map != next_map) {
        delta[my_map] = 1.0f / my_sigma;
      }
    }
  }
}

__kernel void vcsr_backward(
    const int global_id_range,
    const int curr_dist,
    __global int* vmap,
    __global int* vptr,
    __global int* adj,
    __global int* dist,
    __global float* delta) {
  const int my_vi = get_global_id(0);
  if (my_vi < global_id_range) {
    const int my_map = vmap[my_vi];
    int my_ptr = vptr[my_vi];
    const int next_ptr = vptr[my_vi + 1];
    if (dist[my_map] == curr_dist - 1) {
      float sum = 0.0f;
      for (; my_ptr != next_ptr; my_ptr++) {
        const int other_i = adj[my_ptr];
        if (dist[other_i] == curr_dist) {
          sum += delta[other_i];
        }
      }
      if (sum != 0.0f) {
        atomic_addf(&delta[my_map], sum);
      }
    }
  }
}

__kernel void vcsr_sum(
    const int global_id_range,
    const int source,
    __global int* dist,
    __global int* sigma,
    __global float* delta,
    __global float* bc) {
  const int my_i = get_global_id(0);
  if (my_i < global_id_range && my_i != source && dist[my_i] != -1) {
    bc[my_i] += delta[my_i] * sigma[my_i] - 1;
  }
}

