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
struct Virtual {
  int ptr_;
  int map_;
} __attribute__((packed));

__kernel void vcsr_init(
    const int global_id_range,
    __global float* bc) {
  int my_i = get_global_id(0);
  if (my_i < global_id_range) {
    bc[my_i] = 0.0f;
  }
}

__kernel void vcsr_init_source(
    const int global_id_range,
    const int source,
    __global int* dist,
    __global int* sigma) {
  int my_i = get_global_id(0);
  if (my_i < global_id_range) {
    dist[my_i] = select(-1, 0, source == my_i);
    sigma[my_i] = select(0, 1, source == my_i);
  }
}

__kernel void vcsr_forward(
    const int global_id_range,
    const int curr_dist,
    __global bool* proceed,
    __global struct Virtual* vlst,
    __global int* adj,
    __global int* dist,
    __global int* sigma,
    __global float* delta) {
  int my_vi = get_global_id(0);
  if (my_vi < global_id_range) {
    struct Virtual my_pm = vlst[my_vi];
    struct Virtual next_pm = vlst[my_vi + 1];
    int my_dist = dist[my_pm.map_];
    int my_sigma = sigma[my_pm.map_];
    if (my_dist == curr_dist) {
      int k = my_pm.ptr_;
      const int k_end = next_pm.ptr_;
      for (; k != k_end; k++) {
        int other_i = adj[k];
        int other_d = dist[other_i];
        if (other_d == -1) {
          dist[other_i] = other_d = curr_dist + 1;
          *proceed = true;
        }
        if (other_d == curr_dist + 1 && my_sigma != 0.0f) {
          atomic_add(&sigma[other_i], my_sigma);
        }
      }
      if (my_pm.map_ != next_pm.map_) {
        delta[my_pm.map_] = 1.0f / my_sigma;
      }
    }
  }
}

__kernel void vcsr_backward(
    const int global_id_range,
    const int curr_dist,
    __global struct Virtual* vlst,
    __global int* adj,
    __global int* dist,
    __global int* sigma,
    __global float* delta) {
  int my_vi = get_global_id(0);
  if (my_vi < global_id_range) {
    struct Virtual my_pm = vlst[my_vi];
    if (dist[my_pm.map_] == curr_dist - 1) {
      int k = my_pm.ptr_;
      const int k_end = vlst[my_vi + 1].ptr_;
      float sum = 0.0f;
      for (; k != k_end; k++) {
        int other_i = adj[k];
        if (dist[other_i] == curr_dist) {
          sum += delta[other_i];
        }
      }
      if (sum != 0.0f) {
        atomic_addf(&delta[my_pm.map_], sum);
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
  int my_i = get_global_id(0);
  if (my_i < global_id_range && my_i != source) {
    if (dist[my_i] != -1) {
      bc[my_i] += delta[my_i] * sigma[my_i] - 1;
    }
  }
}

