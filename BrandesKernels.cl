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

struct DistSigma {
  int dist_;
  volatile int sigma_;
} __attribute__((packed));

__kernel void vcsr_init(
    const int global_id_range,
    __global float* bc) {
  int my_i = get_global_id(0);
  if (my_i < global_id_range) {
    bc[my_i] = 0;
  }
}

__kernel void vcsr_init_source(
    const int global_id_range,
    const int source,
    __global struct DistSigma* ds) {
  int my_i = get_global_id(0);
  if (my_i < global_id_range) {
    struct DistSigma my_ds;
    my_ds.dist_ = select(-1, 0, source == my_i);
    my_ds.sigma_ = select(0, 1, source == my_i);
    ds[my_i] = my_ds;
  }
}

__kernel void vcsr_forward(
    const int global_id_range,
    const int curr_dist,
    __global bool* proceed,
    __global struct Virtual* vlst,
    __global int* adj,
    __global struct DistSigma* ds) {
  int my_vi = get_global_id(0);
  if (my_vi < global_id_range) {
    struct Virtual my_pm = vlst[my_vi];
    struct DistSigma my_ds = ds[my_pm.map_];
    if (my_ds.dist_ == curr_dist) {
      int k = my_pm.ptr_;
      const int k_end = vlst[my_vi + 1].ptr_;
      for (; k != k_end; k++) {
        int other_i = adj[k];
        int other_d = ds[other_i].dist_;
        if (other_d == -1) {
          ds[other_i].dist_ = other_d = curr_dist + 1;
          *proceed = true;
        }
        if (other_d == curr_dist + 1) {
          atomic_add(&(ds[other_i].sigma_), ds[my_pm.map_].sigma_);
        }
      }
    }
  }
}

__kernel void vcsr_interm(
    const int global_id_range,
    __global struct DistSigma* ds,
    __global float* delta) {
  int my_i = get_global_id(0);
  if (my_i < global_id_range) {
    delta[my_i] = 1.0f / ds[my_i].sigma_;
  }
}

__kernel void vcsr_backward(
    const int global_id_range,
    const int curr_dist,
    __global struct Virtual* vlst,
    __global int* adj,
    __global struct DistSigma* ds,
    __global float* delta) {
  int my_vi = get_global_id(0);
  if (my_vi < global_id_range) {
    struct Virtual my_pm = vlst[my_vi];
    if (ds[my_pm.map_].dist_ == curr_dist - 1) {
      int k = my_pm.ptr_;
      const int k_end = vlst[my_vi + 1].ptr_;
      float sum = 0;
      for (; k != k_end; k++) {
        int other_i = adj[k];
        if (ds[other_i].dist_ == curr_dist) {
          sum += delta[other_i];
        }
      }
      atomic_addf(&delta[my_pm.map_], sum);
    }
  }
}

__kernel void vcsr_sum(
    const int global_id_range,
    const int source,
    __global struct DistSigma* ds,
    __global float* delta,
    __global float* bc) {
  int my_i = get_global_id(0);
  if (my_i < global_id_range && my_i != source) {
    struct DistSigma my_ds = ds[my_i];
    if (my_ds.dist_ != -1) {
      bc[my_i] += delta[my_i] * my_ds.sigma_ - 1;
    }
  }
}

