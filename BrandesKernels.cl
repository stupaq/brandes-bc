/** @author Mateusz Machalica */

// TODO(stupaq) remove this
__kernel void square(
    __global float* input,
    __global float* output,
    const unsigned int count) {
  int i = get_global_id(0);
  if(i < count)
    output[i] = input[i] * input[i];
}

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
    const int source,
    __global struct DistSigma* ds) {
  int my_i = get_global_id(0);
  ds[my_i].dist_ = -1 * (source == my_i);
  ds[my_i].sigma_ = 0;
}

__kernel void vcsr_forward(
    const int current_wave,
    __global bool* proceed,
    __global struct Virtual* vlst,
    __global int* adj,
    __global struct DistSigma* ds) {
  int my_vi = get_global_id(0);
  struct Virtual my_pm = vlst[my_vi];
  struct DistSigma my_ds = ds[my_pm.map_];
  if (my_ds.dist_ == current_wave) {
    int k = my_pm.ptr_;
    const int k_end = vlst[my_vi + 1].ptr_;
    for (; k != k_end; k++) {
      int other_i = adj[k];
      int other_d = ds[other_i].dist_;
      if (other_d == -1) {
        ds[other_i].dist_ = other_d = current_wave + 1;
        *proceed = true;
      }
      if (other_d == current_wave + 1) {
        atomic_add(&(ds[other_i].sigma_), ds[my_pm.map_].sigma_);
      }
    }
  }
}

