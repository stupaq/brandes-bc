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

struct Intermediate {
  int dist_;
  volatile int sigma_;
  float delta_;
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
    __global struct Intermediate* im) {
  int my_i = get_global_id(0);
  if (my_i < global_id_range) {
    struct Intermediate my_im;
    my_im.dist_ = select(-1, 0, source == my_i);
    my_im.sigma_ = select(0, 1, source == my_i);
    im[my_i] = my_im;
  }
}

__kernel void vcsr_forward(
    const int global_id_range,
    const int curr_dist,
    __global bool* proceed,
    __global struct Virtual* vlst,
    __global int* adj,
    __global struct Intermediate* im) {
  int my_vi = get_global_id(0);
  if (my_vi < global_id_range) {
    struct Virtual my_pm = vlst[my_vi];
    struct Virtual next_pm = vlst[my_vi + 1];
    struct Intermediate my_im = im[my_pm.map_];
    if (my_im.dist_ == curr_dist) {
      int k = my_pm.ptr_;
      const int k_end = next_pm.ptr_;
      for (; k != k_end; k++) {
        int other_i = adj[k];
        int other_d = im[other_i].dist_;
        if (other_d == -1) {
          im[other_i].dist_ = other_d = curr_dist + 1;
          *proceed = true;
        }
        if (other_d == curr_dist + 1 && my_im.sigma_ != 0.0f) {
          atomic_add(&(im[other_i].sigma_), my_im.sigma_);
        }
      }
      if (my_pm.map_ != next_pm.map_) {
        im[my_pm.map_].delta_ = 1.0f / my_im.sigma_;
      }
    }
  }
}

__kernel void vcsr_backward(
    const int global_id_range,
    const int curr_dist,
    __global struct Virtual* vlst,
    __global int* adj,
    __global struct Intermediate* im) {
  int my_vi = get_global_id(0);
  if (my_vi < global_id_range) {
    struct Virtual my_pm = vlst[my_vi];
    if (im[my_pm.map_].dist_ == curr_dist - 1) {
      int k = my_pm.ptr_;
      const int k_end = vlst[my_vi + 1].ptr_;
      float sum = 0.0f;
      for (; k != k_end; k++) {
        int other_i = adj[k];
        struct Intermediate other_im = im[other_i];
        if (other_im.dist_ == curr_dist) {
          sum += other_im.delta_;
        }
      }
      if (sum != 0.0f) {
        atomic_addf(&im[my_pm.map_].delta_, sum);
      }
    }
  }
}

__kernel void vcsr_sum(
    const int global_id_range,
    const int source,
    __global struct Intermediate* im,
    __global float* bc) {
  int my_i = get_global_id(0);
  if (my_i < global_id_range && my_i != source) {
    struct Intermediate my_im = im[my_i];
    if (my_im.dist_ != -1) {
      bc[my_i] += my_im.delta_ * my_im.sigma_ - 1;
    }
  }
}

