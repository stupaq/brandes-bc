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

__kernel void virtual_forward(
    __constant int* current_wave,
    __global bool* proceed,
    __global struct Virtual* vlst,
    __global int* adj,
    __global int* distances,
    __global volatile int* sigma) {
  int i = get_global_id(0);
  struct Virtual ui = vlst[i];
  if (distances[ui.map_] == *current_wave) {
    int itadj = adj + ui.ptr_;
    const int itadjN = adj + vlst[i + 1].ptr_;
    while (itadj != itadjN) {
      int vj = *itadj;
      int vj_dist = distances[vj];
      if (vj_dist == -1) {
        distances[vj] = vj_dist = *current_wave + 1;
        *proceed = true;
      }
      if (vj_dist == *current_wave + 1) {
        atomic_add(&sigma[vj], sigma[ui.map_]);
      }
      begin++;
    }
  }
}

