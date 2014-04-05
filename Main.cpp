/** @author Mateusz Machalica */

#include <boost/preprocessor/stringize.hpp>

#if   OPTIMIZE == 3
#pragma message "Optimized build - no error checking."
#define NDEBUG
#elif OPTIMIZE == 2
#pragma message "Partially optimized build - rare error checking."
#define NDEBUG
#define MYCL_ERROR_CHECKING
#elif OPTIMIZE == 1
#pragma message "Partially optimized build for microbenchmarks."
#define NDEBUG
#define MICROBENCH_ENABLE
#define MYCL_ERROR_CHECKING
#elif OPTIMIZE == 0
#pragma message "Non-optimized build with extra assertions."
#define MICROBENCH_ENABLE
#define MYCL_ERROR_CHECKING
#endif

#if   ALGORITHM == 0x100
#define ALGORITHM_PIPE csr_create<ocsr_pass<vcsr_pass<betweenness<postprocess>>>>
#elif ALGORITHM == 0x110
#define ALGORITHM_PIPE csr_create<ocsr_create<vcsr_pass<betweenness<postprocess>>>>
#elif ALGORITHM == 0x101
#define ALGORITHM_PIPE csr_create<ocsr_pass<vcsr_create<4, betweenness<32, postprocess>>>>
#elif ALGORITHM == 0x111 || ALGORITHM == 0
#define ALGORITHM_PIPE csr_create<ocsr_create<vcsr_create<4, betweenness<32, postprocess>>>>
#endif

#include <cstdio>
#include <cassert>

#include <utility>
#include <string>
#include <future>

#include "./MicroBench.h"
#include "./MyCL.h"
#include "./Brandes.h"

using namespace brandes;

static Accelerator init_device() {
  Accelerator acc;
  MICROBENCH_START(setup_opencl_device);
  acc.context_ = initialize_nvidia();
  VECTOR_CLASS<cl::Device> devices = acc.context_.getInfo<CL_CONTEXT_DEVICES>();
  acc.queue_ = cl::CommandQueue(acc.context_, devices[0]);
  acc.program_ = program_from_file(acc.context_, devices,
      "BrandesKernels.cl");
  MICROBENCH_END(setup_opencl_device);
  return acc;
}

template<typename Result>
static inline void generic_write(Result& res, const char* file_path) {
  MICROBENCH_START(writing_results);
  FILE* fp = fopen(file_path, "w");
  assert(fp);
  for (auto v : res) {
    fprintf(fp, "%f\n", v);
  }
  fclose(fp);
  MICROBENCH_END(writing_results);
}

int main(int argc, const char* argv[]) {
  MICROBENCH_START(main_total);
  assert(argc == 3); SUPPRESS_UNUSED(argc);

  Context ctx = std::async(std::launch::async, init_device);
#ifdef MYCL_ERROR_CHECKING
  try {
#endif
    auto res = generic_read<ALGORITHM_PIPE, std::vector<float>>(ctx, argv[1]);
    generic_write(res, argv[2]);
#ifdef MYCL_ERROR_CHECKING
  } catch (cl::Error error) {
    fprintf(stderr, "%s (error code: %d)\n", error.what(), error.err());
  }
#endif

  MICROBENCH_END(main_total);
  return 0;
}
