/** @author Mateusz Machalica */

#if OPTIMIZE == 1
#define NDEBUG
#elif OPTIMIZE == -1
#warning "Non-optimized build with extra assertions."
#define MICROBENCH
#else
#warning "Partially optimized build for microbenchmarks."
#define NDEBUG
#define MICROBENCH
#endif

#include <cstdio>
#include <cassert>

#include <utility>
#include <string>
#include <stdexcept>

#include "./MicroBench.h"
#include "./MyCL.h"
#include "./Brandes.h"

using namespace brandes;

Accelerator init_device() {
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

int main(int argc, const char* argv[]) {
  MICROBENCH_START(main_total);
  assert(argc == 3); SUPPRESS_UNUSED(argc);

  DeviceCtx ctx;
  fprintf(stderr, "--------\n");
  ctx = std::async(std::launch::async | std::launch::deferred, init_device);
  generic_read<csr_create<ocsr_pass<vcsr_pass<Terminal>>>, int>(ctx, argv[1]);
  fprintf(stderr, "--------\n");
  ctx = std::async(std::launch::async | std::launch::deferred, init_device);
  generic_read<csr_create<ocsr_pass<vcsr_create<4, Terminal>>>, int>(ctx, argv[1]);
  fprintf(stderr, "--------\n");
  ctx = std::async(std::launch::async | std::launch::deferred, init_device);
  generic_read<csr_create<ocsr_create<vcsr_pass<Terminal>>>, int>(ctx, argv[1]);
  fprintf(stderr, "--------\n");
  ctx = std::async(std::launch::async | std::launch::deferred, init_device);
  generic_read<csr_create<ocsr_create<vcsr_create<4, Terminal>>>, int>(ctx, argv[1]);
  fprintf(stderr, "--------\n");

  MICROBENCH_END(main_total);
  return 0;
}
