/** @author Mateusz Machalica */

#ifdef OPTIMIZE
#define NDEBUG
#else
#define MICROBENCH
#endif

#include <cassert>
#include <utility>
#include <string>
#include <stdexcept>
#include <cstdio>

#include "./MicroBench.h"
#include "./MyCL.h"
#include "./Brandes.h"

#define DATA_SIZE (1024 * 1240)

using namespace brandes;

int main(int argc, const char* argv[]) {
  MICROBENCH_START(total);
  assert(argc == 3);

  GraphOCSR csr =
    GraphOCSR::create(
      GraphCSR::create(
        GraphGeneric::read(argv[1])));

  try {
    MICROBENCH_START(setup_opencl_device);
    cl::Context context = initialize_nvidia();
    VECTOR_CLASS<cl::Device> devices = context.getInfo<CL_CONTEXT_DEVICES>();
    cl::CommandQueue queue = cl::CommandQueue(context, devices[0]);
    cl::Program program = program_from_file(context, devices,
        "BrandesKernels.cl");
    MICROBENCH_END(setup_opencl_device);

    // TODO(stupaq) short test
    cl::Kernel kernel(program, "square");

    float* data = new float[DATA_SIZE];
    float* results = new float[DATA_SIZE];

    int count = DATA_SIZE;
    for (int i = 0; i < count; i++)
      data[i] = rand() / static_cast<float>(RAND_MAX);

    cl::Buffer input = cl::Buffer(context, CL_MEM_READ_ONLY, count *
        sizeof(int));
    cl::Buffer output = cl::Buffer(context, CL_MEM_WRITE_ONLY, count *
        sizeof(int));

    queue.enqueueWriteBuffer(input, CL_TRUE, 0, count * sizeof(int), data);

    kernel.setArg(0, input);
    kernel.setArg(1, output);
    kernel.setArg(2, count);

    cl::NDRange global(count);
    cl::NDRange local(1);
    queue.enqueueNDRangeKernel(kernel, cl::NullRange, global, local);

    queue.enqueueReadBuffer(output, CL_TRUE, 0, count * sizeof(int), results);
    queue.finish();

    int correct = 0;
    for (int i = 0; i < count; i++) {
      if (results[i] == data[i] * data[i])
        correct++;
    }

    printf("correct: %d / %d\n", correct, count);
  } catch(cl::Error error) {
    fprintf(stderr, "%s (error code: %d)\n", error.what(), error.err());
  }

  MICROBENCH_END(total);
  return 0;
}
