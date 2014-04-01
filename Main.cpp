/** @author Mateusz Machalica */

#define MICROBENCH

#include <cassert>
#include <utility>
#include <string>
#include <stdexcept>

#include "./MicroBench.h"
#include "./MyCL.h"
#include "./Brandes.h"

#define DATA_SIZE (1024 * 1240)

using namespace brandes;

int main(int argc, const char* argv[]) {
  assert(argc == 3);

  GraphCSR csr = GraphCSR::create(GraphGeneric::read(argv[1]));

  // TODO(stupaq)
  return 0;

  try {
    MICROBENCH_START(setup_opencl_device);
    cl::Context context = initialize_nvidia();
    VECTOR_CLASS<cl::Device> devices = context.getInfo<CL_CONTEXT_DEVICES>();
    cl::CommandQueue queue = cl::CommandQueue(context, devices[0]);
    cl::Program program = program_from_file(context, devices,
        "BrandesKernels.cl");

    cl::Kernel kernel(program, "square");
    MICROBENCH_END(setup_opencl_device);

    // TODO(stupaq) short test
    float* data = new float[DATA_SIZE];
    float* results = new float[DATA_SIZE];

    int count = DATA_SIZE;
    for (int i = 0; i < count; i++)
      data[i] = rand() / (float) RAND_MAX;

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
    std::cout << error.what() << "(" << error.err() << ")" << std::endl;
  } catch(std::runtime_error error) {
    std::cout << error.what() << std::endl;
  }

  return 0;
}
