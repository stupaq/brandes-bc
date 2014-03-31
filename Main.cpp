/** @author Mateusz Machalica */

#include <cassert>
#include <utility>
#include <iostream>
#include <fstream>
#include <string>
#include <stdexcept>

#define __CL_ENABLE_EXCEPTIONS
#include <CL/cl.hpp>

#include "./MicroBench.h"
#include "./Brandes.h"

#define DATA_SIZE (1024 * 1240)

using namespace brandes;

int main(int argc, const char* argv[]) {

  try {
    VECTOR_CLASS<cl::Platform> platforms;
    cl::Platform::get(&platforms);

    auto pi = platforms.begin();
    for (; pi != platforms.end(); pi++) {
      if (pi->getInfo<CL_PLATFORM_VENDOR>().compare("NVIDIA Corporation"))
        break;
    }
    if (pi == platforms.end()) {
      throw std::runtime_error("Platform not found");
    }
    cl_context_properties cps[] = { CL_CONTEXT_PLATFORM, (cl_context_properties)(*pi)(), 0 };
    cl::Context context(CL_DEVICE_TYPE_GPU, cps);

    VECTOR_CLASS<cl::Device> devices = context.getInfo<CL_CONTEXT_DEVICES>();

    cl::CommandQueue queue = cl::CommandQueue(context, devices[0]);

    std::ifstream sourceFile("BrandesKernels.cl");
    std::string sourceCode(
        std::istreambuf_iterator<char>(sourceFile),
        (std::istreambuf_iterator<char>()));
    cl::Program::Sources source(1, std::make_pair(sourceCode.c_str(), sourceCode.length()+1));

    cl::Program program = cl::Program(context, source);

    program.build(devices);

    cl::Kernel kernel(program, "square");

    // TODO short test
    float* data = new float[DATA_SIZE];
    float* results = new float[DATA_SIZE];

    int count = DATA_SIZE;
    for(int i = 0; i < count; i++)
      data[i] = rand() / (float)RAND_MAX;

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
    for(int i = 0; i < count; i++) {
      if(results[i] == data[i] * data[i])
        correct++;
    }

    printf("correct: %d / %d\n", correct, count);

  } catch(cl::Error error) {
    std::cout << error.what() << "(" << error.err() << ")" << std::endl;
  }

  return 0;
}
