/** @author Mateusz Machalica */

#define __CL_ENABLE_EXCEPTIONS
#include <CL/cl.hpp>

#include <cassert>
#include <utility>
#include <string>
#include <stdexcept>

#include "./MicroBench.h"
#include "./Brandes.h"

#define DATA_SIZE (1024 * 1240)

inline cl::Program program_from_file(const cl::Context& context,
    const VECTOR_CLASS<cl::Device>& devices, const char* file_path) {
  ssize_t sz;
  FILE* fp = NULL;
  char *code = NULL;

  try {
    MICROBENCH_START(read_and_compile_program);
    fp = fopen(file_path, "rb");
    if (!fp)
      throw std::runtime_error("fopen() failed");

    fseek(fp , 0L , SEEK_END);
    sz = ftell(fp);
    rewind(fp);

    code = new char[sz + 1];
    if (!code) {
      throw std::runtime_error("malloc() failed");
    }

    if (1 != fread(code, sz, 1, fp)) {
      throw std::runtime_error("fread() failed");
    }

    cl::Program::Sources source(1, std::make_pair(code, sz + 1));
    cl::Program program = cl::Program(context, source);
    program.build(devices);

    fclose(fp);
    delete[] code;
    MICROBENCH_END(read_and_compile_program);
    return program;
  } catch (...) {
    fclose(fp);
    delete[] code;
    throw;
  }
}

inline cl::Context initialize_nvidia() {
  VECTOR_CLASS<cl::Platform> platforms;
  cl::Platform::get(&platforms);

  auto pi = platforms.begin();
  for (; pi != platforms.end(); pi++) {
    if (pi->getInfo<CL_PLATFORM_VENDOR>().compare("NVIDIA Corporation") == 0)
      break;
  }
  if (pi == platforms.end()) {
    throw cl::Error(CL_DEVICE_NOT_FOUND, "NVIDIA platform not found");
  }
  cl_context_properties cps[] = { CL_CONTEXT_PLATFORM,
    (cl_context_properties)(*pi)(), 0 };

  return cl::Context(CL_DEVICE_TYPE_GPU, cps);
}

int main(int argc, const char* argv[]) {
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
