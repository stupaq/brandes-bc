/** @author Mateusz Machalica */
#ifndef MYCL_H_
#define MYCL_H_

#define __CL_ENABLE_EXCEPTIONS
#include <CL/cl.hpp>

#include "./MicroBench.h"

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

#endif  // MYCL_H_
