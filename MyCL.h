/** @author Mateusz Machalica */
#ifndef MYCL_H_
#define MYCL_H_

#define __CL_ENABLE_EXCEPTIONS
#include <CL/cl.hpp>

#include <boost/iostreams/device/mapped_file.hpp>

#include "./MicroBench.h"

struct Accelerator {
  cl::Context context_;
  cl::CommandQueue queue_;
  cl::Program program_;
};

inline cl::Program program_from_file(const cl::Context& context,
    const VECTOR_CLASS<cl::Device>& devices, const char* file_path) {
  using boost::iostreams::mapped_file;
  MICROBENCH_START(read_and_compile_program);
  mapped_file mf(file_path, mapped_file::readonly);
  cl::Program::Sources source(1, std::make_pair(mf.const_data(), mf.size()));
  cl::Program program = cl::Program(context, source);
  program.build(devices);
  MICROBENCH_END(read_and_compile_program);
  return program;
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
