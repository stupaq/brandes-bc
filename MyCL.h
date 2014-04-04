/** @author Mateusz Machalica */
#ifndef MYCL_H_
#define MYCL_H_

#include <iostream>

#include <boost/iostreams/device/mapped_file.hpp>

#ifdef MYCL_ERROR_CHECKING
#define __CL_ENABLE_EXCEPTIONS
#endif
#include <CL/cl.hpp>

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
#ifdef MYCL_ERROR_CHECKING
  try {
    program.build(devices);
  } catch (...) {
    std::cerr << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(devices[0]) << '\n';
    throw;
  }
#else
  program.build(devices);
#endif
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
#ifdef MYCL_ERROR_CHECKING
  if (pi == platforms.end()) {
    throw cl::Error(CL_DEVICE_NOT_FOUND, "NVIDIA platform not found");
  }
#endif
  cl_context_properties cps[] = { CL_CONTEXT_PLATFORM,
    (cl_context_properties)(*pi)(), 0 };
  return cl::Context(CL_DEVICE_TYPE_GPU, cps);
}

#endif  // MYCL_H_
