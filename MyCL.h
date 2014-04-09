/** @author Mateusz Machalica */
#ifndef MYCL_H_
#define MYCL_H_

#include <boost/iostreams/device/mapped_file.hpp>
#include <vector>
#ifdef MYCL_ERROR_CHECKING
#define __CL_ENABLE_EXCEPTIONS
#endif
#include <CL/cl.hpp>

#include "./MicroBench.h"

#define MYCL_OPTIONS "-Werror     "\
  "-cl-single-precision-constant  "\
  "-cl-finite-math-only           "\
  "-cl-no-signed-zeros            "
#define MYCL_STREAM stdout
#define MYCL_BUFFER_FOREACH(q, buf_cl, n, Elem, el)\
  for (Elem el : mycl_debug::read<Elem>(q, buf_cl, n))

namespace mycl {
  struct Accelerator {
    cl::Context context_;
    cl::CommandQueue queue_;
    cl::Program program_;
  };

  inline cl::Program build_program(const cl::Context& context,
      const VECTOR_CLASS<cl::Device>& devices, const char* file_path) {
    using boost::iostreams::mapped_file;
    MICROPROF_START(build_program);
    mapped_file mf(file_path, mapped_file::readonly);
    assert(mf.const_data()[mf.size() - 1]);
    cl::Program::Sources source(1, std::make_pair(mf.const_data(), mf.size()));
    cl::Program program = cl::Program(context, source);
#ifdef MYCL_ERROR_CHECKING
    try {
      program.build(devices, MYCL_OPTIONS);
    } catch (...) {
      fprintf(MYCL_STREAM, "OpenCL program build log:\n%s\n",
          program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(devices[0]).c_str());
      throw;
    }
#else
    program.build(devices, MYCL_OPTIONS);
#endif
    MICROPROF_END(build_program);
    return program;
  }

  inline cl::Context nvidia_context() {
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

  Accelerator init_device() {
    Accelerator acc;
    MICROPROF_START(init_device);
    acc.context_ = nvidia_context();
    VECTOR_CLASS<cl::Device> devices =
      acc.context_.getInfo<CL_CONTEXT_DEVICES>();
    acc.queue_ = cl::CommandQueue(acc.context_, devices[0]);
    acc.program_ = build_program(acc.context_, devices,
        "BrandesKernels.cl");
    MICROPROF_END(init_device);
    return acc;
  }

  template<typename Vector> inline size_t bytes(Vector& lst) {
    return lst.size() * sizeof(typename Vector::value_type);
  }
}  // namespace mycl

namespace mycl_debug {
  using mycl::bytes;

  template<typename Elem>
    std::vector<Elem> read(cl::CommandQueue q, cl::Buffer buf_cl, size_t n) {
      std::vector<Elem> buf(n);
      q.enqueueReadBuffer(buf_cl, true, 0, bytes(buf), buf.data());
      q.finish();
      return buf;
    }

}  // namespace mycl_debug

#endif  // MYCL_H_
