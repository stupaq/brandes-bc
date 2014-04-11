/** @author Mateusz Machalica */
#ifndef BRANDESCOMMONS_H_
#define BRANDESCOMMONS_H_

#include <cassert>
#include <vector>
#include <future>

#include "./MicroBench.h"
#include "./MyCL.h"

/* This is not very important since entire continuation gets inlined and
 * compiler proves lack  of aliasing by itself. */
#define __pass__ &__restrict__

#define CONT_BIND(...) Cont().template cont<Return>(__VA_ARGS__)

namespace brandes {
  using mycl::Accelerator;

  typedef cl_int VertexId;
  typedef std::vector<VertexId> VertexList;

  struct Context {
    std::future<Accelerator> dev_future_;
    const int kMDegLog2_;
    const int kWGroupLog2_;
    const int kCPUJobs_;
    const bool kUseGPU_;
  };

}  // namespace brandes

#endif  // BRANDESCOMMONS_H_
