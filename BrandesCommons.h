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

  struct Context {
    std::future<Accelerator> dev_future_;
    const int kMDegLog2_;
    const int kWGroup_;
    const int kCPUJobs_;
    const bool kUseGPU_;

    Context(
        std::future<Accelerator> &&dev,
        int m_deg,
        int wgroup,
        int cpu_jobs,
        bool use_gpu
        ) :
      dev_future_(std::move(dev)),
      kMDegLog2_(std::ceil(std::log2(m_deg))),
      kWGroup_(wgroup),
      kCPUJobs_(cpu_jobs),
      kUseGPU_(use_gpu)
    {
      assert(1 << kMDegLog2_ == m_deg);
      assert(wgroup % MYCL_WGROUP_MULTIPLE == 0);
      assert(cpu_jobs > 0 || use_gpu);
    }
  };

}  // namespace brandes

#endif  // BRANDESCOMMONS_H_
