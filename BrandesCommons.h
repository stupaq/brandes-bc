/** @author Mateusz Machalica */
#ifndef BRANDESCOMMONS_H_
#define BRANDESCOMMONS_H_

#include <cassert>
#include <vector>
#include <future>

#include "./MicroBench.h"
#include "./MyCL.h"

#define CONT_BIND(...) Cont().template cont<Return>(__VA_ARGS__)

namespace brandes {
  using mycl::Accelerator;

  struct Context {
    std::future<Accelerator> dev_future_;
    const int kMDegLog2_;
    const int kWGroupLog2_;
  };

  typedef cl_int VertexId;
  typedef std::vector<VertexId> VertexList;

  struct Identity {
    inline int operator[](int x) const {
      return x;
    }
  };

  template<int kVal> struct Constant {
    inline int operator[](int x) const {
      SUPPRESS_UNUSED(x);
      return kVal;
    }
  };

}  // namespace brandes

#endif  // BRANDESCOMMONS_H_
