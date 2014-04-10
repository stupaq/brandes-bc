/** @author Mateusz Machalica */
#ifndef BRANDESVCSR_H_
#define BRANDESVCSR_H_

#include <cassert>
#include <vector>

#include "./BrandesDEG1.h"

template<typename Int> inline Int round_up(Int value, Int factor) {
  return value + factor - 1 - (value - 1) % factor;
}

template<typename Int> inline Int divide_up(Int value, Int factor) {
  return (value + factor - 1) / factor;
}

namespace brandes {

  template<typename Cont> struct vcsr_create {
    template<typename Return, typename Weights>
      inline Return cont(Context& ctx, VertexList& ptr, VertexList& adj,
          Weights& weight, VertexList& ccs) const {
        MICROPROF_INFO("CONFIGURATION:\tvirtualized deg\t%d\n", ctx.kMDeg_);
        MICROPROF_START(virtualization);
        const size_t kN1Estimate = ptr.size() + adj.size() / ctx.kMDeg_;
        const VertexId n = ptr.size() - 1;
        VertexList vmap, voff;
        vmap.reserve(kN1Estimate);
        voff.reserve(kN1Estimate);
        for (VertexId ind = 0; ind < n; ind++) {
          VertexId deg = ptr[ind + 1] - ptr[ind],
                   vcnt = divide_up(deg, ctx.kMDeg_);
          if (vcnt == 0) {
            vmap.push_back(ind);
            voff.push_back(0);
          } else {
            for (int off = 0; off < vcnt; off++) {
              vmap.push_back(ind);
              voff.push_back(off);
            }
          }
        }
        MICROPROF_WARN(kN1Estimate < vmap.capacity() || kN1Estimate <
            voff.capacity(), "virtual vertex count estimate too small");
#ifndef NDEBUG
        assert(vmap.size() == voff.size());
        assert(vmap.back() == n - 1);
        const VertexId n1 = vmap.size();
        for (VertexId vind = 0; vind < n1 - 1; vind++) {
          assert(vmap[vind + 1] >= vmap[vind]);
          assert(vmap[vind + 1] <= vmap[vind] + 1);
          VertexId ind = vmap[vind];
          VertexId deg = ptr[ind + 1] - ptr[ind];
          if (vmap[vind + 1] == vmap[vind]) {
            assert(voff[vind + 1] == voff[vind] + 1);
          } else {
            assert(voff[vind + 1] == 0);
            if (ptr[ind + 1] != ptr[ind]) {
              assert(voff[vind] + 1 == (deg + ctx.kMDeg_ - 1) / ctx.kMDeg_);
            }
          }
        }
#endif  // NDEBUG
        MICROPROF_END(virtualization);
        return CONT_BIND(ctx, ptr, adj, weight, vmap, voff, ccs);
      }
  };

}  // namespace brandes

#endif  // BRANDESVCSR_H_
