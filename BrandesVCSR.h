/** @author Mateusz Machalica */
#ifndef BRANDESVCSR_H_
#define BRANDESVCSR_H_

#include <cassert>
#include <vector>

#include "./BrandesDEG1.h"

namespace brandes {

  template<typename Int> inline Int round_up(Int value, Int factor) {
    Int factor_mask = (1 << factor) - 1;
    return value + factor_mask - ((value - 1) & factor_mask);
  }

  template<typename Int> inline Int divide_up(Int value, Int factor) {
    return (value + (1 << factor) - 1) >> factor;
  }

  template<typename Cont> struct vcsr_create {
    template<typename Return, typename Weights>
      inline Return cont(
          Context& ctx,
          const VertexList& ptr,
          const VertexList& adj,
          const Weights& weight
          ) const {
        MICROPROF_INFO("CONFIGURATION:\tvirtualized deg\t%d\n",
            1 << ctx.kMDegLog2_);
        MICROPROF_START(virtualization);
        const size_t kN1Estimate = ptr.size() + (adj.size() >> ctx.kMDegLog2_)
          + 10;
        const VertexId n = ptr.size() - 1;
        VertexList vmap, voff;
        vmap.reserve(kN1Estimate);
        voff.reserve(kN1Estimate);
        for (VertexId ind = 0; ind < n; ind++) {
          VertexId deg = ptr[ind + 1] - ptr[ind],
                   vcnt = divide_up(deg, ctx.kMDegLog2_);
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
        vmap.push_back(n);
        voff.push_back(0);
        MICROPROF_WARN(kN1Estimate < vmap.capacity() || kN1Estimate <
            voff.capacity(), "virtual vertex count estimate too small");
#ifndef NDEBUG
        assert(vmap.size() == voff.size());
        assert(vmap.back() == n);
        const VertexId n1 = vmap.size() - 1;
        for (VertexId vind = 0; vind < n1; vind++) {
          assert(vmap[vind + 1] >= vmap[vind]);
          assert(vmap[vind + 1] <= vmap[vind] + 1);
          VertexId ind = vmap[vind];
          VertexId deg = ptr[ind + 1] - ptr[ind];
          if (vmap[vind + 1] == vmap[vind]) {
            assert(voff[vind + 1] == voff[vind] + 1);
          } else {
            assert(voff[vind + 1] == 0);
            if (ptr[ind + 1] != ptr[ind]) {
              const int kMDeg = 1 << ctx.kMDegLog2_;
              assert(voff[vind] + 1 == (deg + kMDeg - 1) / kMDeg);
            }
          }
        }
#endif  // NDEBUG
        MICROPROF_END(virtualization);
        return CONT_BIND(ctx, vmap, voff, ptr, adj, weight);
      }
  };

}  // namespace brandes

#endif  // BRANDESVCSR_H_
