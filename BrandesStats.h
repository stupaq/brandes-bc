/** @author Mateusz Machalica */
#ifndef BRANDESSTATS_H_
#define BRANDESSTATS_H_

#include <cassert>
#include <vector>
#include <algorithm>

#include "./BrandesOCSR.h"

#define PRINT_STATS(fmt, ...)\
  fprintf(MICROPROF_STREAM, "STATS:\t\t" fmt, __VA_ARGS__)

namespace brandes {

  template<typename Cont> struct statistics {
    template<typename Return>
      inline Return cont(
          Context& ctx,
          VertexList __pass__ ptr,
          VertexList __pass__ adj,
          VertexList __pass__ ccs
          ) const {
        MICROPROF_START(statistics);
        VertexId lastc = 0, maxcs = 0;
        for (auto c : ccs) {
          maxcs = std::max(c - lastc, maxcs);
          lastc = c;
        }
        PRINT_STATS("biggest component\t%d / %d = %f\n", maxcs, ccs.back(),
            static_cast<float>(maxcs) / ccs.back());
        const VertexId low_thr = 2, big_thr = 1 << ctx.kMDegLog2_;
        VertexId low_count = 0, big_count = 0;
        VertexId last_p = - (low_thr + big_thr) / 2;
        for (auto p : ptr) {
          VertexId deg = p - last_p;
          if (deg < low_thr) {
            low_count++;
          }
          if (deg > big_thr) {
            big_count++;
          }
          last_p = p;
        }
        PRINT_STATS("degree < %d count\t%d / %d = %f\n", low_thr, low_count,
            ccs.back(), static_cast<float>(low_count) / ccs.back());
        PRINT_STATS("degree > %d count\t%d / %d = %f\n", big_thr, big_count,
            ccs.back(), static_cast<float>(big_count) / ccs.back());
        MICROPROF_END(statistics);
        return CONT_BIND(ctx, ptr, adj, ccs);
      }
  };

}  // namespace brandes

#undef PRINT_STATS

#endif  // BRANDESSTATS_H_
