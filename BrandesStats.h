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
    template<typename Return, typename Reordering>
      inline Return cont(Context& ctx, Reordering& ord, VertexList& ptr,
          VertexList& adj, VertexList& ccs) const {
        MICROPROF_START(statistics);
        VertexId lastc = 0, maxcs = 0;
        for (auto c : ccs) {
          maxcs = std::max(c - lastc, maxcs);
          lastc = c;
        }
        PRINT_STATS("biggest component\t%d / %d = %f\n", maxcs, ccs.back(),
            static_cast<float>(maxcs) / ccs.back());
        const int low_thr = 2, big_thr = ctx.kMDeg_;
        int low_count = 0, big_count = 0;
        VertexId last_p = - (low_thr + big_thr) / 2;
        for (auto p : ptr) {
          int deg = p - last_p;
          if (deg < low_thr) {
            low_count++;
          }
          if (deg > big_thr) {
            big_count++;
          }
          last_p = p;
        }
        PRINT_STATS("degree 0-1 count\t%d / %d = %f\n", low_count, ccs.back(),
            static_cast<float>(low_count) / ccs.back());
        PRINT_STATS("high degree count\t%d / %d = %f\n", big_count, ccs.back(),
            static_cast<float>(big_count) / ccs.back());
        MICROPROF_END(statistics);
        return CONT_BIND(ctx, ord, ptr, adj, ccs);
      }
  };

}  // namespace brandes

#undef PRINT_STATS

#endif  // BRANDESSTATS_H_
