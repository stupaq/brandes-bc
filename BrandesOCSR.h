/** @author Mateusz Machalica */
#ifndef BRANDESOCSR_H_
#define BRANDESOCSR_H_

#include <cassert>
#include <algorithm>

#include "./BrandesDEG1.h"

namespace brandes {

#ifndef NDEBUG
#define STATS(fmt, ...) fprintf(MICROPROF_STREAM, "STATS:\t\t" fmt, __VA_ARGS__)
  static inline void stats(const Context& ctx, const VertexList& ptr, const
      VertexList& adj, const VertexList& ccs) {
    SUPPRESS_UNUSED(adj);
    VertexId lastc = 0, maxcs = 0;
    for (auto c : ccs) {
      maxcs = std::max(c - lastc, maxcs);
      lastc = c;
    }
    STATS("biggest component\t%d / %d = %f\n", maxcs, ccs.back(),
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
    STATS("degree 0-1 count\t%d / %d = %f\n", low_count, ccs.back(),
        static_cast<float>(low_count) / ccs.back());
    STATS("high degree count\t%d / %d = %f\n", big_count, ccs.back(),
        static_cast<float>(big_count) / ccs.back());
  }
#undef STATS
#endif  // NDEBUG

  template<typename Cont> struct ocsr_create {
    template<typename Return>
      inline Return cont(Context& ctx, VertexList& ptr, VertexList& adj)
      const {
        MICROPROF_START(cc_ordering);
        const VertexId n = ptr.size() - 1;
        VertexList bfsno(n, -1);
        VertexList queue(n);
        VertexList ccs;
        auto qfront = queue.begin(), qback = queue.begin();
        VertexId bfsi = 0;
        for (VertexId root = 0; root < n; root++) {
          if (bfsno[root] >= 0) {
            continue;
          }
          ccs.push_back(bfsi);
          bfsno[root] = bfsi++;
          *qback++ = root;
          while (qfront != qback) {
            VertexId curr = *qfront++;
            assert(bfsno[curr] >= 0);
            assert(static_cast<size_t>(n) + 1 == ptr.size());
            auto next = adj.begin() + ptr[curr],
                 last = adj.begin() + ptr[curr + 1];
            while (next != last) {
              VertexId neigh = *next++;
              if (bfsno[neigh] >= 0) {
                continue;
              }
              bfsno[neigh] = bfsi++;
              *qback++ = neigh;
            }
          }
        }
        ccs.push_back(bfsi);
#ifndef NDEBUG
        for (auto no : bfsno) {
          assert(no >= 0);
        }
        for (VertexId orig = 0; orig < n; orig++) {
          assert(queue[bfsno[orig]] == orig);
        }
        assert(std::is_sorted(ccs.begin(), ccs.end()));
        assert(ccs.back() == n);
#endif  // NDEBUG
        MICROPROF_END(cc_ordering);
#ifndef NDEBUG
        stats(ctx, ptr, adj, ccs);
#endif  // NDEBUG
        return CONT_BIND(ctx, bfsno, queue, ptr, adj, ccs);
      }
  };

  template<typename Cont> struct ocsr_pass {
    template<typename Return>
      inline Return cont(Context& ctx, VertexList& ptr, VertexList& adj)
      const {
        MICROPROF_START(cc_ordering);
        const VertexId n = ptr.size() - 1;
        VertexList ccs = { 0, n };
        MICROPROF_END(cc_ordering);
#ifndef NDEBUG
        stats(ctx, ptr, adj, ccs);
#endif  // NDEBUG
        return CONT_BIND(ctx, ptr, adj, ccs);
      }
  };

}  // namespace brandes

#endif  // BRANDESOCSR_H_
