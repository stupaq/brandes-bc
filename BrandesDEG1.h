/** @author Mateusz Machalica */
#ifndef BRANDESDEG1_H_
#define BRANDESDEG1_H_

#include <cassert>
#include <vector>

#include "./BrandesStats.h"

namespace brandes {

  template<typename Cont> struct deg1_reduce {
    template<typename Return, typename Reordering>
      inline Return cont(Context& ctx, Reordering& ord, VertexList& ptr,
          VertexList& adj, VertexList& ccs) const
      {
        const VertexId n = ptr.size() - 1;
        MICROPROF_START(deg1_reduction);
        std::vector<bool> removed(n, false);
        std::vector<int> weight(n, 1), deg(n);
        std::vector<VertexId> queue(n);
        std::vector<float> bc(n);
        auto qfront = queue.begin(), qback = queue.begin();
        for (VertexId i = 0; i < n; i++) {
          deg[i] = ptr[i + 1] - ptr[i];
          if (deg[i] == 0) {
            weight[i] = 0;
          } else if (deg[i] == 1) {
            *qback++ = i;
          }
        }
        while (qfront != qback) {
          VertexId u = *qfront++;
          assert(0 == deg[u] || deg[u] == 1);
          if (deg[u] != 1) {
            continue;
          }
          bc[u] += 0;  // FIXME(stupaq)
          removed[u] = true;
          auto next = adj.begin() + ptr[u];
          const auto last = adj.begin() + ptr[u + 1];
          for (; next != last; next++) {
            VertexId v = *next;
            if (!removed[v]) {
              bc[v] += 0;  // FIXME(stupaq)
              weight[v] += weight[u];
              if (--deg[v] == 1) {
                *qback++ = v;
              }
            }
          }
        }
        MICROPROF_END(deg1_reduction);
        return CONT_BIND(ctx, ord, ptr, adj, ccs);
      }
  };

}  // namespace brandes

#endif  // BRANDESDEG1_H_
