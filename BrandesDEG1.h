/** @author Mateusz Machalica */
#ifndef BRANDESDEG1_H_
#define BRANDESDEG1_H_

#include <cassert>
#include <vector>

#include "./BrandesCSR.h"

namespace brandes {

  template<typename Cont> struct deg1_reduct {
    template<typename Return>
      inline Return cont(Context& ctx, VertexList& ptr, VertexList& adj) const
      {
        const VertexId n = ptr.size() - 1;
        MICROPROF_START(deg1_reduction);
        std::vector<bool> removed(n, false);
        std::vector<VertexId> weight(n, 1), queue(n);
        std::vector<int> deg(n);
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
          assert(ptr[u + 1] - ptr[u] == 1);
          VertexId v = adj[ptr[u]];
          weight[v] += weight[u];
          // FIXME
          if (--deg[v] == 1) {
            *qback++ = v;
          }
        }
        MICROPROF_END(deg1_reduction);
        return CONT_BIND(ctx, ptr, adj);
      }
  };

  template<typename Cont> struct deg1_pass {
    template<typename Return>
      inline Return cont(Context& ctx, VertexList& ptr, VertexList& adj) const
      {
        return CONT_BIND(ctx, ptr, adj);
      }
  };

}  // namespace brandes

#endif  // BRANDESDEG1_H_
