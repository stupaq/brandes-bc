/** @author Mateusz Machalica */
#ifndef BRANDESCSR_H_
#define BRANDESCSR_H_

#include <cassert>
#include <vector>

#include "./BrandesCOO.h"

namespace brandes {

  template<typename Cont> struct csr_create {
    template<typename Return>
      inline Return cont(Context& ctx, const VertexId n, EdgeList& E) const {
        MICROPROF_START(adjacency);
        VertexList ptr(n + 1), adj(2 * E.size());
        for (auto e : E) {
          ptr[e.v1_]++;
          ptr[e.v2_]++;
        }
        assert(!ptr.empty());
        VertexId sum = 0;
        for (auto& d : ptr) {
          VertexId tmp = d;
          d = sum;
          sum += tmp;
        }
        assert((size_t) sum == 2 * E.size());
        std::vector<int> alloc(n);
        for (auto e : E) {
          adj[ptr[e.v1_] + alloc[e.v1_]++] = e.v2_;
          adj[ptr[e.v2_] + alloc[e.v2_]++] = e.v1_;
        }
#ifndef NDEBUG
        for (VertexId i = 0; i < n; i++) {
          assert(alloc[i] == ptr[i + 1] - ptr[i]);
        }
#endif  // NDEBUG
        MICROPROF_END(adjacency);
        return CONT_BIND(ctx, ptr, adj);
      }
  };

  template<typename Cont> struct csr_reduct {
    template<typename Return>
      inline Return cont(Context& ctx, VertexList& ptr, VertexList& adj) const
      {
        const VertexId n = ptr.size() - 1;
        MICROPROF_START(graph_reduction);
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
        MICROPROF_END(graph_reduction);
        return CONT_BIND(ctx, ptr, adj);
      }
  };

}  // namespace brandes

#endif  // BRANDESCSR_H_
