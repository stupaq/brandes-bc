/** @author Mateusz Machalica */
#ifndef BRANDESCSR_H_
#define BRANDESCSR_H_

#include <cassert>
#include <vector>

#include "./BrandesCOO.h"

namespace brandes {

  template<typename Cont> struct csr_create {
    template<typename Return>
      inline Return cont(
          Context& ctx,
          const VertexId n,
          const EdgeList __pass__ E
          ) const {
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
        VertexList alloc(n);
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

}  // namespace brandes

#endif  // BRANDESCSR_H_
