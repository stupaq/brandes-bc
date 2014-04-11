/** @author Mateusz Machalica */
#ifndef BRANDESOCSR_H_
#define BRANDESOCSR_H_

#include <cassert>
#include <algorithm>

#include "./BrandesCSR.h"

namespace brandes {

  template<typename Cont> struct ocsr_create {
    template<typename Return>
      inline Return cont(
          Context& ctx,
          const VertexList& ptr,
          const VertexList& adj
          ) const {
        MICROPROF_START(bfs_ordering);
        const VertexId n = ptr.size() - 1;
        VertexList bfsno(n, -1), queue(n), ccs;
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
        VertexList optr(ptr.size()), oadj(adj.size());
        auto itoadj0 = oadj.begin(),
             itoadj = itoadj0,
             itoptr = optr.begin();
        for (auto curr : queue) {
          *itoptr++ = itoadj - itoadj0;
          auto next = adj.begin() + ptr[curr],
               last = adj.begin() + ptr[curr + 1];
          while (next != last) {
            *itoadj++ = bfsno[*next++];
          }
        }
        *itoptr = itoadj - itoadj0;
#ifndef NDEBUG
        assert(static_cast<size_t>(*itoptr) == adj.size());
        assert(itoptr + 1 == optr.end());
        for (VertexId orig = 0; orig < n; orig++) {
          VertexId ordv = bfsno[orig];
          assert(optr[ordv + 1] - optr[ordv] == ptr[orig + 1] - ptr[orig]);
          auto next = adj.begin() + ptr[orig],
               last = adj.begin() + ptr[orig + 1];
          auto itoadj = oadj.begin() + optr[ordv];
          while (next != last) {
            assert(queue[*itoadj++] == *next++);
          }
        }
#endif  // NDEBUG
        MICROPROF_END(bfs_ordering);
        auto bc1 = CONT_BIND(ctx, optr, oadj, ccs);
        Return bc(bc1.size());
        for (VertexId orig = 0, end = bc1.size(); orig < end; orig++) {
          bc[orig] = bc1[bfsno[orig]];
        }
        return bc;
      }
  };

  template<typename Cont> struct ocsr_pass {
    template<typename Return>
      inline Return cont(
          Context& ctx,
          const VertexList& ptr,
          const VertexList& adj
          ) const {
        MICROPROF_START(bfs_ordering);
        const VertexId n = ptr.size() - 1;
        VertexList ccs = { 0, n };
        MICROPROF_END(bfs_ordering);
        return CONT_BIND(ctx, ptr, adj, ccs);
      }
  };

}  // namespace brandes

#endif  // BRANDESOCSR_H_
