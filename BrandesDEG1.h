/** @author Mateusz Machalica */
#ifndef BRANDESDEG1_H_
#define BRANDESDEG1_H_

#include <cassert>
#include <vector>

#include "./BrandesStats.h"

namespace brandes {

  template<typename Cont> struct deg1_reduce {
    template<typename Return>
      inline Return cont(
          Context& ctx,
          VertexList& ptr,
          VertexList& adj,
          const VertexList& ccs
          ) const {
        const VertexId n = ptr.size() - 1;
        MICROPROF_START(deg1_reduction);
        std::vector<float> bc(n, 0.0f);
        std::vector<int> weight(n, 1), deg(n), ccsz(n);
        VertexList queue(n), newind(n, 1);
        auto qfront = queue.begin(), qback = queue.begin();
        for (VertexId i = 0; i < n; i++) {
          deg[i] = ptr[i + 1] - ptr[i];
          assert(deg[i] >= 0);
          if (deg[i] <= 1) {
            *qback++ = i;
          }
        }
        auto itccs = ccs.begin(),
             itccsN = ccs.end() - 1;
        while (itccs != itccsN) {
          auto next = *itccs,
               end = *++itccs,
               size = end - next;
          for (; next < end; next++) {
            assert(next < ccsz.size());
            ccsz[next] = size;
          }
        }
        while (qfront != qback) {
          VertexId u = *qfront++;
          assert(0 == deg[u] || deg[u] == 1);
          if (deg[u] == 0) {
            newind[u] = 0;
          } else {
            float rest = static_cast<float>(ccsz[u] - weight[u]);
            bc[u] += rest * (weight[u] - 1);
            newind[u] = 0;
            auto next = adj.begin() + ptr[u];
            const auto last = adj.begin() + ptr[u + 1];
            for (; next != last; next++) {
              VertexId v = *next;
              if (newind[v] == 1) {
                bc[v] += weight[u] * (rest - 1);
                weight[v] += weight[u];
                if (--deg[v] == 1) {
                  *qback++ = v;
                }
              }
            }
          }
        }
        VertexId sum = 0;
        for (auto& ind : newind) {
          if (ind) {
            ind = sum;
            ++sum;
          } else {
            ind = -1;
          }
        }
        VertexId icadj = 0, nind = 0;
        for (VertexId oind = 0; oind < n; oind++) {
          if (newind[oind] >= 0) {
            assert(newind[oind] == nind);
            auto next = adj.begin() + ptr[oind];
            const auto last = adj.begin() + ptr[oind + 1];
            assert(nind <= oind);
            weight[nind] = weight[oind];
            ptr[nind++] = icadj;
            for (; next != last; next++) {
              if (newind[*next] >= 0) {
                assert(icadj <= next - adj.begin());
                adj[icadj++] = newind[*next];
              }
            }
          }
        }
        ptr[nind++] = icadj;
        ptr.resize(nind);
        weight.resize(nind - 1);
        adj.resize(icadj);
        assert(ptr.back() == adj.size());
        assert(ptr.size() > 0);
        assert(ptr.size() > 1 || ptr.back() == 0);
        assert(weight.size() == ptr.size() - 1);
        MICROPROF_END(deg1_reduction);
        if (adj.size() > 0) {
          /* We don't fix ccs because we don't use it anymore. */
          auto bc1 = CONT_BIND(ctx, ptr, adj, weight);
          MICROPROF_START(deg1_expansion);
          for (VertexId oind = 0; oind < n; oind++) {
            if (newind[oind] != -1) {
              bc[oind] += bc1[newind[oind]];
            }
          }
          MICROPROF_END(deg1_expansion);
        } else {
          fprintf(stderr, "0\n0\n");
        }
        return bc;
      }
  };

  template<typename Cont> struct deg1_pass {
    template<typename Return>
      inline Return cont(
          Context& ctx,
          VertexList& ptr,
          VertexList& adj,
          const VertexList&
          ) const {
        const VertexId n = ptr.size() - 1;
        // TODO(stupaq) sir, it can be done better
        std::vector<int> weight(n, 1);
        return CONT_BIND(ctx, ptr, adj, weight);
      }
  };

}  // namespace brandes

#endif  // BRANDESDEG1_H_
