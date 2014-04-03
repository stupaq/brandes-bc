/** @author Mateusz Machalica */
#ifndef BRANDES_H_
#define BRANDES_H_

#include <cassert>
#include <cstdlib>
#include <cstdio>
#include <cstdint>

#include <vector>
#include <utility>

#include <boost/fusion/adapted.hpp>
#include <boost/spirit/include/qi.hpp>
#include <boost/iostreams/device/mapped_file.hpp>

#include "./MicroBench.h"
#include "./MyCL.h"

#define CONT_BIND(...) return Cont().template cont<Return>(__VA_ARGS__);

namespace brandes {
  const size_t kEdgesInit = 1<<20;

  typedef int VertexId;
  typedef std::vector<VertexId> VertexList;

  struct Edge {
    VertexId v1_;
    VertexId v2_;
  };

  typedef std::vector<Edge> EdgeList;

  template<typename Cont, typename Return>
    inline Return generic_read(const char* file_path) {
      using boost::iostreams::mapped_file;
      using boost::spirit::qi::phrase_parse;
      using boost::spirit::qi::int_;
      using boost::spirit::qi::eol;
      using boost::spirit::ascii::blank;
      MICROBENCH_START(reading_graph);
      mapped_file mf(file_path, mapped_file::readonly);
      EdgeList E;
      E.reserve(kEdgesInit);
      {
        auto dat0 = mf.const_data(), dat1 = dat0 + mf.size();
        bool r = phrase_parse(dat0, dat1, (int_ >> int_) % eol > eol, blank, E);
        assert(r);
        assert(dat0 == dat1);
      }
      VertexId n = 0;
      for (auto& e : E) {
        assert(e.v1_ < e.v2_);
        n = (n <= e.v2_) ? e.v2_ + 1 : n;
      }
      assert(!E.empty());
#ifndef NDEBUG
      for (const Edge& e : E) {
        assert(n > e.v1_ && n > e.v2_);
      }
#endif
      MICROBENCH_END(reading_graph);
      CONT_BIND(n, E);
    }

  template<typename Cont> struct CSRCreate {
    template<typename Return>
      inline Return cont(const VertexId n, EdgeList& E) const {
        MICROBENCH_START(csr_transformation);
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
#endif
        MICROBENCH_END(csr_transformation);
        CONT_BIND(n, ptr, adj);
      }
  };

  struct Permutation {
    VertexList perm_;
    inline VertexId operator[](VertexId orig) const {
      return perm_[orig];
    }
  };

  struct Identity {
    inline VertexId operator[](VertexId orig) const {
      return orig;
    }
  };

  template<typename Cont> struct OCSRCreate {
    template<typename Return>
      inline Return cont(const VertexId n, VertexList& ptr,
          VertexList& adj) const {
        MICROBENCH_START(cc_ordering);
        VertexList bfsno(n, -1);
        VertexList queue(n);
        VertexList ccs = { 0 };
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
          ccs.push_back(n);
#ifndef NDEBUG
        for (auto no : bfsno) {
          assert(no >= 0);
        }
        for (VertexId orig = 0; orig < n; orig++) {
          assert(queue[bfsno[orig]] == orig);
        }
#endif
        Permutation ord = { bfsno };
        MICROBENCH_END(cc_ordering);
        CONT_BIND(ord, queue, n, ptr, adj, ccs);
      }
  };

  template<typename Cont> struct OCSRPass {
    template<typename Return>
      inline Return cont(const VertexId n, VertexList& ptr,
          VertexList& adj) const {
        MICROBENCH_START(cc_ordering);
        Identity id;
        VertexList queue;
        VertexList ccs = { 0, n + 1 };
        MICROBENCH_END(cc_ordering);
        CONT_BIND(id, queue, n, ptr, adj, ccs);
      }
  };

  template<typename Cont, int kMDeg = 4> struct VCSRCreate {
    template<typename Return, typename Reordering>
      inline Return cont(Reordering& ord, const VertexList& queue, const VertexId
          n, const VertexList& ptr, const VertexList& adj, VertexList& ccs)
      const {
        // FIXME ccs
        MICROBENCH_START(virtualization);
        VertexList vptr, vmap;
        // TODO(stupaq) this is pretty fair estimate, leave unless profiling
        // shows multiple reallocations happening
        vptr.reserve(ptr.size() + adj.size() / 4);
        vmap.reserve(ptr.size() + adj.size() / 4);
        VertexList oadj(adj.size());
          auto itoadj0 = oadj.begin(),
               itoadj = itoadj0;
          VertexId aggr = 0;
          for (auto orig : queue) {
            auto next = adj.begin() + ptr[orig],
                 last = adj.begin() + ptr[orig + 1];
            for (int i = 0; next != last; i++, itoadj++, next++) {
              if (i % kMDeg == 0) {
                vptr.push_back(itoadj - itoadj0);
                vmap.push_back(aggr);
              }
              *itoadj = ord[*next];
            }
            aggr++;
          }
          vptr.push_back(itoadj - itoadj0);
          vmap.push_back(aggr);
        const VertexId n1 = vptr.size() - 1;
#ifndef NDEBUG
        assert(static_cast<size_t>(vptr.back()) == adj.size());
        assert(vmap.size() == vptr.size());
        for (VertexId virt = 0; virt < n1; virt++) {
          VertexId orig = queue[vmap[virt]];
          assert(vptr[virt + 1] - vptr[virt] <= kMDeg);
          assert(vptr[virt + 1] - vptr[virt] <= ptr[orig + 1] - ptr[orig]);
          if (ptr[orig + 1] != ptr[orig]) {
            assert(vptr[virt + 1] > vptr[virt]);
          }
          if (virt == 0 || vmap[virt - 1] != vmap[virt]) {
            auto next = adj.begin() + ptr[orig],
                 last = adj.begin() + ptr[orig + 1];
            auto itoadj = oadj.begin() + vptr[virt];
            while (next != last) {
              assert(queue[*itoadj++] == *next++);
            }
          }
        }
#endif
        MICROBENCH_END(virtualization);
        // FIXME
        return 0;
      }

    template<typename Return>
      inline Return cont(Identity&, const VertexList&, const VertexId n,
          const VertexList& ptr, VertexList& adj, VertexList& ccs)
      const {
        // FIXME ccs
        MICROBENCH_START(virtualization);
        VertexList vptr, vmap;
        // TODO(stupaq) this is pretty fair estimate, leave unless profiling
        // shows multiple reallocations happening
        vptr.reserve(ptr.size() + adj.size() / 4);
        vmap.reserve(ptr.size() + adj.size() / 4);
        for (VertexId orig = 0; orig < n; orig++) {
          VertexId first = ptr[orig],
                   last = ptr[orig + 1];
          for (; first < last; first += kMDeg) {
            vptr.push_back(first);
            vmap.push_back(orig);
          }
        }
        vptr.push_back(ptr[n]);
        vmap.push_back(n);
        const VertexId n1 = vptr.size() - 1;
#ifndef NDEBUG
        assert(static_cast<size_t>(vptr.back()) == adj.size());
        assert(vmap.size() == vptr.size());
        for (VertexId virt = 0; virt < n1; virt++) {
          VertexId orig = vmap[virt];  // queue == id
          assert(vptr[virt + 1] - vptr[virt] <= kMDeg);
          assert(vptr[virt + 1] - vptr[virt] <= ptr[orig + 1] - ptr[orig]);
          if (ptr[orig + 1] != ptr[orig]) {
            assert(vptr[virt + 1] > vptr[virt]);
          }
          if (virt == 0 || vmap[virt - 1] != vmap[virt]) {
            assert(vptr[virt] == ptr[orig]);
          }
        }
#endif
        MICROBENCH_END(virtualization);
        // FIXME
        return 0;
      }
  };

  template<typename Cont> struct VCSRPass {
    template<typename Return, typename Reordering>
      inline Return cont(Reordering& ord, const VertexList& queue, const
          VertexId n, VertexList& ptr, const VertexList& adj, VertexList& ccs)
      const {
        MICROBENCH_START(virtualization);
        VertexList optr(ptr.size()), oadj(adj.size());
        auto itoadj0 = oadj.begin(),
             itoadj = itoadj0,
             itoptr = optr.begin();
        for (auto curr : queue) {
          *itoptr++ = itoadj - itoadj0;
          auto next = adj.begin() + ptr[curr],
               last = adj.begin() + ptr[curr + 1];
          while (next != last) {
            *itoadj++ = ord[*next++];
          }
        }
        *itoptr = itoadj - itoadj0;
#ifndef NDEBUG
        assert(static_cast<size_t>(*itoptr) == adj.size());
        assert(itoptr + 1 == optr.end());
        for (VertexId orig = 0; orig < n; orig++) {
          VertexId ordv = ord[orig];
          assert(optr[ordv + 1] - optr[ordv] == ptr[orig + 1] - ptr[orig]);
          auto next = adj.begin() + ptr[orig],
               last = adj.begin() + ptr[orig + 1];
          auto itoadj = oadj.begin() + optr[ordv];
          while (next != last) {
            assert(queue[*itoadj++] == *next++);
          }
        }
#endif
        MICROBENCH_END(virtualization);
        // FIXME
        return 0;
      }

    template<typename Return>
      inline Return cont(Identity&, const VertexList&, const VertexId n,
          VertexList& ptr, VertexList& adj, VertexList& ccs)
      const {
        MICROBENCH_START(virtualization);
        MICROBENCH_END(virtualization);
        // FIXME
        return 0;
      }
  };

}  // namespace brandes

BOOST_FUSION_ADAPT_STRUCT(brandes::Edge,
    (brandes::VertexId, v1_)
    (brandes::VertexId, v2_))

#endif  // BRANDES_H_
