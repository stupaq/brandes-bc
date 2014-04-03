/** @author Mateusz Machalica */
#ifndef BRANDES_H_
#define BRANDES_H_

#include <boost/fusion/adapted.hpp>
#include <boost/spirit/include/qi.hpp>
#include <boost/iostreams/device/mapped_file.hpp>

#include <cassert>
#include <cstdlib>
#include <cstdio>
#include <cstdint>

#include <vector>
#include <utility>

#include "./MicroBench.h"
#include "./MyCL.h"

#define CONT_BIND(...) return Cont().template cont<Return>(__VA_ARGS__);

namespace brandes {

  typedef int VertexId;
  typedef std::vector<VertexId> VertexList;

  struct Edge {
    VertexId v1_;
    VertexId v2_;
  };

  typedef std::vector<Edge> EdgeList;

  template<typename Cont, typename Return>
    inline Return generic_read(const char* file_path) {
      const size_t kEdgesInit = 1<<20;
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

  template<typename Cont> struct csr_create {
    template<typename Return>
      inline Return cont(const VertexId n, EdgeList& E) const {
        MICROBENCH_START(adjacency);
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
        MICROBENCH_END(adjacency);
        CONT_BIND(ptr, adj);
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

  template<typename Cont> struct ocsr_create {
    template<typename Return>
      inline Return cont(VertexList& ptr, VertexList& adj) const {
        MICROBENCH_START(cc_ordering);
        const VertexId n = ptr.size() - 1;
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
#endif
        Permutation ord = { bfsno };
        MICROBENCH_END(cc_ordering);
        CONT_BIND(ord, queue, ptr, adj, ccs);
      }
  };

  template<typename Cont> struct ocsr_pass {
    template<typename Return>
      inline Return cont(VertexList& ptr, VertexList& adj) const {
        MICROBENCH_START(cc_ordering);
        const VertexId n = ptr.size() - 1;
        Identity id;
        VertexList queue;
        VertexList ccs = { 0, n };
        MICROBENCH_END(cc_ordering);
        CONT_BIND(id, queue, ptr, adj, ccs);
      }
  };

  template<int kMDeg, typename Cont> struct vcsr_create {
    template<typename Return, typename Reordering>
      inline Return cont(Reordering& ord, const VertexList& queue, const
          VertexList& ptr, const VertexList& adj, const VertexList& ccs) const
      {
        MICROBENCH_START(virtualization);
        VertexList vptr, vmap, vccs;
        vccs.reserve(ccs.size());
        // TODO(stupaq) this is pretty fair estimate, leave unless profiling
        // shows multiple reallocations happening
        const size_t kN1Estimate = ptr.size() + adj.size() / kMDeg;
        vptr.reserve(kN1Estimate);
        vmap.reserve(kN1Estimate);
        VertexList oadj(adj.size());
        auto itoadj0 = oadj.begin(),
             itoadj = itoadj0;
        VertexId aggr = 0;
        for (auto orig : queue) {
          auto next = adj.begin() + ptr[orig],
               last = adj.begin() + ptr[orig + 1];
          if (next == last) {
            vptr.push_back(itoadj - itoadj0);
            vmap.push_back(aggr);
          } else {
            for (int i = 0; next != last; i++, itoadj++, next++) {
              if (i % kMDeg == 0) {
                vptr.push_back(itoadj - itoadj0);
                vmap.push_back(aggr);
              }
              *itoadj = ord[*next];
            }
          }
          aggr++;
        }
        vptr.push_back(itoadj - itoadj0);
        vmap.push_back(aggr);
        const VertexId n1 = vptr.size() - 1;
        auto itccs = ccs.begin();
        for (VertexId virt = 0; virt <= n1; virt++) {
          if (vmap[virt] == *itccs) {
            vccs.push_back(virt);
            itccs++;
          }
        }
        MICROBENCH_WARN(kN1Estimate < vptr.capacity(), "vptr estimate too small");
        MICROBENCH_WARN(kN1Estimate < vmap.capacity(), "vmap estimate too small");
#ifndef NDEBUG
        assert(static_cast<size_t>(vptr.back()) == adj.size());
        assert(std::is_sorted(vptr.begin(), vptr.end()));
        assert(std::is_sorted(vmap.begin(), vmap.end()));
        assert(vmap.size() == vptr.size());
        for (VertexId virt = 0; virt < n1; virt++) {
          assert(vmap[virt + 1] >= vmap[virt]);
          assert(vmap[virt + 1] <= vmap[virt] + 1);
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
        assert(vccs.front() == 0);
        assert(vccs.back() == n1);
        assert(std::is_sorted(vccs.begin(), vccs.end()));
        assert(vccs.size() == ccs.size());
#endif
        MICROBENCH_END(virtualization);
        CONT_BIND(ord, vptr, oadj, vccs);
      }

    template<typename Return>
      inline Return cont(Identity& ord, const VertexList&, const VertexList&
          ptr, VertexList& adj, const VertexList& ccs) const {
        MICROBENCH_START(virtualization);
        const VertexId n = ptr.size() - 1;
        VertexList vptr, vmap, vccs;
        vccs.reserve(ccs.size());
        // TODO(stupaq) this is pretty fair estimate, leave unless profiling
        // shows multiple reallocations happening
        const size_t kN1Estimate = ptr.size() + adj.size() / kMDeg;
        vptr.reserve(kN1Estimate);
        vmap.reserve(kN1Estimate);
        auto itccs = ccs.begin();
        for (VertexId orig = 0; orig < n; orig++) {
          VertexId virt = vptr.size(),
                   first = ptr[orig],
                   last = ptr[orig + 1];
          if (orig == *itccs) {
            vccs.push_back(virt);
            itccs++;
          }
          do {
            vptr.push_back(first);
            vmap.push_back(orig);
            first += kMDeg;
          } while (first < last);
        }
        vptr.push_back(ptr[n]);
        vmap.push_back(n);
        const VertexId n1 = vptr.size() - 1;
        vccs.push_back(n1);
        MICROBENCH_WARN(kN1Estimate < vptr.capacity(), "vptr estimate too small");
        MICROBENCH_WARN(kN1Estimate < vmap.capacity(), "vmap estimate too small");
#ifndef NDEBUG
        assert(static_cast<size_t>(vptr.back()) == adj.size());
        assert(std::is_sorted(vmap.begin(), vmap.end()));
        assert(vmap.size() == vptr.size());
        for (VertexId virt = 0; virt < n1; virt++) {
          assert(vmap[virt + 1] >= vmap[virt]);
          assert(vmap[virt + 1] <= vmap[virt] + 1);
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
        assert(vccs.front() == 0);
        assert(vccs.back() == n1);
        assert(std::is_sorted(vccs.begin(), vccs.end()));
        assert(vccs.size() == ccs.size());
#endif
        MICROBENCH_END(virtualization);
        CONT_BIND(ord, vptr, adj, vccs);
      }
  };

  template<typename Cont> struct vcsr_pass {
    template<typename Return, typename Reordering>
      inline Return cont(Reordering& ord, const VertexList& queue, VertexList&
          ptr, const VertexList& adj, VertexList& ccs) const {
        MICROBENCH_START(virtualization);
        const VertexId n = ptr.size() - 1;
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
        CONT_BIND(ord, optr, oadj, ccs);
      }

    template<typename Return>
      inline Return cont(Identity& ord, const VertexList&, VertexList& ptr,
          VertexList& adj, VertexList& ccs) const {
        MICROBENCH_START(virtualization);
        MICROBENCH_END(virtualization);
        CONT_BIND(ord, ptr, adj, ccs);
      }
  };

  // TODO(stupaq) this is only a placeholder
  struct Terminal {
    template<typename Return, typename Reordering>
      inline Return cont(Reordering&, VertexList&, VertexList&, VertexList&)
      const {
        return 0;
      }
  };

}  // namespace brandes

BOOST_FUSION_ADAPT_STRUCT(brandes::Edge,
    (brandes::VertexId, v1_)
    (brandes::VertexId, v2_))

#endif  // BRANDES_H_
