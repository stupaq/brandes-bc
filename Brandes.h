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

namespace brandes {
  const size_t kEdgesInit = 1<<20;

  typedef int VertexId;

  struct Edge {
    VertexId v1_;
    VertexId v2_;
  };

  struct GraphGeneric {
    VertexId n_;
    std::vector<Edge> E_;

    static inline GraphGeneric read(const char* file_path) {
      using boost::iostreams::mapped_file;
      using boost::spirit::qi::phrase_parse;
      using boost::spirit::qi::int_;
      using boost::spirit::qi::eol;
      using boost::spirit::ascii::blank;
      MICROBENCH_START(reading_graph);
      mapped_file mf(file_path, mapped_file::readonly);
      std::vector<Edge> E;
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
      return GraphGeneric { n, std::move(E) };
    }
  };

  struct GraphCSR {
    std::vector<VertexId> ptr_;
    std::vector<VertexId> adj_;
    std::vector<VertexId> ccs_;

    static inline GraphCSR create(const GraphGeneric G) {
      const VertexId n = G.n_;
      MICROBENCH_START(csr_transformation);
      std::vector<VertexId> ptr(n + 1), adj(2 * G.E_.size());
      for (const Edge& e : G.E_) {
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
      assert((size_t) sum == 2 * G.E_.size());
      std::vector<int> alloc(n);
      for (const Edge& e : G.E_) {
        adj[ptr[e.v1_] + alloc[e.v1_]++] = e.v2_;
        adj[ptr[e.v2_] + alloc[e.v2_]++] = e.v1_;
      }
#ifndef NDEBUG
      for (VertexId i = 0; i < n; i++) {
        assert(alloc[i] == ptr[i + 1] - ptr[i]);
      }
#endif
      std::vector<VertexId> ccs = { 0, n + 1 };
      MICROBENCH_END(csr_transformation);
      return GraphCSR { std::move(ptr), std::move(adj), std::move(ccs) };
    }
  };

  struct GraphOCSR {
    std::vector<VertexId> omap_;
    std::vector<VertexId> optr_;
    std::vector<VertexId> oadj_;
    std::vector<VertexId> ccs_;

    static inline GraphOCSR create(const GraphCSR G) {
      const VertexId n = G.ptr_.size() - 1;
      const auto Gitadj0 = G.adj_.begin();
      MICROBENCH_START(cc_ordering);
      std::vector<VertexId> bfsno(n, -1);
      std::vector<VertexId> queue(n);
      std::vector<VertexId> ccs = { 0 };
      {
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
            assert(static_cast<size_t>(n) + 1 == G.ptr_.size());
            auto next = Gitadj0 + G.ptr_[curr],
                 last = Gitadj0 + G.ptr_[curr + 1];
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
#ifndef NDEBUG
        for (auto no : bfsno) {
          assert(no >= 0);
        }
#endif
      }
      std::vector<VertexId> optr(G.ptr_.size()), oadj(G.adj_.size());
      {
        auto itadj0 = oadj.begin(),
             itadj = itadj0,
             itptr = optr.begin();
        for (auto curr : queue) {
          *itptr++ = itadj - itadj0;
          auto next = Gitadj0 + G.ptr_[curr],
               last = Gitadj0 + G.ptr_[curr + 1];
          while (next != last) {
            *itadj++ = bfsno[*next++];
          }
        }
        *itptr = itadj - itadj0;
        assert(static_cast<size_t>(*itptr) == G.adj_.size());
        assert(itptr + 1 == optr.end());
      }
#ifndef NDEBUG
      for (VertexId origv = 0; origv < n; origv++) {
        assert(queue[bfsno[origv]] == origv);
      }
      for (VertexId origv = 0; origv < n; origv++) {
        VertexId newv = bfsno[origv];
        assert(optr[newv + 1] - optr[newv] == G.ptr_[origv + 1] - G.ptr_[origv]);
        auto next = G.adj_.begin() + G.ptr_[origv],
             last = G.adj_.begin() + G.ptr_[origv + 1];
        auto itadj = oadj.begin() + optr[newv];
        while (next != last) {
          assert(queue[*itadj++] == *next++);
        }
      }
#endif
      MICROBENCH_END(cc_ordering);
      return GraphOCSR { std::move(bfsno), std::move(optr), std::move(oadj),
        std::move(ccs) };
    }
  };

}  // namespace brandes

BOOST_FUSION_ADAPT_STRUCT(brandes::Edge,
    (brandes::VertexId, v1_)
    (brandes::VertexId, v2_))

#endif  // BRANDES_H_
