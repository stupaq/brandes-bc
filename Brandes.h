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

  template<typename Cont>
    struct CSRCreate {
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
    inline VertexId operator()(VertexId orig) const {
      return perm_[orig];
    }
  };

  struct Identity {
    inline VertexId operator()(VertexId orig) const {
      return orig;
    }
  };

  template<typename Cont>
    struct OCSRCreate {
      template<typename Return>
        inline Return cont(const VertexId n, VertexList& ptr,
            VertexList& adj) const {
          const auto itadj0 = adj.begin();
          MICROBENCH_START(cc_ordering);
          VertexList bfsno(n, -1);
          VertexList queue(n);
          VertexList ccs = { 0 };
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
                assert(static_cast<size_t>(n) + 1 == ptr.size());
                auto next = itadj0 + ptr[curr],
                     last = itadj0 + ptr[curr + 1];
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
          VertexList optr(ptr.size()), oadj(adj.size());
          {
            auto itoadj0 = oadj.begin(),
                 itoadj = itoadj0,
                 itoptr = optr.begin();
            for (auto curr : queue) {
              *itoptr++ = itoadj - itoadj0;
              auto next = itadj0 + ptr[curr],
                   last = itadj0 + ptr[curr + 1];
              while (next != last) {
                *itoadj++ = bfsno[*next++];
              }
            }
            *itoptr = itoadj - itoadj0;
            assert(static_cast<size_t>(*itoptr) == adj.size());
            assert(itoptr + 1 == optr.end());
          }
#ifndef NDEBUG
          for (VertexId origv = 0; origv < n; origv++) {
            assert(queue[bfsno[origv]] == origv);
          }
          for (VertexId origv = 0; origv < n; origv++) {
            VertexId newv = bfsno[origv];
            assert(optr[newv + 1] - optr[newv] == ptr[origv + 1] - ptr[origv]);
            auto next = adj.begin() + ptr[origv],
                 last = adj.begin() + ptr[origv + 1];
            auto itoadj = oadj.begin() + optr[newv];
            while (next != last) {
              assert(queue[*itoadj++] == *next++);
            }
          }
#endif
          MICROBENCH_END(cc_ordering);
          CONT_BIND(Permutation { std::move(bfsno) }, n, optr, oadj, ccs);
        }
    };

  template<typename Cont>
    struct OCSRPass {
      template<typename Return>
        inline Return cont(const VertexId n, VertexList& ptr,
            VertexList& adj) const {
          VertexList ccs = { 0, n + 1 };
          CONT_BIND(Identity {}, n, ptr, adj, ccs);
        }
    };

  template<typename Cont>
    struct VCSRCreate {
      template<typename Return, typename Reordering>
        inline int cont(Reordering r, const VertexId n, VertexList& ptr,
            VertexList& adj, VertexList& ccs) const {
          return 0;
        }
    };

}  // namespace brandes

BOOST_FUSION_ADAPT_STRUCT(brandes::Edge,
    (brandes::VertexId, v1_)
    (brandes::VertexId, v2_))

#endif  // BRANDES_H_
