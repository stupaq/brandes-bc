/** @author Mateusz Machalica */
#ifndef BRANDES_H_
#define BRANDES_H_

#include <cassert>
#include <cstdlib>
#include <cstdio>
#include <cstdint>

#include <vector>
#include <utility>

#include "./MicroBench.h"
#include "./MyCL.h"

#define BRANDES_VERTICES_INIT   4096

namespace brandes {

  typedef int32_t VertexId;

  struct Edge {
    VertexId v1_;
    VertexId v2_;
  };

  struct GraphGeneric {
    VertexId n_;
    std::vector<Edge> E_;

    static inline GraphGeneric read(const char* file_path) {
      MICROBENCH_START(reading_graph);
      FILE* f = fopen(file_path, "r");
      assert(f);
      std::vector<Edge> E;
      E.reserve(BRANDES_VERTICES_INIT);
      VertexId u, v, n = 0;
      while (fscanf(f, "%d %d", &u, &v) != EOF) {
        assert(u < v);
        E.push_back(Edge {u, v});
        n = (n <= v) ? v + 1 : n;
      }
      assert(!E.empty());
#ifndef NDEBUG
      for (const Edge& e : E) {
        assert(n > e.v1_ && n > e.v2_);
      }
#endif
      assert(!ferror(f));
      fclose(f);
      MICROBENCH_END(reading_graph);
      return GraphGeneric { n, E };
    }
  };

  struct GraphCSR {
    std::vector<VertexId> vptr_;
    std::vector<VertexId> eadj_;

    static inline GraphCSR create(GraphGeneric graph) {
      MICROBENCH_START(csr_transformation);
      std::vector<VertexId> vptr(graph.n_ + 1), eadj(2 * graph.E_.size());
      for (const Edge& e : graph.E_) {
        vptr[e.v1_]++;
        vptr[e.v2_]++;
      }
      assert(!vptr.empty());
      int sum = 0;
      for (int& deg : vptr) {
        int tmp = deg;
        deg = sum;
        sum += tmp;
      }
      assert((size_t) sum == 2 * graph.E_.size());
      std::vector<int> alloc(graph.n_);
      for (const Edge& e : graph.E_) {
        eadj[vptr[e.v1_] + alloc[e.v1_]++] = e.v2_;
        eadj[vptr[e.v2_] + alloc[e.v2_]++] = e.v1_;
      }
      MICROBENCH_END(csr_transformation);
    }
  };

}

#endif  // BRANDES_H_
