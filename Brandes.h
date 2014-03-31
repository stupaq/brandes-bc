/** @author Mateusz Machalica */
#ifndef BRANDES_H_
#define BRANDES_H_

#include <cassert>
#include <cstdlib>
#include <cstdio>

#include <vector>
#include <utility>

#define __NO_STD_VECTOR
#define __CL_ENABLE_EXCEPTIONS
#include <CL/cl.hpp>

#include "./MicroBench.h"

#define BRANDES_EDGES_INIT      4096

namespace brandes {

  typedef int VertexId;
  typedef cl_int CLVertexId;
  typedef std::pair<VertexId, VertexId> Edge;
  typedef std::vector<Edge> GraphGeneric;

  GraphGeneric read_graph_generic(const char* file_path) {
    MICROBENCH_START(reading_graph);
    FILE* f = fopen(file_path, "r");
    assert(f);
    GraphGeneric graph;
    int u, v;
    while (fscanf(f, "%d %d", &u, &v) != EOF) {
      graph.push_back(std::make_pair(u, v));
    }
    assert(!ferror(f));
    fclose(f);
    MICROBENCH_END(reading_graph);
    return graph;
  }

  struct GraphCSR {
    CLVertexId n_;
    CLVertexId* ptrs_;
    CLVertexId* adjs_;
  };



}

#endif  // BRANDES_H_
