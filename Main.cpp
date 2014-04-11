/** @author Mateusz Machalica */

#if   OPTIMIZE == 3
#pragma message "Optimized build, no error checking."
#define NDEBUG
#elif OPTIMIZE == 2
#pragma message "Partially optimized build, device error checking."
#define NDEBUG
#define MYCL_ERROR_CHECKING
#elif OPTIMIZE == 1
#pragma message "Microbenchmarks optimized build."
#define NDEBUG
#define MICROPROF_ENABLE
#define MYCL_ERROR_CHECKING
#elif OPTIMIZE == 0
#pragma message "Non-optimized build with extra assertions."
#define MICROPROF_ENABLE
#define MYCL_ERROR_CHECKING
#endif

// FIXME(stupaq) other options?
#define ALGORITHM_PIPE\
  csr_create<ocsr_create<statistics<deg1_reduce<cpu_driver<vcsr_create<betweenness>>>>>>  // NOLINT(whitespace/line_length)

#include <boost/lexical_cast.hpp>

#include <cstdio>
#include <cmath>
#include <cassert>
#include <utility>
#include <string>
#include <future>
#include <vector>

#include "./BrandesBetweenness.h"

template<typename Result>
static inline void generic_write(Result& res, const char* file_path) {
  MICROPROF_START(writing_results);
  FILE* fp = fopen(file_path, "w");
  assert(fp);
  for (auto v : res) {
    fprintf(fp, "%f\n", v);
  }
  fclose(fp);
  MICROPROF_END(writing_results);
}

static inline int log2ceil(int x) {
  return std::ceil(std::log2(x));
}

int main(int argc, const char* argv[]) {
  using namespace brandes;  // NOLINT(build/namespaces)
  MICROPROF_START(main_total);
  assert(argc > 2); SUPPRESS_UNUSED(argc);

  Context ctx = {
    std::async(std::launch::async, mycl::init_device),
    argc > 3 ? log2ceil(boost::lexical_cast<int>(argv[3])) : 4,
    argc > 4 ? log2ceil(boost::lexical_cast<int>(argv[4])) : 7,
    argc > 5 ? boost::lexical_cast<int>(argv[5]) : 1,
    argc > 6 ? boost::lexical_cast<bool>(argv[6]) : false,
  };
#ifdef MYCL_ERROR_CHECKING
  try {
#endif
    auto res = generic_read<ALGORITHM_PIPE, std::vector<float>>(ctx, argv[1]);
    generic_write(res, argv[2]);
#ifdef MYCL_ERROR_CHECKING
  } catch (cl::Error error) {
    fprintf(MYCL_STREAM, "%s (error code: %d)\n", error.what(), error.err());
  }
#endif

  MICROPROF_END(main_total);
  return 0;
}
