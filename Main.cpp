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

#ifndef DEFAULT_MDEG_LOG2
#define DEFAULT_MDEG_LOG2 4
#endif

#ifndef DEFAULT_WGROUP_LOG2
#define DEFAULT_WGROUP_LOG2 7
#endif

#ifndef DEFAULT_CPU_JOBS
#define DEFAULT_CPU_JOBS 3
#endif

#ifndef DEFAULT_USE_GPU
#define DEFAULT_USE_GPU true
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
  using boost::lexical_cast;
  using namespace brandes;  // NOLINT(build/namespaces)
  MICROPROF_START(main_total);
  assert(argc > 2); SUPPRESS_UNUSED(argc);

  Context ctx = {
    std::async(std::launch::async, mycl::init_device),
    argc > 3 ? log2ceil(lexical_cast<int>(argv[3])) : DEFAULT_MDEG_LOG2,
    argc > 4 ? log2ceil(lexical_cast<int>(argv[4])) : DEFAULT_WGROUP_LOG2,
    argc > 5 ? lexical_cast<int>(argv[5]) : DEFAULT_CPU_JOBS,
    argc > 6 ? lexical_cast<bool>(argv[6]) : DEFAULT_USE_GPU,
  };
#ifdef MYCL_ERROR_CHECKING
  try {
#endif
    auto res = generic_read<ALGORITHM_PIPE>(ctx, argv[1]);
    generic_write(res, argv[2]);
#ifdef MYCL_ERROR_CHECKING
  } catch (cl::Error error) {
    fprintf(MYCL_STREAM, "%s (error code: %d)\n", error.what(), error.err());
  }
#endif

  MICROPROF_END(main_total);
  return 0;
}
