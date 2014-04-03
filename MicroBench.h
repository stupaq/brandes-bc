/** @author Mateusz Machalica */
#ifndef MICROBENCH_H_
#define MICROBENCH_H_

#include <cstdio>

#include <chrono>

#define GCC_VERSION (__GNUC__ * 10000 \
    + __GNUC_MINOR__ * 100 \
    + __GNUC_PATCHLEVEL__)

#if GCC_VERSION < 40700
typedef std::chrono::monotonic_clock MicroBenchClock;
#else
typedef std::chrono::steady_clock MicroBenchClock;
#endif
typedef std::chrono::microseconds MicroBenchUnits;

#ifdef MICROBENCH
#define MICROBENCH_START(name)\
  MicroBenchClock::time_point name ## _start = MicroBenchClock::now();
#define MICROBENCH_END(name)\
  MicroBenchClock::time_point name ## _end = MicroBenchClock::now();\
  fprintf(stderr, #name ":\t%ld\n", std::chrono::duration_cast<MicroBenchUnits>\
      (name ## _end - name ## _start).count());
#else
#define MICROBENCH_START(name)
#define MICROBENCH_END(name)
#endif

#endif  // MICROBENCH_H_
