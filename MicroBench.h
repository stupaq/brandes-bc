/** @author Mateusz Machalica */
#ifndef MICROBENCH_H_
#define MICROBENCH_H_

#include <chrono>
#include <iostream>

#define GCC_VERSION (__GNUC__ * 10000 \
    + __GNUC_MINOR__ * 100 \
    + __GNUC_PATCHLEVEL__)

#if GCC_VERSION < 40700
typedef std::chrono::monotonic_clock MicroBenchClock;
#else
typedef std::chrono::steady_clock MicroBenchClock;
#endif
typedef std::chrono::microseconds MicroBenchUnits;

#define MICROBENCH_START(name)\
  MicroBenchClock::time_point name ## _start = MicroBenchClock::now();

#define MICROBENCH_END(name)\
  MicroBenchClock::time_point name ## _end = MicroBenchClock::now();\
  std::cerr << #name << " : " << std::chrono::duration_cast<MicroBenchUnits>(name ## _end - name ## _start).count() << "\n";

#endif  // MICROBENCH_H_
