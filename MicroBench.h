/** @author Mateusz Machalica */
#ifndef MICROBENCH_H_
#define MICROBENCH_H_

#include <cstdio>
#include <chrono>

#if (__GNUC__ < 4 || __GNUC_MINOR__ < 7)
typedef std::chrono::monotonic_clock MicroBenchClock;
#else
typedef std::chrono::steady_clock MicroBenchClock;
#endif

#define MICROBENCH_TIMEPOINT(name)\
  MicroBenchClock::time_point name = MicroBenchClock::now()
#define MICROBENCH_REPORT(start, end, os, fmt, units)\
  fprintf(os, fmt, std::chrono::duration_cast<units>(end - start).count())

#ifdef MICROPROF_ENABLE
typedef std::chrono::duration<double, std::milli> MicroProfUnits;
#define MICROPROF_STREAM stdout
#define MICROPROF_START(name) MICROBENCH_TIMEPOINT(name ## _start)
#define MICROPROF_END(name)\
  MICROBENCH_TIMEPOINT(name ## _end);\
  MICROBENCH_REPORT(name ## _start, name ## _end, MICROPROF_STREAM, \
      "PROFILING:\t" #name "\t%.3f\n", MicroProfUnits)
#define MICROPROF_WARN(cond, warn)\
  if (cond) fprintf(MICROPROF_STREAM, "WARNING:\t%s\n", warn)
#else
#define MICROPROF_START(name)
#define MICROPROF_END(name)
#define MICROPROF_WARN(cond, warn)
#endif

#define SUPPRESS_UNUSED(x) (static_cast<void>(x))

#endif  // MICROBENCH_H_
