Implementation of Brandes algorithm in OpenCL.
==============================================

Running `make` without any parameters compiles the fastest implementation.
You can adjust the algorithm by passing the following defines in `CPPFLAGS`.
* `-DOPTIMIZE=n` - sets optimization level, ranges from 0 to 3, where 0
  enables all assertions and 3 disables all safety checks
* `-DDEFAULT_MDEG=n` - sets virtual vertex degree, see the report
* `-DDEFAULT_WGROUP=n` - sets work group size
* `-DDEFAULT_CPU_JOBS=n` - sets number of CPU workers to use
* `-DDEFAULT_USE_GPU=true/false` - turns on/off GPU acceleration
* `-DNO_DEG1` - disables tree contraction
* `-DNO_BFS` - disables BFS ordering of the graph
* `-DNO_STATS` - disables printing graph statistics

