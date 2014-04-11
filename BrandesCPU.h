/** @author Mateusz Machalica */
#ifndef BRANDESCPU_H_
#define BRANDESCPU_H_

#include <cassert>
#include <vector>
#include <atomic>
#include <future>

#include "./BrandesDEG1.h"

namespace brandes {

  template<typename Return>
    static inline Return bc_cpu_worker(
        const VertexList __pass__ ptr,
        const VertexList __pass__ adj,
        const Return __pass__ weight,
        /* This sounds like a bug in stdlib++, I couldn't pass atomic by
         * reference to std::async task... */
        std::atomic_int* source_dispatch
        ) {
      const VertexId n = ptr.size() - 1;
      Return bc(n, 0.0f), delta(n);
      VertexList queue(n);
      VertexList dist(n), sigma(n);
      VertexId source, processed_count = 0;
      while ((source = (*source_dispatch)++) < n) {
        auto qfront = queue.begin(), qback = qfront;
        /* Init source. */
        std::fill(dist.begin(), dist.end(), -1);
        dist[source] = 0;
        std::fill(sigma.begin(), sigma.end(), 0);
        sigma[source] = 1;
        *qback++ = source;
        /* Forward. */
        while (qfront != qback) {
          VertexId v = *qfront++;
          assert(v < n);
          auto itadj = adj.begin() + ptr[v];
          const auto itadjN = adj.begin() + ptr[v + 1];
          while (itadj != itadjN) {
            VertexId w = *itadj++;
            assert(w < n);
            if (dist[w] < 0) {
              *qback++ = w;
              dist[w] = dist[v] + 1;
            }
            if (dist[w] == dist[v] + 1) {
              sigma[w] += sigma[v];
              assert(sigma[w] >= 0);
            }
          }
        }
        /* Intermediate. */
        for (VertexId v = 0; v < n; v++) {
          delta[v] = weight[v] / sigma[v];
        }
        /* Backward. */
        assert(qfront == qback);
        auto sfront = std::reverse_iterator<VertexList::iterator>(qback),
             sback = queue.rend();
        while (sfront != sback) {
          VertexId w = *sfront++;
          assert(w < n);
          auto itadj = adj.begin() + ptr[w];
          const auto itadjN = adj.begin() + ptr[w + 1];
          while (itadj != itadjN) {
            VertexId v = *itadj++;
            assert(v < n);
            if (dist[w] == dist[v] + 1) {
              delta[v] += delta[w];
            }
          }
        }
        /* Sum. */
        for (VertexId v = 0; v < n; v++) {
          if (v != source && dist[v] >= 0) {
            bc[v] += (delta[v] * sigma[v] - 1) * weight[source];
          }
        }
        processed_count++;
      }
      MICROPROF_INFO("CPU_WORKER:\tsources processed:\t%d\n", processed_count);
      return bc;
    }

  template<typename Cont> struct cpu_driver {
    template<typename Return>
      inline Return cont(
          Context& ctx,
          const VertexList __pass__ ptr,
          const VertexList __pass__ adj,
          const Return __pass__ weight
          ) const {
        assert(ctx.kUseGPU_ || ctx.kCPUJobs_ > 0);
        MICROPROF_INFO("CONFIGURATION:\tshould use GPU\t%d\n", ctx.kUseGPU_);
        MICROPROF_INFO("CONFIGURATION:\tCPU jobs count\t%d\n", ctx.kCPUJobs_);
        const VertexId n = ptr.size() - 1;
        std::atomic_int source_dispatch(0);
        MICROPROF_WARN(!source_dispatch.is_lock_free(),
            "Atomic integer is not lock free.");
        std::vector<std::future<Return>> cpu_jobs;
        MICROPROF_START(cpu_scheduling);
        for (int i = 0; i < ctx.kCPUJobs_; i++) {
          cpu_jobs.push_back(std::async(std::launch::async,
                brandes::bc_cpu_worker<Return>,
                ptr, adj, weight, &source_dispatch));
        }
        MICROPROF_END(cpu_scheduling);
        Return bc = ctx.kUseGPU_
          ? CONT_BIND(ctx, ptr, adj, weight, source_dispatch)
          : Return(n, 0.0f);
        if (!ctx.kUseGPU_) {
          fprintf(stderr, "0\n0\n");
        }
        MICROPROF_START(cpu_driver_combine);
        for (auto& cpu_job : cpu_jobs) {
          auto bc1  = cpu_job.get();
          assert(bc.size() == bc1.size());
          auto itbc = bc.begin(),
               itbc1 = bc1.begin();
          const auto itbcN = bc.end();
          while (itbc != itbcN) {
            *itbc++ += *itbc1++;
          }
        }
        MICROPROF_END(cpu_driver_combine);
        return bc;
      }
  };

}  // namespace brandes

#endif  // BRANDESCPU_H_
