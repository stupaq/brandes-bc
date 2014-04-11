/** @author Mateusz Machalica */
#ifndef BRANDESCPU_H_
#define BRANDESCPU_H_

#include <cassert>
#include <vector>
#include <atomic>
#include <future>
#include <list>

#include "./BrandesDEG1.h"

namespace brandes {

  template<typename Return, typename Weights>
    static inline Return bc_cpu_worker(
        const VertexList& ptr,
        const VertexList& adj,
        const Weights& weight,
        /* This sounds like a bug in stdlib++, I couldn't pass atomic by
         * reference to std::async task... */
        std::atomic_int* source_dispatch
        ) {
      typedef typename Return::value_type FloatType;
      SUPPRESS_UNUSED(ptr);
      SUPPRESS_UNUSED(adj);
      SUPPRESS_UNUSED(weight);
      const VertexId n = ptr.size() - 1;
      Return bc(n, 0.0f), delta(n);
      VertexList queue(n);
      std::vector<int> dist(n), sigma(n);
      // TODO(stupaq) get rid of this if possible, it breaks cache locality
      std::vector<std::list<VertexId>> pred(n);
      VertexId source, processed_count = 0;
      while ((source = source_dispatch->operator++()) < n) {
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
              pred[w].push_back(v);
            }
          }
        }

        /* Intermediate. */
        for (VertexId v = 0; v < n; v++) {
          delta[v] = static_cast<FloatType>(weight[v]) / sigma[v];
        }

        /* Backward. */
        assert(qfront == qback);
        qfront = queue.begin();
        while (qfront != qback) {
          VertexId w = *--qback;
          for (auto v : pred[w]) {
            delta[v] += delta[w];
          }
          pred[w].clear();
        }

        /* Sum. */
        for (VertexId v = 0; v < source; v++) {
          bc[v] += delta[v] * sigma[v] - 1;
        }

        processed_count++;
      }
      MICROPROF_INFO("CPU_WORKER:\tsources processed:\t%d\n", processed_count);
      return bc;
    }

  template<typename Cont> struct cpu_driver {
    template<typename Return, typename Weights>
      inline Return cont(
          Context& ctx,
          const VertexList& ptr,
          const VertexList& adj,
          const Weights& weight
          ) const {
        assert(ctx.kUseGPU_ || ctx.kCPUJobs_ > 0);
        MICROPROF_INFO("CONFIGURATION:\tshould use GPU\t%d\n", ctx.kUseGPU_);
        MICROPROF_INFO("CONFIGURATION:\tCPU jobs count\t%d\n", ctx.kCPUJobs_);
        const VertexId n = ptr.size() - 1;
        std::atomic_int source_dispatch(0);
        MICROPROF_WARN(!source_dispatch.is_lock_free(),
            "Atomic integer is not lock free.");
        std::vector<std::future<Return>> cpu_jobs;
        MICROPROF_START(cpu_starting_jobs);
        for (int i = 0; i < ctx.kCPUJobs_; i++) {
          cpu_jobs.push_back(std::async(std::launch::async,
                brandes::bc_cpu_worker<Return, Weights>,
                ptr, adj, weight, &source_dispatch));
        }
        MICROPROF_END(cpu_starting_jobs);
        Return bc = ctx.kUseGPU_
          ? CONT_BIND(ctx, ptr, adj, weight, source_dispatch)
          : Return(n, 0.0f);
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
