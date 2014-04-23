/** @author Mateusz Machalica */
#ifndef BRANDESBETWEENNESS_H_
#define BRANDESBETWEENNESS_H_

#include <cassert>
#include <vector>
#include <chrono>
#include <atomic>
#include <algorithm>

#include "./BrandesVCSR.h"

namespace brandes {
  using mycl::bytes;

  struct betweenness {
    template<typename Int>
      static inline Int round_up(Int value, Int factor) {
        return value + factor - 1 - ((value - 1) % factor);
      }

    template<typename Return, typename VertexList>
      inline Return cont(
          Context& ctx,
          const VertexList __pass__ vmap,
          const VertexList __pass__ voff,
          const VertexList __pass__ ptr,
          const VertexList __pass__ adj,
          const Return __pass__ weight,
          std::atomic_int& source_dispatch
          ) const {
        typedef typename VertexList::value_type VertexId;
        typedef typename Return::value_type Result;
        static_assert(sizeof(VertexId) == sizeof(cl_int),
            "VertexId type not compatible");
        static_assert(sizeof(SigmaInt) == sizeof(cl_int),
            "SigmaInt type not compatible");
        static_assert(sizeof(Result) == sizeof(cl_float),
            "Result type not compatible");

        /* We prepare our ranges for one extra thread (for inits). */
        cl::NDRange local(ctx.kWGroup_);
        const VertexId n = ptr.size() - 1;
        cl::NDRange n_global(round_up(n + 1, ctx.kWGroup_));
        const VertexId n1 = vmap.size() - 1;
        cl::NDRange n1_global(round_up(n1 + 1, ctx.kWGroup_));

        assert(local.dimensions() == 1);
        assert(n_global.dimensions() == 1);
        assert(n1_global.dimensions() == 1);
        assert(n_global[0] * local[0] >= (n + 1) * ctx.kWGroup_);
        assert(n1_global[0] * local[0] >= (n1 + 1) * ctx.kWGroup_);

        assert(vmap.back() == n);
        assert(voff.back() == 0);
        assert(ptr.back() == adj.size());

        MICROPROF_INFO("CONFIGURATION:\twork group\t%d\n", ctx.kWGroup_);
        MICROPROF_START(device_wait);
        Accelerator acc = ctx.dev_future_.get();
        cl::CommandQueue& q = acc.queue_;
        MICROPROF_END(device_wait);

        MICROBENCH_TIMEPOINT(moving_data);
        MICROPROF_START(graph_to_gpu);
        cl::Buffer proceed_cl(acc.context_, CL_MEM_READ_WRITE, sizeof(bool));
        cl::Buffer vmap_cl(acc.context_, CL_MEM_READ_ONLY, bytes(vmap));
        q.enqueueWriteBuffer(vmap_cl, false, 0, bytes(vmap), vmap.data());
        cl::Buffer voff_cl(acc.context_, CL_MEM_READ_ONLY, bytes(voff));
        q.enqueueWriteBuffer(voff_cl, false, 0, bytes(voff), voff.data());
        cl::Buffer rmap_cl(acc.context_, CL_MEM_READ_ONLY, bytes(vmap));
        cl::Buffer ptr_cl(acc.context_, CL_MEM_READ_ONLY, bytes(ptr));
        q.enqueueWriteBuffer(ptr_cl, false, 0, bytes(ptr), ptr.data());
        cl::Buffer adj_cl(acc.context_, CL_MEM_READ_ONLY, bytes(adj));
        q.enqueueWriteBuffer(adj_cl, false, 0, bytes(adj), adj.data());
        cl::Buffer weight_cl(acc.context_, CL_MEM_READ_ONLY, bytes(weight));
        q.enqueueWriteBuffer(weight_cl, false, 0, bytes(weight), weight.data());
        cl::Buffer
          dist_cl(acc.context_, CL_MEM_READ_WRITE, sizeof(VertexId) * n),
          sigma_cl(acc.context_, CL_MEM_READ_WRITE, sizeof(SigmaInt) * n),
          delta_cl(acc.context_, CL_MEM_READ_WRITE, sizeof(Result) * n),
          red_cl(acc.context_, CL_MEM_READ_WRITE, std::max(sizeof(SigmaInt),
                sizeof(Result)) * n1),
          bc_cl(acc.context_, CL_MEM_READ_WRITE, sizeof(Result) * n);
        MICROPROF_END(graph_to_gpu);

        /* Architecture of the driver hides latencies of starting all kernels
         * but the first one. We measure kernel execution times using CPU wall
         * clock, because otherwise, when usin OpenCL profiling-enabled command
         * queue, we would have to wait for events completion and keep the
         * device underutilized. */
        MICROBENCH_TIMEPOINT(starting_kernels);
        { /* This approach appears to be measurably faster for big graphs. */
          cl::Kernel k_init_n(acc.program_, "vcsr_init_n");
          k_init_n.setArg(0, n);
          k_init_n.setArg(1, bc_cl);
          q.enqueueNDRangeKernel(k_init_n, cl::NullRange, n_global, local);
        }
        { /* Note that n1_global range is prepared for one extra thread. */
          cl::Kernel k_init_n1(acc.program_, "vcsr_init_n1");
          k_init_n1.setArg(0, n1 + 1);
          k_init_n1.setArg(1, n1);
          k_init_n1.setArg(2, vmap_cl);
          k_init_n1.setArg(3, voff_cl);
          k_init_n1.setArg(4, rmap_cl);
          q.enqueueNDRangeKernel(k_init_n1, cl::NullRange, n1_global, local);
        }

        /** We can move some arguments setting outside of the loop. */
        cl::Kernel k_source(acc.program_, "vcsr_init_source");
        k_source.setArg(0, n);
        k_source.setArg(2, proceed_cl);
        k_source.setArg(3, dist_cl);
        k_source.setArg(4, sigma_cl);
        cl::Kernel k_fwd(acc.program_, "vcsr_forward");
        k_fwd.setArg(0, n1);
        k_fwd.setArg(2, ctx.kMDegLog2_);
        k_fwd.setArg(3, proceed_cl);
        k_fwd.setArg(4, vmap_cl);
        k_fwd.setArg(5, voff_cl);
        k_fwd.setArg(6, ptr_cl);
        k_fwd.setArg(7, adj_cl);
        k_fwd.setArg(8, dist_cl);
        k_fwd.setArg(9, sigma_cl);
        k_fwd.setArg(10, red_cl);
        cl::Kernel k_fwd_red(acc.program_, "vcsr_forward_reduce");
        k_fwd_red.setArg(0, n);
        k_fwd_red.setArg(2, proceed_cl);
        k_fwd_red.setArg(3, rmap_cl);
        k_fwd_red.setArg(4, weight_cl);
        k_fwd_red.setArg(5, dist_cl);
        k_fwd_red.setArg(6, sigma_cl);
        k_fwd_red.setArg(7, delta_cl);
        k_fwd_red.setArg(8, red_cl);
        cl::Kernel k_back(acc.program_, "vcsr_backward");
        k_back.setArg(0, n1);
        k_back.setArg(2, ctx.kMDegLog2_);
        k_back.setArg(3, vmap_cl);
        k_back.setArg(4, voff_cl);
        k_back.setArg(5, ptr_cl);
        k_back.setArg(6, adj_cl);
        k_back.setArg(7, dist_cl);
        k_back.setArg(8, delta_cl);
        k_back.setArg(9, red_cl);
        cl::Kernel k_back_red(acc.program_, "vcsr_backward_reduce");
        k_back_red.setArg(0, n);
        k_back_red.setArg(2, rmap_cl);
        k_back_red.setArg(3, dist_cl);
        k_back_red.setArg(4, delta_cl);
        k_back_red.setArg(5, red_cl);
        cl::Kernel k_sum(acc.program_, "vcsr_sum");
        k_sum.setArg(0, n);
        k_sum.setArg(2, weight_cl);
        k_sum.setArg(3, dist_cl);
        k_sum.setArg(4, sigma_cl);
        k_sum.setArg(5, delta_cl);
        k_sum.setArg(6, bc_cl);

        VertexId source;
        while ((source = source_dispatch++) < n) {
          k_source.setArg(1, source);
          q.enqueueNDRangeKernel(k_source, cl::NullRange, n_global, local);

          bool proceed;
          VertexId curr_dist = 0;
          do {
            k_fwd.setArg(1, curr_dist);
            q.enqueueNDRangeKernel(k_fwd, cl::NullRange, n1_global, local);
            /* Note that we must first obtain proceed flag and then run
             * parallel reduction kernel as it sets proceed to false. */
            cl::Event evt;
            q.enqueueReadBuffer(proceed_cl, false, 0, sizeof(bool), &proceed,
                NULL, &evt);
            /* Performing aggregation for source (curr_dist == 0) is not
             * correct since we explicitly set sigma[source] = 1. */
            if (curr_dist > 0) {
              k_fwd_red.setArg(1, curr_dist);
              q.enqueueNDRangeKernel(k_fwd_red, cl::NullRange, n_global, local);
            }
            curr_dist++;
            /* The fact that we use specific event instead of clFinish() call
             * makes GPU busy at all times, once buffer reading completes we
             * can read the value and GPU executes the kernel without any lag.
             * This effectively hides kernel setup time. */
            evt.wait();
          } while (proceed);

          while (--curr_dist > 0) {
            k_back.setArg(1, curr_dist);
            q.enqueueNDRangeKernel(k_back, cl::NullRange, n1_global, local);
            k_back_red.setArg(1, curr_dist);
            q.enqueueNDRangeKernel(k_back_red, cl::NullRange, n_global, local);
          }

          k_sum.setArg(1, source);
          q.enqueueNDRangeKernel(k_sum, cl::NullRange, n_global, local);

          if (source % (n / 24) == 0) {
            MICROPROF_INFO("PROGRESS:\t%d / %d\n", source, n);
          }
        }
        q.finish();
        MICROBENCH_TIMEPOINT(kernels_completed);

        Return bc(n);
        q.enqueueReadBuffer(bc_cl, true, 0, bytes(bc), bc.data());
        q.finish();
        MICROBENCH_TIMEPOINT(fetched_results);

        MICROBENCH_REPORT(starting_kernels, kernels_completed, stderr, "%ld\n",
            std::chrono::milliseconds);
        MICROBENCH_REPORT(moving_data, fetched_results, stderr, "%ld\n",
            std::chrono::milliseconds);
        return bc;
      }
  };

}  // namespace brandes

#endif  // BRANDESBETWEENNESS_H_
