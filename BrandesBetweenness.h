/** @author Mateusz Machalica */
#ifndef BRANDESBETWEENNESS_H_
#define BRANDESBETWEENNESS_H_

#include <cassert>
#include <vector>
#include <chrono>
#include <atomic>

#include "./BrandesVCSR.h"

namespace brandes {
  using mycl::bytes;

  struct betweenness {
    template<typename Return>
      inline Return cont(
          Context& ctx,
          const VertexList& vmap,
          const VertexList& voff,
          const VertexList& ptr,
          const VertexList& adj,
          const Return& weight,
          std::atomic_int& source_dispatch
          ) const {
        MICROPROF_INFO("CONFIGURATION:\twork group\t%d\n",
            1 << ctx.kWGroupLog2_);
        MICROPROF_START(device_wait);
        Accelerator acc = ctx.dev_future_.get();
        cl::CommandQueue& q = acc.queue_;
        MICROPROF_END(device_wait);

        /* We prepare our ranges for one extra thread (for inits). */
        cl::NDRange local(1 << ctx.kWGroupLog2_);
        const VertexId n = ptr.size() - 1;
        cl::NDRange n_global(round_up(n + 1, ctx.kWGroupLog2_));
        const VertexId n1 = vmap.size() - 1;
        cl::NDRange n1_global(round_up(n1 + 1, ctx.kWGroupLog2_));
        assert(vmap.back() == n);
        assert(voff.back() == 0);
        assert(ptr.back() == adj.size());

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
        cl::Buffer dist_cl(acc.context_, CL_MEM_READ_WRITE, sizeof(int) * n);
        cl::Buffer sigma_cl(acc.context_, CL_MEM_READ_WRITE, sizeof(int) * n);
        cl::Buffer delta_cl(acc.context_, CL_MEM_READ_WRITE, sizeof(float) * n);
        cl::Buffer red_cl(acc.context_, CL_MEM_READ_WRITE, sizeof(float) * n1);
        cl::Buffer bc_cl(acc.context_, CL_MEM_READ_WRITE, sizeof(float) * n);
        MICROPROF_END(graph_to_gpu);

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
        k_source.setArg(2, dist_cl);
        k_source.setArg(3, sigma_cl);
        cl::Kernel k_fwd(acc.program_, "vcsr_forward");
        k_fwd.setArg(0, n1);
        k_fwd.setArg(2, ctx.kMDegLog2_);
        k_fwd.setArg(3, proceed_cl);
        k_fwd.setArg(4, vmap_cl);
        k_fwd.setArg(5, voff_cl);
        k_fwd.setArg(6, ptr_cl);
        k_fwd.setArg(7, adj_cl);
        k_fwd.setArg(8, weight_cl);
        k_fwd.setArg(9, dist_cl);
        k_fwd.setArg(10, sigma_cl);
        k_fwd.setArg(11, delta_cl);
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

        int source;
        while ((source = source_dispatch++) < n) {
          k_source.setArg(1, source);
          q.enqueueNDRangeKernel(k_source, cl::NullRange, n_global, local);

          bool proceed;
          int curr_dist = 0;
          do {
            proceed = false;
            q.enqueueWriteBuffer(proceed_cl, false, 0, sizeof(bool), &proceed);
            k_fwd.setArg(1, curr_dist++);
            q.enqueueNDRangeKernel(k_fwd, cl::NullRange, n1_global, local);
            q.enqueueReadBuffer(proceed_cl, true, 0, sizeof(bool), &proceed);
            q.finish();
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

        Return bc(n);
        q.enqueueReadBuffer(bc_cl, true, 0, bytes(bc), bc.data());
        q.finish();

        MICROBENCH_TIMEPOINT(fetched_results);
        MICROBENCH_REPORT(starting_kernels, fetched_results, stderr, "%ld\n",
            std::chrono::milliseconds);
        MICROBENCH_REPORT(moving_data, fetched_results, stderr, "%ld\n",
            std::chrono::milliseconds);
        return bc;
      }
  };

}  // namespace brandes

#endif  // BRANDESBETWEENNESS_H_
