/** @author Mateusz Machalica */
#ifndef BRANDES_H_
#define BRANDES_H_

#include <boost/fusion/adapted.hpp>
#include <boost/spirit/include/qi.hpp>
#include <boost/iostreams/device/mapped_file.hpp>

#include <cassert>
#include <cstdlib>
#include <cstdio>
#include <cstdint>

#include <vector>
#include <utility>
#include <future>

#include "./MicroBench.h"
#include "./MyCL.h"

#define CONT_BIND(...) return Cont().template cont<Return>(__VA_ARGS__)
#define ROUND_UP(value, factor) (value + factor - 1 - (value - 1) % factor)

namespace brandes {
  using mycl::Accelerator;
  using mycl::bytes;

  typedef std::future<Accelerator> Context;

  typedef cl_int VertexId;
  typedef std::vector<VertexId> VertexList;

  struct Edge {
    VertexId v1_;
    VertexId v2_;
  };
  typedef std::vector<Edge> EdgeList;

  struct Virtual {
    VertexId ptr_;
    VertexId map_;
    Virtual(VertexId ptr, VertexId map) : ptr_(ptr), map_(map) {}
  } __attribute__((packed));
  typedef std::vector<Virtual> VirtualList;

  template<typename Cont, typename Return>
    inline Return generic_read(Context& ctx, const char* file_path) {
      const size_t kEdgesInit = 1<<20;
      using boost::iostreams::mapped_file;
      using boost::spirit::qi::phrase_parse;
      using boost::spirit::qi::int_;
      using boost::spirit::qi::eol;
      using boost::spirit::ascii::blank;
      MICROPROF_START(reading_graph);
      mapped_file mf(file_path, mapped_file::readonly);
      EdgeList E;
      E.reserve(kEdgesInit);
      {
        auto dat0 = mf.const_data(), dat1 = dat0 + mf.size();
        bool r = phrase_parse(dat0, dat1, (int_ >> int_) % eol > eol, blank, E);
        assert(r); SUPPRESS_UNUSED(r);
        assert(dat0 == dat1);
      }
      VertexId n = 0;
      for (auto& e : E) {
        assert(e.v1_ < e.v2_);
        n = (n <= e.v2_) ? e.v2_ + 1 : n;
      }
      assert(!E.empty());
#ifndef NDEBUG
      for (const Edge& e : E) {
        assert(n > e.v1_ && n > e.v2_);
      }
#endif  // NDEBUG
      MICROPROF_END(reading_graph);
      CONT_BIND(ctx, n, E);
    }

  template<typename Cont> struct csr_create {
    template<typename Return>
      inline Return cont(Context& ctx, const VertexId n, EdgeList& E) const {
        MICROPROF_START(adjacency);
        VertexList ptr(n + 1), adj(2 * E.size());
        for (auto e : E) {
          ptr[e.v1_]++;
          ptr[e.v2_]++;
        }
        assert(!ptr.empty());
        VertexId sum = 0;
        for (auto& d : ptr) {
          VertexId tmp = d;
          d = sum;
          sum += tmp;
        }
        assert((size_t) sum == 2 * E.size());
        std::vector<int> alloc(n);
        for (auto e : E) {
          adj[ptr[e.v1_] + alloc[e.v1_]++] = e.v2_;
          adj[ptr[e.v2_] + alloc[e.v2_]++] = e.v1_;
        }
#ifndef NDEBUG
        for (VertexId i = 0; i < n; i++) {
          assert(alloc[i] == ptr[i + 1] - ptr[i]);
        }
#endif  // NDEBUG
        MICROPROF_END(adjacency);
        CONT_BIND(ctx, ptr, adj);
      }
  };

  struct Permutation {
    VertexList perm_;
    inline VertexId operator[](VertexId orig) const {
      return perm_[orig];
    }
  };

  struct Identity {
    inline VertexId operator[](VertexId orig) const {
      return orig;
    }
  };

  template<typename Cont> struct ocsr_create {
    template<typename Return>
      inline Return cont(Context& ctx, VertexList& ptr, VertexList& adj)
      const {
        MICROPROF_START(cc_ordering);
        const VertexId n = ptr.size() - 1;
        VertexList bfsno(n, -1);
        VertexList queue(n);
        VertexList ccs;
        auto qfront = queue.begin(), qback = queue.begin();
        VertexId bfsi = 0;
        for (VertexId root = 0; root < n; root++) {
          if (bfsno[root] >= 0) {
            continue;
          }
          ccs.push_back(bfsi);
          bfsno[root] = bfsi++;
          *qback++ = root;
          while (qfront != qback) {
            VertexId curr = *qfront++;
            assert(bfsno[curr] >= 0);
            assert(static_cast<size_t>(n) + 1 == ptr.size());
            auto next = adj.begin() + ptr[curr],
                 last = adj.begin() + ptr[curr + 1];
            while (next != last) {
              VertexId neigh = *next++;
              if (bfsno[neigh] >= 0) {
                continue;
              }
              bfsno[neigh] = bfsi++;
              *qback++ = neigh;
            }
          }
        }
        ccs.push_back(bfsi);
#ifndef NDEBUG
        for (auto no : bfsno) {
          assert(no >= 0);
        }
        for (VertexId orig = 0; orig < n; orig++) {
          assert(queue[bfsno[orig]] == orig);
        }
        assert(std::is_sorted(ccs.begin(), ccs.end()));
        assert(ccs.back() == n);
#endif  // NDEBUG
        Permutation ord = { bfsno };
        MICROPROF_END(cc_ordering);
        CONT_BIND(ctx, ord, queue, ptr, adj, ccs);
      }
  };

  template<typename Cont> struct ocsr_pass {
    template<typename Return>
      inline Return cont(Context& ctx, VertexList& ptr, VertexList& adj)
      const {
        MICROPROF_START(cc_ordering);
        const VertexId n = ptr.size() - 1;
        VertexList ccs = { 0, n };
        MICROPROF_END(cc_ordering);
        CONT_BIND(ctx, ptr, adj, ccs);
      }
  };

  template<int kMDeg, typename Cont> struct vcsr_create {
    template<typename Return, typename Reordering>
      inline Return cont(Context& ctx, Reordering& ord, const VertexList&
          queue, const VertexList& ptr, const VertexList& adj, const
          VertexList& ccs) const
      {
        MICROPROF_START(virtualization);
        VirtualList vlst;
        VertexList vccs;
        vccs.reserve(ccs.size());
        const size_t kN1Estimate = ptr.size() + adj.size() / kMDeg;
        vlst.reserve(kN1Estimate);
        VertexList oadj(adj.size());
        auto itoadj0 = oadj.begin(),
             itoadj = itoadj0;
        VertexId aggr = 0;
        for (auto orig : queue) {
          auto next = adj.begin() + ptr[orig],
               last = adj.begin() + ptr[orig + 1];
          if (next == last) {
            vlst.push_back(Virtual(itoadj - itoadj0, aggr));
          } else {
            for (int i = 0; next != last; i++, itoadj++, next++) {
              if (i % kMDeg == 0) {
                vlst.push_back(Virtual(itoadj - itoadj0, aggr));
              }
              *itoadj = ord[*next];
            }
          }
          aggr++;
        }
        vlst.push_back(Virtual(itoadj - itoadj0, aggr));
        auto itccs = ccs.begin();
        for (VertexId virt = 0, end = vlst.size(); virt < end; virt++) {
          if (vlst[virt].map_ == *itccs) {
            vccs.push_back(virt);
            itccs++;
          }
        }
        MICROPROF_WARN(kN1Estimate < vlst.capacity(),
            "vlst estimate too small");
#ifndef NDEBUG
        assertions(queue, ptr, adj, ccs, vlst, oadj, vccs);
#endif  // NDEBUG
        MICROPROF_END(virtualization);
        CONT_BIND(ctx, ord, vlst, oadj, vccs);
      }

    template<typename Return>
      inline Return cont(Context& ctx, const VertexList& ptr, VertexList& adj,
          const VertexList& ccs) const {
        MICROPROF_START(virtualization);
        const VertexId n = ptr.size() - 1;
        VirtualList vlst;
        VertexList vccs;
        vccs.reserve(ccs.size());
        const size_t kN1Estimate = ptr.size() + adj.size() / kMDeg;
        vlst.reserve(kN1Estimate);
        auto itccs = ccs.begin();
        for (VertexId orig = 0; orig < n; orig++) {
          VertexId virt = vlst.size(),
                   first = ptr[orig],
                   last = ptr[orig + 1];
          if (orig == *itccs) {
            vccs.push_back(virt);
            itccs++;
          }
          do {
            vlst.push_back(Virtual(first, orig));
            first += kMDeg;
          } while (first < last);
        }
        vccs.push_back(vlst.size());
        vlst.push_back(Virtual(ptr[n], n));
        MICROPROF_WARN(kN1Estimate < vlst.capacity(),
            "vlst estimate too small");
        Identity id;
#ifndef NDEBUG
        assertions(id, ptr, adj, ccs, vlst, adj, vccs);
#endif  // NDEBUG
        MICROPROF_END(virtualization);
        CONT_BIND(ctx, id, vlst, adj, vccs);
      }

#ifndef NDEBUG
    private:
    template<typename Reordering>
      static inline void assertions(const Reordering& rord, const VertexList&
          ptr, const VertexList& adj, const VertexList& ccs, const VirtualList&
          vlst, const VertexList& oadj, const VertexList& vccs) {
        const VertexId n1 = vlst.size() - 1;
        assert(static_cast<size_t>(vlst.back().ptr_) == adj.size());
        for (VertexId virt = 0; virt < n1; virt++) {
          assert(vlst[virt + 1].map_ >= vlst[virt].map_);
          assert(vlst[virt + 1].map_ <= vlst[virt].map_ + 1);
          VertexId orig = rord[vlst[virt].map_];
          assert(vlst[virt + 1].ptr_ >= vlst[virt].ptr_);
          assert(vlst[virt + 1].ptr_ - vlst[virt].ptr_ <= kMDeg);
          assert(vlst[virt + 1].ptr_ - vlst[virt].ptr_
              <= ptr[orig + 1] - ptr[orig]);
          if (ptr[orig + 1] != ptr[orig]) {
            assert(vlst[virt + 1].ptr_ > vlst[virt].ptr_);
          }
          if (virt == 0 || vlst[virt - 1].map_ != vlst[virt].map_) {
            auto next = adj.begin() + ptr[orig],
                 last = adj.begin() + ptr[orig + 1];
            auto itoadj = oadj.begin() + vlst[virt].ptr_;
            while (next != last) {
              assert(rord[*itoadj++] == *next++);
            }
          }
        }
        assert(vccs.front() == 0);
        assert(vccs.back() == n1);
        assert(std::is_sorted(vccs.begin(), vccs.end()));
        assert(vccs.size() == ccs.size());
      }
#endif  // NDEBUG
  };

  template<typename Cont> struct vcsr_pass {
    template<typename Return, typename Reordering>
      inline Return cont(Context& ctx, Reordering& ord, const VertexList&
          queue, VertexList& ptr, const VertexList& adj, VertexList& ccs) const
      {
        MICROPROF_START(virtualization);
        VertexList optr(ptr.size()), oadj(adj.size());
        auto itoadj0 = oadj.begin(),
             itoadj = itoadj0,
             itoptr = optr.begin();
        for (auto curr : queue) {
          *itoptr++ = itoadj - itoadj0;
          auto next = adj.begin() + ptr[curr],
               last = adj.begin() + ptr[curr + 1];
          while (next != last) {
            *itoadj++ = ord[*next++];
          }
        }
        *itoptr = itoadj - itoadj0;
#ifndef NDEBUG
        const VertexId n = ptr.size() - 1;
        assert(static_cast<size_t>(*itoptr) == adj.size());
        assert(itoptr + 1 == optr.end());
        for (VertexId orig = 0; orig < n; orig++) {
          VertexId ordv = ord[orig];
          assert(optr[ordv + 1] - optr[ordv] == ptr[orig + 1] - ptr[orig]);
          auto next = adj.begin() + ptr[orig],
               last = adj.begin() + ptr[orig + 1];
          auto itoadj = oadj.begin() + optr[ordv];
          while (next != last) {
            assert(queue[*itoadj++] == *next++);
          }
        }
#endif  // NDEBUG
        MICROPROF_END(virtualization);
        CONT_BIND(ctx, ord, optr, oadj, ccs);
      }

    template<typename Return>
      inline Return cont(Context& ctx, VertexList& ptr, VertexList& adj,
          VertexList& ccs) const {
        MICROPROF_START(virtualization);
        Identity id;
        MICROPROF_END(virtualization);
        CONT_BIND(ctx, id, ptr, adj, ccs);
      }
  };

  template<int kWGroup, typename Cont> struct betweenness {
    template<typename Return, typename Reordering>
      inline Return cont(Context& ctx, Reordering& ord, VirtualList& vlst,
          VertexList& adj, VertexList&) const {
        // TODO(stupaq) making use of connected components information
        // might require aligning component's boundaries with warp size
        // otherwise we might get bank conflicts
        MICROPROF_START(device_wait);
        Accelerator acc = ctx.get();
        cl::CommandQueue& q = acc.queue_;
        MICROPROF_END(device_wait);

        cl::NDRange local(kWGroup);
        const int n = vlst.back().map_;
        cl::NDRange n_global(ROUND_UP(n, kWGroup));
        const int n1 = vlst.size();
        cl::NDRange n1_global(ROUND_UP(n1, kWGroup));

        MICROBENCH_TIMEPOINT(moving_data);
        MICROPROF_START(graph_to_gpu);
        // TODO(stupaq) do we need to make these writes synchronously?
        cl::Buffer proceed_cl(acc.context_, CL_MEM_READ_WRITE, sizeof(bool));
        cl::Buffer vlst_cl(acc.context_, CL_MEM_READ_ONLY, bytes(vlst));
        cl::Buffer adj_cl(acc.context_, CL_MEM_READ_ONLY, bytes(adj));
        cl::Buffer ds_cl(acc.context_, CL_MEM_READ_WRITE, 2 * sizeof(int) * n);
        cl::Buffer delta_cl(acc.context_, CL_MEM_READ_WRITE, sizeof(float) * n);
        cl::Buffer bc_cl(acc.context_, CL_MEM_READ_WRITE, sizeof(float) * n);
        q.enqueueWriteBuffer(vlst_cl, true, 0, bytes(vlst), vlst.data());
        q.enqueueWriteBuffer(adj_cl, true, 0, bytes(adj), adj.data());
        MICROPROF_END(graph_to_gpu);

        MICROBENCH_TIMEPOINT(starting_kernels);
        /* This approach appears to be measurably faster for bigger graphs
           and we do not really care about dozens of us for small ones. */
        cl::Kernel k_init(acc.program_, "vcsr_init");
        k_init.setArg(0, n);
        k_init.setArg(1, bc_cl);
        q.enqueueNDRangeKernel(k_init, cl::NullRange, n_global, local);

        /** We can move some arguments setting outside of the loop. */
        cl::Kernel k_source(acc.program_, "vcsr_init_source");
        k_source.setArg(0, n);
        k_source.setArg(2, ds_cl);
        cl::Kernel k_fwd(acc.program_, "vcsr_forward");
        k_fwd.setArg(0, n1);
        k_fwd.setArg(2, proceed_cl);
        k_fwd.setArg(3, vlst_cl);
        k_fwd.setArg(4, adj_cl);
        k_fwd.setArg(5, ds_cl);
        cl::Kernel k_interm(acc.program_, "vcsr_interm");
        k_interm.setArg(0, n);
        k_interm.setArg(1, ds_cl);
        k_interm.setArg(2, delta_cl);
        cl::Kernel k_back(acc.program_, "vcsr_backward");
        k_back.setArg(0, n1);
        k_back.setArg(2, vlst_cl);
        k_back.setArg(3, adj_cl);
        k_back.setArg(4, ds_cl);
        k_back.setArg(5, delta_cl);
        cl::Kernel k_sum(acc.program_, "vcsr_sum");
        k_sum.setArg(0, n);
        k_sum.setArg(2, ds_cl);
        k_sum.setArg(3, delta_cl);
        k_sum.setArg(4, bc_cl);

        for (int source = 0; source < n; source++) {
          k_source.setArg(1, source);
          q.enqueueNDRangeKernel(k_source, cl::NullRange, n_global, local);

          bool proceed;
          int curr_dist = 0;
          do {
            proceed = false;
            // TODO(stupaq) is it beneficial to merge it with the kernel?
            q.enqueueWriteBuffer(proceed_cl, true, 0, sizeof(bool), &proceed);
            k_fwd.setArg(1, curr_dist++);
            q.enqueueNDRangeKernel(k_fwd, cl::NullRange, n1_global, local);
            q.enqueueReadBuffer(proceed_cl, true, 0, sizeof(bool), &proceed);
            q.finish();
          } while (proceed);

          q.enqueueNDRangeKernel(k_interm, cl::NullRange, n_global, local);

          while (curr_dist > 1) {
            k_back.setArg(1, --curr_dist);
            q.enqueueNDRangeKernel(k_back, cl::NullRange, n1_global, local);
          }

          k_sum.setArg(1, source);
          q.enqueueNDRangeKernel(k_sum, cl::NullRange, n_global, local);

          if (source % (n / 24) == 0)
            MICROPROF_INFO("PROGRESS: %d / %d\n", source, n);
        }

        std::vector<float> bc(n);
        q.enqueueReadBuffer(bc_cl, true, 0, bytes(bc), bc.data());
        q.finish();

        MICROBENCH_TIMEPOINT(fetched_results);
        MICROBENCH_REPORT(starting_kernels, fetched_results, stderr, "%ld\n",
            std::chrono::milliseconds);
        MICROBENCH_REPORT(moving_data, fetched_results, stderr, "%ld\n",
            std::chrono::milliseconds);

        CONT_BIND(ord, bc);
      }

    template<typename Return, typename Reordering>
      inline Return cont(Context&, Reordering& ord, VertexList&,
          VertexList&, VertexList&) const {
        // TODO(stupaq) this is probably not worth looking at
        std::vector<float> bc;
        CONT_BIND(ord, bc);
      }
  };

  struct postprocess {
    template<typename Return, typename Reordering, typename Result>
      inline Return cont(Reordering& ord, Result& reordered) const {
        Return original(reordered.size());
        for (VertexId orig = 0, end = original.size(); orig < end; orig++) {
          original[orig] =
            cast<typename Return::value_type>(reordered[ord[orig]]);
        }
        return original;
      }

    template<typename Return>
      inline Return cont(Identity&, Return& reordered) const {
        return reordered;
      }

    private:
    template<typename Out, typename In>
      static inline Out cast(In v) {
        return static_cast<Out>(v);
      }
  };

}  // namespace brandes

BOOST_FUSION_ADAPT_STRUCT(brandes::Edge,
    (brandes::VertexId, v1_)
    (brandes::VertexId, v2_))

#endif  // BRANDES_H_
