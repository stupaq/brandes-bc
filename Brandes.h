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

#define CONT_BIND(...) return Cont().template cont<Return>(__VA_ARGS__);

namespace brandes {

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
      MICROBENCH_START(reading_graph);
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
      MICROBENCH_END(reading_graph);
      CONT_BIND(ctx, n, E);
    }

  template<typename Cont> struct csr_create {
    template<typename Return>
      inline Return cont(Context& ctx, const VertexId n, EdgeList& E) const {
        MICROBENCH_START(adjacency);
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
        MICROBENCH_END(adjacency);
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
        MICROBENCH_START(cc_ordering);
        const VertexId n = ptr.size() - 1;
        VertexList bfsno(n, -1);
        VertexList queue(n);
        VertexList ccs = { 0 };
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
        MICROBENCH_END(cc_ordering);
        CONT_BIND(ctx, ord, queue, ptr, adj, ccs);
      }
  };

  template<typename Cont> struct ocsr_pass {
    template<typename Return>
      inline Return cont(Context& ctx, VertexList& ptr, VertexList& adj)
      const {
        MICROBENCH_START(cc_ordering);
        const VertexId n = ptr.size() - 1;
        VertexList ccs = { 0, n };
        MICROBENCH_END(cc_ordering);
        CONT_BIND(ctx, ptr, adj, ccs);
      }
  };

  template<int kMDeg, typename Cont> struct vcsr_create {
    template<typename Return, typename Reordering>
      inline Return cont(Context& ctx, Reordering& ord, const VertexList&
          queue, const VertexList& ptr, const VertexList& adj, const
          VertexList& ccs) const
      {
        MICROBENCH_START(virtualization);
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
        for (VertexId virt = 0,  end = vlst.size(); virt < end; virt++) {
          if (vlst[virt].map_ == *itccs) {
            vccs.push_back(virt);
            itccs++;
          }
        }
        MICROBENCH_WARN(kN1Estimate < vlst.capacity(),
            "vlst estimate too small");
#ifndef NDEBUG
        assertions(queue, ptr, adj, ccs, vlst, oadj, vccs);
#endif  // NDEBUG
        MICROBENCH_END(virtualization);
        CONT_BIND(ctx, ord, vlst, oadj, vccs);
      }

    template<typename Return>
      inline Return cont(Context& ctx, const VertexList& ptr, VertexList& adj,
          const VertexList& ccs) const {
        MICROBENCH_START(virtualization);
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
        MICROBENCH_WARN(kN1Estimate < vlst.capacity(),
            "vlst estimate too small");
        Identity id;
#ifndef NDEBUG
        assertions(id, ptr, adj, ccs, vlst, adj, vccs);
#endif  // NDEBUG
        MICROBENCH_END(virtualization);
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
        MICROBENCH_START(virtualization);
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
        MICROBENCH_END(virtualization);
        CONT_BIND(ctx, ord, optr, oadj, ccs);
      }

    template<typename Return>
      inline Return cont(Context& ctx, VertexList& ptr, VertexList& adj,
          VertexList& ccs) const {
        MICROBENCH_START(virtualization);
        Identity id;
        MICROBENCH_END(virtualization);
        CONT_BIND(ctx, id, ptr, adj, ccs);
      }
  };

  template<typename Cont> struct betweenness {
    template<typename Return, typename Reordering>
      inline Return cont(Context& ctx, Reordering& ord, VirtualList& vlst,
          VertexList& adj, VertexList&) const {
        // TODO(stupaq) making use of connected components information
        // might require aligning component's boundaries with warp size
        // otherwise we might get bank conflicts
        MICROBENCH_START(device_wait);
        Accelerator acc = ctx.get();
        cl::Context& dev = acc.context_;
        cl::CommandQueue& q = acc.queue_;
        cl::Program& prog = acc.program_;
        // TODO(stupaq) move once you determine that it takes no time
        cl::Kernel forward(prog, "virtual_forward");
        MICROBENCH_END(device_wait);

        cl::Buffer current_waveCl(dev, CL_MEM_READ_ONLY, sizeof(int));
        cl::Buffer proceedCl(dev, CL_MEM_READ_ONLY, sizeof(int));
        cl::Buffer vlstCl(dev, CL_MEM_READ_ONLY, bytes(vlst));
        cl::Buffer adjCl(dev, CL_MEM_READ_ONLY, bytes(adj));

        // TODO(stupaq) short test
        cl::Kernel kernel(acc.program_, "square");

        const int count = 1024 * 1024;
        float* data = new float[count];
        float* results = new float[count];

        for (int i = 0; i < count; i++)
          data[i] = rand() / static_cast<float>(RAND_MAX);

        cl::Buffer input = cl::Buffer(dev, CL_MEM_READ_ONLY, count *
            sizeof(int));
        cl::Buffer output = cl::Buffer(dev, CL_MEM_WRITE_ONLY, count
            * sizeof(int));

        q.enqueueWriteBuffer(input, CL_TRUE, 0, count * sizeof(int),
            data);

        kernel.setArg(0, input);
        kernel.setArg(1, output);
        kernel.setArg(2, count);

        cl::NDRange global(count);
        cl::NDRange local(1);
        q.enqueueNDRangeKernel(kernel, cl::NullRange, global,
            local);

        q.enqueueReadBuffer(output, CL_TRUE, 0, count * sizeof(int),
            results);
        q.finish();

        int correct = 0;
        for (int i = 0; i < count; i++) {
          if (results[i] == data[i] * data[i])
            correct++;
        }
        printf("correct: %d / %d\n", correct, count);

        std::vector<float> res;

        CONT_BIND(ord, res);
      }

    template<typename Return, typename Reordering>
      inline Return cont(Context&, Reordering& ord, VertexList&, VertexList&,
          VertexList&) const {
        // TODO(stupaq) this is probably not worth looking at
        std::vector<float> res;
        CONT_BIND(ord, res);
      }

    private:
    template<typename List>
      static inline size_t bytes(List& lst) {
        return lst.size() * sizeof(typename List::value_type);
      }
  };

  struct postprocess {
    template<typename Return, typename Reordering, typename Result>
      inline Return cont(Reordering& ord, Result& reordered) const {
        Return original(reordered.size());
        for (VertexId orig = 0, end = original.size(); orig < end; orig++) {
          original[orig] = cast<typename Return::value_type>(reordered[ord[orig]]);
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
