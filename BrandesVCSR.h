/** @author Mateusz Machalica */
#ifndef BRANDESVCSR_H_
#define BRANDESVCSR_H_

#include <cassert>
#include <vector>

#include "./BrandesOCSR.h"

namespace brandes {

  struct Virtual {
    VertexId ptr_;
    VertexId map_;
    Virtual(VertexId ptr, VertexId map) : ptr_(ptr), map_(map) {}
  } __attribute__((packed));
  typedef std::vector<Virtual> VirtualList;

  template<typename Cont> struct vcsr_create {
    template<typename Return, typename Reordering>
      inline Return cont(Context& ctx, Reordering& ord, const VertexList&
          queue, const VertexList& ptr, const VertexList& adj, const
          VertexList& ccs) const
      {
        MICROPROF_INFO("CONFIGURATION:\tvirtualized deg\t%d\n", ctx.kMDeg_);
        MICROPROF_START(virtualization);
        VirtualList vlst;
        VertexList vccs;
        vccs.reserve(ccs.size());
        const size_t kN1Estimate = ptr.size() + adj.size() / ctx.kMDeg_;
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
              if (i % ctx.kMDeg_ == 0) {
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
        assertions(ctx, queue, ptr, adj, ccs, vlst, oadj, vccs);
#endif  // NDEBUG
        MICROPROF_END(virtualization);
        return CONT_BIND(ctx, ord, vlst, oadj, vccs);
      }

    template<typename Return>
      inline Return cont(Context& ctx, const VertexList& ptr, VertexList& adj,
          const VertexList& ccs) const {
        MICROPROF_START(virtualization);
        const VertexId n = ptr.size() - 1;
        VirtualList vlst;
        VertexList vccs;
        vccs.reserve(ccs.size());
        const size_t kN1Estimate = ptr.size() + adj.size() / ctx.kMDeg_;
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
            first += ctx.kMDeg_;
          } while (first < last);
        }
        vccs.push_back(vlst.size());
        vlst.push_back(Virtual(ptr[n], n));
        MICROPROF_WARN(kN1Estimate < vlst.capacity(),
            "vlst estimate too small");
        Identity id;
#ifndef NDEBUG
        assertions(ctx, id, ptr, adj, ccs, vlst, adj, vccs);
#endif  // NDEBUG
        MICROPROF_END(virtualization);
        return CONT_BIND(ctx, id, vlst, adj, vccs);
      }

#ifndef NDEBUG
    template<typename Reordering>
      static inline void assertions(const Context& ctx, const Reordering& rord,
          const VertexList& ptr, const VertexList& adj, const VertexList& ccs,
          const VirtualList& vlst, const VertexList& oadj, const VertexList&
          vccs) {
        const VertexId n1 = vlst.size() - 1;
        assert(static_cast<size_t>(vlst.back().ptr_) == adj.size());
        for (VertexId virt = 0; virt < n1; virt++) {
          assert(vlst[virt + 1].map_ >= vlst[virt].map_);
          assert(vlst[virt + 1].map_ <= vlst[virt].map_ + 1);
          VertexId orig = rord[vlst[virt].map_];
          assert(vlst[virt + 1].ptr_ >= vlst[virt].ptr_);
          assert(vlst[virt + 1].ptr_ - vlst[virt].ptr_ <= ctx.kMDeg_);
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
        return CONT_BIND(ctx, ord, optr, oadj, ccs);
      }

    template<typename Return>
      inline Return cont(Context& ctx, VertexList& ptr, VertexList& adj,
          VertexList& ccs) const {
        MICROPROF_START(virtualization);
        Identity id;
        MICROPROF_END(virtualization);
        return CONT_BIND(ctx, id, ptr, adj, ccs);
      }
  };

}  // namespace brandes

#endif  // BRANDESVCSR_H_
