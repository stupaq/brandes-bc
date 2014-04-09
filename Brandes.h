/** @author Mateusz Machalica */
#ifndef BRANDES_H_
#define BRANDES_H_

#include "./BrandesBetweenness.h"

// FIXME(stupaq) this is crap, remove it!
namespace brandes {

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

    template<typename Out, typename In>
      static inline Out cast(In v) {
        return static_cast<Out>(v);
      }
  };

}  // namespace brandes

#endif  // BRANDES_H_
