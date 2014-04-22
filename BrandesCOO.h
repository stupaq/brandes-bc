/** @author Mateusz Machalica */
#ifndef BRANDESCOO_H_
#define BRANDESCOO_H_

#include <boost/fusion/adapted.hpp>
#include <boost/spirit/include/qi.hpp>
#include <boost/iostreams/device/mapped_file.hpp>

#include <cassert>
#include <vector>

#include "./BrandesCommons.h"

namespace brandes {

  struct Edge {
    typedef int VertexId;
    VertexId v1_;
    VertexId v2_;
  };

  template<typename Cont, typename Return = std::vector<float>>
    inline Return generic_read(Context& ctx, const char* file_path) {
      const size_t kEdgesInit = 1<<20;
      using boost::iostreams::mapped_file;
      using boost::spirit::qi::phrase_parse;
      using boost::spirit::qi::int_;
      using boost::spirit::qi::eol;
      using boost::spirit::ascii::blank;
      MICROPROF_START(reading_graph);
      mapped_file mf(file_path, mapped_file::readonly);
      std::vector<Edge> E;
      E.reserve(kEdgesInit);
      {
        auto dat0 = mf.const_data(), dat1 = dat0 + mf.size();
        bool r = phrase_parse(dat0, dat1, (int_ >> int_) % eol > eol, blank, E);
        assert(r); SUPPRESS_UNUSED(r);
        assert(dat0 == dat1);
      }
      Edge::VertexId n = 0;
      for (auto& e : E) {
        n = (n <= e.v1_) ? e.v1_ + 1 : n;
        n = (n <= e.v2_) ? e.v2_ + 1 : n;
      }
      assert(!E.empty());
#ifndef NDEBUG
      for (const Edge& e : E) {
        assert(n > e.v1_ && n > e.v2_);
      }
#endif  // NDEBUG
      MICROPROF_END(reading_graph);
      return CONT_BIND(ctx, n, E);
    }

}  // namespace brandes

BOOST_FUSION_ADAPT_STRUCT(brandes::Edge,
    (brandes::Edge::VertexId, v1_)
    (brandes::Edge::VertexId, v2_))

#endif  // BRANDESCOO_H_
