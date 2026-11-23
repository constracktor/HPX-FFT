// Small utilities: FFTW plan flag parsing, vector bounds/serialization/comparison helpers
#include "../../core/include/hpxfft/util/adapter_fftw.hpp"
#include "../../core/include/hpxfft/util/blocks.hpp"
#include "../../core/include/hpxfft/util/real_length.hpp"
#include "../../core/include/hpxfft/util/vector_2d.hpp"
#include "../../core/include/hpxfft/util/vector_3d.hpp"
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>
#include <hpx/serialization/input_archive.hpp>
#include <hpx/serialization/output_archive.hpp>
#include <hpx/serialization/serialize.hpp>
#include <type_traits>
#include <utility>
#include <vector>

namespace fa = hpxfft::util::fftw_adapter;

TEST_CASE("FFTW plan flag strings", "[util][adapter]")
{
    REQUIRE(fa::string_to_fftw_plan_flag("estimate") == fa::plan_flag::estimate);
    REQUIRE(fa::string_to_fftw_plan_flag("measure") == fa::plan_flag::measure);
    REQUIRE(fa::string_to_fftw_plan_flag("patient") == fa::plan_flag::patient);
    REQUIRE(fa::string_to_fftw_plan_flag("exhaustive") == fa::plan_flag::exhaustive);
    REQUIRE_THROWS_AS(fa::string_to_fftw_plan_flag("bogus"), std::invalid_argument);
}

TEST_CASE("vector_2d checked access and comparison", "[vector_2d]")
{
    const hpxfft::util::vector_2d<double> c(2, 3, 1.0);
    REQUIRE(c.at(1, 2) == 1.0);
    REQUIRE_THROWS_AS(c.at(2, 3), std::runtime_error);
    hpxfft::util::vector_2d<double> a(2, 3, 1.0), b(3, 2, 1.0);
    REQUIRE_FALSE(a == b);
    a.at(0, 0) = 5.0;
    REQUIRE(a.at(0, 0) == 5.0);
    REQUIRE_FALSE(a == c);
}

TEST_CASE("vector_2d swap and serialization round trip", "[vector_2d]")
{
    hpxfft::util::vector_2d<double> a(2, 3, 1.5), b(1, 1, 7.0);
    swap(a, b);
    REQUIRE(a.n_row() == 1);
    REQUIRE(b.n_col() == 3);

    std::vector<char> buffer;
    {
        hpx::serialization::output_archive oa(buffer);
        oa << b;
    }
    hpxfft::util::vector_2d<double> restored;
    {
        hpx::serialization::input_archive ia(buffer);
        ia >> restored;
    }
    REQUIRE(restored == b);
}

namespace
{
// element type that counts its live instances: shows that a vector_2d destroys (frees) its elements
struct counted
{
    inline static long alive = 0;
    double v = 0.0;

    counted() { ++alive; }

    counted(const counted &o) :
        v(o.v)
    {
        ++alive;
    }

    counted &operator=(const counted &) = default;

    ~counted() { --alive; }
};

template <typename T>
bool is_empty(const hpxfft::util::vector_2d<T> &v)
{
    return v.size() == 0 && v.data() == nullptr && v.n_row() == 0 && v.n_col() == 0 && v.begin() == v.end();
}

template <typename T>
hpxfft::util::vector_2d<T> round_trip(const hpxfft::util::vector_2d<T> &src, hpxfft::util::vector_2d<T> target)
{
    std::vector<char> buffer;
    {
        hpx::serialization::output_archive oa(buffer);
        oa << src;
    }
    {
        hpx::serialization::input_archive ia(buffer);
        ia >> target;
    }
    return target;
}
}  // namespace

static_assert(std::is_nothrow_move_constructible_v<hpxfft::util::vector_2d<double>>);
static_assert(std::is_nothrow_move_assignable_v<hpxfft::util::vector_2d<double>>);
static_assert(std::is_nothrow_swappable_v<hpxfft::util::vector_2d<double>>);
static_assert(std::is_copy_assignable_v<hpxfft::util::vector_2d<double>>);

TEST_CASE("vector_2d move leaves the source empty", "[vector_2d]")
{
    hpxfft::util::vector_2d<double> a(2, 3, 1.5);
    const double *buffer = a.data();
    hpxfft::util::vector_2d<double> b(std::move(a));
    REQUIRE(is_empty(a));
    REQUIRE(b.data() == buffer);  // the buffer is stolen, not copied
    REQUIRE(b.n_row() == 2);
    REQUIRE(b.n_col() == 3);
    REQUIRE(b(1, 2) == 1.5);

    hpxfft::util::vector_2d<double> c(4, 4, 2.0);
    c = std::move(b);
    REQUIRE(is_empty(b));
    REQUIRE(c.data() == buffer);
    REQUIRE(c.size() == 6);

    // a moved-from vector can be copied, moved, assigned to and destroyed
    hpxfft::util::vector_2d<double> d(a);
    REQUIRE(is_empty(d));
    hpxfft::util::vector_2d<double> e(std::move(a));
    REQUIRE(is_empty(e));
    e = a;
    REQUIRE(is_empty(e));
    a = hpxfft::util::vector_2d<double>(1, 2, 3.0);
    REQUIRE(a.size() == 2);
    REQUIRE(a(0, 1) == 3.0);
    b = c;
    REQUIRE(b == c);
}

TEST_CASE("vector_2d copy and self assignment", "[vector_2d]")
{
    const hpxfft::util::vector_2d<double> src(2, 3, 1.0);
    hpxfft::util::vector_2d<double> a(5, 1, 4.0);
    a = src;  // copy assignment from a const vector: deep copy, src unchanged
    REQUIRE(a == src);
    REQUIRE(a.data() != src.data());
    REQUIRE(src.size() == 6);
    a(0, 0) = 9.0;
    REQUIRE(src(0, 0) == 1.0);

    // copy construction sizes from the source
    hpxfft::util::vector_2d<double> b(a);
    REQUIRE(b == a);
    REQUIRE(b.size() == 6);
    REQUIRE(b.data() != a.data());

    // self copy and self move assignment keep the contents (through references: no self-move warnings)
    hpxfft::util::vector_2d<double> &alias = a;
    const double *buffer = a.data();
    a = alias;
    REQUIRE(a.data() == buffer);
    REQUIRE(a(0, 0) == 9.0);
    a = std::move(alias);
    REQUIRE(a.data() == buffer);
    REQUIRE(a.size() == 6);
    REQUIRE(a(0, 0) == 9.0);
}

TEST_CASE("vector_2d destroys its elements", "[vector_2d]")
{
    REQUIRE(counted::alive == 0);
    {
        hpxfft::util::vector_2d<counted> a(2, 3);
        REQUIRE(counted::alive == 6);
        hpxfft::util::vector_2d<counted> b(a);
        REQUIRE(counted::alive == 12);
        hpxfft::util::vector_2d<counted> c(std::move(b));  // no new elements
        REQUIRE(counted::alive == 12);
        c = hpxfft::util::vector_2d<counted>(1, 1);  // the old buffer of c is freed
        REQUIRE(counted::alive == 7);
        a = c;  // the old buffer of a is freed
        REQUIRE(counted::alive == 2);
        c = std::move(a);
        REQUIRE(counted::alive == 1);
    }
    REQUIRE(counted::alive == 0);
}

TEST_CASE("vector_2d serialization into empty and non-empty targets", "[vector_2d]")
{
    hpxfft::util::vector_2d<double> src(3, 2);
    for (std::size_t i = 0; i < src.size(); ++i)
    {
        src.data()[i] = 0.5 * double(i);
    }
    REQUIRE(round_trip(src, hpxfft::util::vector_2d<double>()) == src);
    // the target buffer (of another size) is replaced
    hpxfft::util::vector_2d<double> restored = round_trip(src, hpxfft::util::vector_2d<double>(7, 5, 1.0));
    REQUIRE(restored == src);
    REQUIRE(restored.size() == 6);
    // an empty (e.g. moved-from) vector round-trips to an empty one
    hpxfft::util::vector_2d<double> moved(std::move(src));
    REQUIRE(is_empty(round_trip(src, hpxfft::util::vector_2d<double>(2, 2, 1.0))));
}

TEST_CASE("vector_3d checked access, rearrange and comparison", "[vector_3d]")
{
    const hpxfft::util::vector_3d<double> c(2, 3, 4, 1.0);
    REQUIRE(c.at(1, 2, 3) == 1.0);
    REQUIRE_THROWS_AS(c.at(2, 0, 0), std::runtime_error);
    hpxfft::util::vector_3d<double> a(2, 3, 4, 1.0);
    REQUIRE_THROWS_AS(a.rearrange(5, 5, 5), std::runtime_error);
    a.rearrange(4, 3, 2);
    REQUIRE_FALSE(a == c);
    hpxfft::util::vector_3d<double> d(2, 3, 4, 2.0);
    REQUIRE_FALSE(c == d);
}

TEST_CASE("block_offsets splits any n into blocks that differ by at most one", "[util][blocks]")
{
    using hpxfft::util::block_offsets;
    REQUIRE(block_offsets(12, 4) == std::vector<std::size_t>{ 0, 3, 6, 9, 12 });
    REQUIRE(block_offsets(10, 4) == std::vector<std::size_t>{ 0, 3, 6, 8, 10 });
    REQUIRE(block_offsets(3, 5) == std::vector<std::size_t>{ 0, 1, 2, 3, 3, 3 });
    REQUIRE(block_offsets(7, 1) == std::vector<std::size_t>{ 0, 7 });
    REQUIRE(block_offsets(0, 3) == std::vector<std::size_t>{ 0, 0, 0, 0 });
}

TEST_CASE("real_length derives or validates the real row length", "[util][real_length]")
{
    using hpxfft::util::real_length;
    REQUIRE(real_length(5, 0, "nz") == 8);  // even length derived from 5 complex values
    REQUIRE(real_length(5, 9, "nz") == 9);  // odd length 9 also gives 5 complex values
    REQUIRE(real_length(1, 1, "nz") == 1);
    REQUIRE_THROWS_WITH(real_length(5, 6, "nz"), Catch::Matchers::ContainsSubstring("nz does not match"));
    REQUIRE_THROWS_AS(real_length(5, 12, "ny"), std::invalid_argument);
}
