#include "genkiml/windowed_model.h"
#include <boost/ut.hpp>

namespace ut = boost::ut;

template<typename FloatType>
constexpr bool is_float_eq(FloatType a, FloatType b, FloatType eps = FloatType(1e-5)) { return std::abs(a - b) <= eps; }

ut::suite windowed_model = []
{
    using namespace ut;
    using namespace ranges::views;

    constexpr auto test_linspace = [](auto min, auto max, auto count)
    {
        auto ls = genki::ml::linspace(min, max, count);

        expect(ls.size() == count);
        expect(ranges::all_of(zip(ls, ls | drop(1)),
                [&](const auto& p) { return is_float_eq(p.second - p.first, (max - min) / (count - 1)); }));
    };

    "linspace"_test = [&]
    {
        test_linspace(0.0, 1.0, 11);
        test_linspace(-5.0, 1.0, 13);
        test_linspace(100.0, 500.0, 1000);

        {
            constexpr auto sample_rate_hz = 100;
            constexpr auto WindowSize     = 200;
            constexpr auto min            = 1.0 / sample_rate_hz * static_cast<double>(1 - static_cast<int>(WindowSize));
            constexpr auto max            = 0.0;

            test_linspace(min, max, WindowSize);
        }

    };
};

