#include "genkiml/windowed_model.h"
#include <boost/ut.hpp>
#include <range/v3/algorithm.hpp>

namespace ut = boost::ut;

ut::suite model = []
{
    using namespace ut;

    "simple model"_test = [] {
        auto model = genki::ml::load_model();

        should("load successfully") = [&] { expect(model != nullptr); };

        should("run inference") = [&]
        {
            const std::array<float, 100> input{};
            const auto result = model->infer({input});

            expect(result.size() == 1);
            expect(result[0].size() == 2);
            expect(ranges::all_of(result[0], [](float f) { return f == 0.0f; }));
        };
    };
};
