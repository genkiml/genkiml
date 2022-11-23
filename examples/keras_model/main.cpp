#include "genki_ml.h"
#include "fmt/ranges.h"

int main()
{
    auto model = genki::ml::load_model();

    const std::array<float, 100> input {};

    const auto ret = model->infer({input});

    fmt::print("Result: {}\n", ret[0]);
}
