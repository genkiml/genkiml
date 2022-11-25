#include <fmt/format.h>
#include "cmrc/cmrc.hpp"
#include "genkiml.h"

CMRC_DECLARE(files);

namespace genki::ml {
std::unique_ptr<Model> load_model()
{
    constexpr auto prefix = "models";

    auto fs         = cmrc::files::get_filesystem();
    auto models_dir = fs.iterate_directory(prefix);
    assert(std::distance(models_dir.begin(), models_dir.end()) == 1);

    const auto model = fmt::format("{}/{}", prefix, (*models_dir.begin()).filename());

    assert(fs.is_file(model));

    const auto model_file = fs.open(model);
    const auto bytes      = gsl::as_bytes(gsl::span(model_file.cbegin(), model_file.cend()));

    return std::make_unique<Model>(bytes);
}

} // namespace genki::ml
