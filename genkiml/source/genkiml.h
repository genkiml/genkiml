#pragma once

#include <gsl/span>
#include <memory>
#include <string_view>
#include <vector>

namespace genki::ml {

using BufferViews = std::vector<gsl::span<const float>>;

struct Model
{
    explicit Model(gsl::span<const gsl::byte> model_data, std::string_view log_id = "");
    ~Model();

    auto infer(const BufferViews&) -> BufferViews;

private:
    struct Impl;
    std::unique_ptr<Impl> impl;
};

std::unique_ptr<Model> load_model();

} // namespace genki::ml
