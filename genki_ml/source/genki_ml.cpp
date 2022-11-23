#include <gsl/span>
#include <fmt/ranges.h>
#include <onnxruntime_cxx_api.h>
#include <range/v3/view.hpp>
#include <range/v3/algorithm.hpp>
#include <range/v3/numeric.hpp>
#include <string>

#include "cmrc/cmrc.hpp"
#include "genki_ml.h"

CMRC_DECLARE(files);

namespace genki::ml {

static void print_model_info(const Ort::Session& session)
{
    constexpr auto print_tensor_info = [](Ort::ConstTensorTypeAndShapeInfo ti)
    {
        constexpr std::array type_names = {
                "UNDEFINED", "FLOAT", "UINT8", "INT8", "UINT16", "INT16",
                "INT32", "INT64", "STRING", "BOOL", "FLOAT16", "DOUBLE",
                "UINT32", "UINT64", "COMPLEX64", "COMPLEX128", "BFLOAT16",
        };

        fmt::print("  Dimensions: {}\n", ti.GetDimensionsCount());
//        fmt::print("  Element count: {}\n", ti.GetElementCount());
        fmt::print("  Element type: {}\n", type_names[ti.GetElementType()]);
        fmt::print("  Shape: {}\n", ti.GetShape());
    };

    fmt::print("Input count: {}\n", session.GetInputCount());
    for (size_t i = 0; i < session.GetInputCount(); ++i)
    {
        const auto name = session.GetInputNameAllocated(i, Ort::AllocatorWithDefaultOptions());

        fmt::print("Input name: {}\n", std::string_view(name.get()));
        print_tensor_info(session.GetInputTypeInfo(i).GetTensorTypeAndShapeInfo());
    }

    fmt::print("\n");
    fmt::print("Output count: {}\n", session.GetOutputCount());
    for (size_t i = 0; i < session.GetOutputCount(); ++i)
    {
        const auto name = session.GetOutputNameAllocated(i, Ort::AllocatorWithDefaultOptions());

        fmt::print("Output name: {}\n", std::string_view(name.get()));
        print_tensor_info(session.GetOutputTypeInfo(i).GetTensorTypeAndShapeInfo());
    }

    fmt::print("\n");
}

static auto get_default_session_options() noexcept
{
    auto opts = Ort::SessionOptions();

    opts.SetInterOpNumThreads(1);
    opts.SetIntraOpNumThreads(1);

    return opts;
}

static auto get_input_names(const Ort::Session& session)
{
    return ranges::views::ints(size_t {}, session.GetInputCount())
           | ranges::views::transform([&](size_t i) { return std::string(session.GetInputNameAllocated(i, Ort::AllocatorWithDefaultOptions()).get()); })
           | ranges::to<std::vector>();
}

static auto get_output_names(const Ort::Session& session)
{
    return ranges::views::ints(size_t {}, session.GetOutputCount())
           | ranges::views::transform([&](size_t i) { return std::string(session.GetOutputNameAllocated(i, Ort::AllocatorWithDefaultOptions()).get()); })
           | ranges::to<std::vector>();
}

template<typename Rng>
constexpr bool has_dynamic_shape(Rng&& rng) { return ranges::any_of(rng, [](int64_t dim) { return dim == -1; }); }

static auto get_input_tensor_shape_func(const Ort::Session& session)
{
    return [&](size_t i)
    {
        auto shp = session.GetInputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape();

        if (has_dynamic_shape(shp))
            fmt::print("Warning: Input tensor has dynamic shape: {}\n", shp);

        return shp
               | ranges::views::transform([](const auto& d) { return std::abs(d); })
               | ranges::to<std::vector>();
    };
}

static auto get_output_tensor_shape_func(const Ort::Session& session)
{
    return [&](size_t i)
    {
        auto shp = session.GetOutputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape();

        if (has_dynamic_shape(shp))
            fmt::print("Warning: Output tensor has dynamic shape: {}\n", shp);

        return shp
               | ranges::views::transform([](const auto& d) { return std::abs(d); })
               | ranges::to<std::vector>();
    };
}

static auto get_output_tensor_buffer_func(const std::vector<std::vector<int64_t>>& shapes)
{
    return [=](size_t i)
    {
        const auto element_count = ranges::accumulate(shapes[i], size_t {1}, [](size_t acc, int64_t d) { return acc * d; });

        return std::vector<float>(element_count);
    };
}

static auto cstrs(gsl::span<const std::string> strs)
{
    return strs
           | ranges::views::transform([](const std::string& s) { return s.c_str(); })
           | ranges::to<std::vector>();
}

template<typename F>
static auto get_tensor_info(size_t count, F&& func)
{
    return ranges::views::ints(size_t {}, count)
           | ranges::views::transform(func)
           | ranges::to<std::vector>();
}

//======================================================================================================================
struct Model::Impl
{
    explicit Impl(gsl::span<const gsl::byte> model_data)
            : env(ORT_LOGGING_LEVEL_WARNING, "com.example.test"),
              session(env, model_data.data(), model_data.size(), get_default_session_options()),
              input_shapes(get_tensor_info(session.GetInputCount(), get_input_tensor_shape_func(session))),
              output_shapes(get_tensor_info(session.GetOutputCount(), get_output_tensor_shape_func(session))),
              input_names(get_input_names(session)),
              output_names(get_output_names(session)),
              input_name_cstrs(cstrs(input_names)),
              output_name_cstrs(cstrs(output_names)),
              output_buffers(get_tensor_info(session.GetOutputCount(), get_output_tensor_buffer_func(output_shapes)))
    {
        print_model_info(session);
    }

    auto infer(const BufferViews& input_buffers) -> BufferViews
    {
        using namespace ranges;

        const auto create_tensors = [&](auto&& tns, auto&& shps)
        {
            const auto create_ort_tensor = [](gsl::span<const float> buf, gsl::span<const int64_t> shape)
            {
                return Ort::Value::CreateTensor<float>(
                        Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU),
                        const_cast<float*>(buf.data()), buf.size(), shape.data(), shape.size()
                );
            };

            return views::zip(tns, shps)
                   | views::transform([&](auto&& t) { return create_ort_tensor(t.first, t.second); })
                   | to<std::vector>();
        };

        const auto ort_inp_tensors  = create_tensors(input_buffers, input_shapes);
        auto       ort_outp_tensors = create_tensors(output_buffers, output_shapes);

        try
        {
            session.Run(Ort::RunOptions {nullptr},
                    input_name_cstrs.data(), ort_inp_tensors.data(), ort_inp_tensors.size(),
                    output_name_cstrs.data(), ort_outp_tensors.data(), ort_outp_tensors.size());
        }
        catch (const Ort::Exception& e)
        {
            fmt::print("Error running inference: {}\n", e.what());
        }

        return output_buffers
               | views::transform([](const auto& b) { return gsl::span<const float>(b); })
               | to<std::vector>();
    }

private:
    Ort::Env     env;
    Ort::Session session;

    const std::vector<std::string>          input_names, output_names;
    const std::vector<std::vector<int64_t>> input_shapes, output_shapes;

    const std::vector<const char*> input_name_cstrs, output_name_cstrs;

    std::vector<std::vector<float>> output_buffers;
};

//======================================================================================================================
Model::Model(gsl::span<const gsl::byte> model_data) : impl(std::make_unique<Impl>(model_data)) {}

auto Model::infer(const BufferViews& input) -> BufferViews { return impl->infer(input); }

Model::~Model() = default;

//======================================================================================================================
std::unique_ptr<Model> load_model()
{
    auto fs = cmrc::files::get_filesystem();

    constexpr auto model_filepath = "model.onnx";
    assert(fs.is_file(model_filepath));

    const auto model_file = fs.open(model_filepath);
    const auto bytes      = gsl::as_bytes(gsl::span(model_file.cbegin(), model_file.cend()));

    return std::make_unique<Model>(bytes);
}

} // namespace genki::ml
