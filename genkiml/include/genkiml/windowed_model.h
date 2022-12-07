#pragma once

#include <deque>
#include <optional>
#include <utility>

#include "genkiml.h"
#include "preprocessing.h"

namespace genki::ml {
//======================================================================================================================
template<typename T>
auto linspace(T min, T max, size_t num) noexcept
{
    using namespace ranges::views;

    assert(num > 1);

    return ints(0, ranges::unreachable)
           | transform([=](int i) { return static_cast<T>(i) * (max - min) / static_cast<T>(num - 1) + min; })
           | take_exactly(static_cast<int>(num));
}

//======================================================================================================================
template<size_t WindowSize, size_t NumSignals, typename SampleType>
struct WindowedModel
{
    using Sample = std::array<SampleType, NumSignals>;
    using Signals = std::array<std::deque<SampleType>, NumSignals>;

    explicit WindowedModel(std::unique_ptr<Model>&& mdl, double interval, double sample_rate_hz)
            : model(std::move(mdl)),
              inference_interval(interval),
              ts_control(linspace(1.0 / sample_rate_hz * static_cast<double>(1 - static_cast<int>(WindowSize)), 0.0, WindowSize) | ranges::to<std::vector>())
    {
    }

    [[nodiscard]] size_t get_num_samples()
    {
        assert(ranges::all_of(signals, [this](const auto& s) { return s.size() == timestamps.size(); }));

        return timestamps.size();
    }

    auto push_sample(const Sample& sample, double timestamp) -> std::optional<BufferViews>
    {
        if (get_num_samples() == WindowSize)
        {
            for (auto& signal: signals)
                signal.pop_front();

            timestamps.pop_front();
        }

        for (auto [signal, signal_sample]: ranges::views::zip(signals, sample))
            signal.push_back(signal_sample);

        timestamps.push_back(timestamp);

        if (get_num_samples() == WindowSize && timestamps.back() - prev_inference_ts >= inference_interval)
        {
            prev_inference_ts = timestamps.back();

            // TODO: Figure out if transpose is necessary by looking at the model shape
            const auto interp = transpose(interpolate<SampleType, WindowSize, NumSignals>(signals, timestamps, ts_control));

            // Not really necessary since interpolate returns a 2d std::array, but this is more correct
            std::array<float, WindowSize * NumSignals> buf {};
            std::memcpy(buf.data(), interp.data(), buf.size() * sizeof(float));

            return model->infer({buf});
        }

        return {};
    }

    //==================================================================================================================
    std::unique_ptr<Model> model;

    double inference_interval;
    double prev_inference_ts {};

    // TODO: Fixed length deque for samples
    Signals                   signals;
    std::deque<double>        timestamps;
    const std::vector<double> ts_control;
};

} // namespace genki::ml
