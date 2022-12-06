#pragma once

#include <genkiml.h>

#include <deque>
#include <optional>
#include <utility>

#include <range/v3/algorithm.hpp>
#include <range/v3/numeric.hpp>
#include <range/v3/view.hpp>
#include <fmt/ranges.h>

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

// TODO: Generic type?
inline float lerp(float a, float b, double t_a, double t_b, double t)
{
    return static_cast<float>(a + (b - a) * static_cast<float>((t - t_a) / (t_b - t_a)));
}

template<typename SampleType, size_t NumSamples>
inline auto interp_1d(const std::vector<SampleType>& samples, const std::vector<double>& ts, const std::vector<double>& ts_control)
{
    [[maybe_unused]] const auto is_growing = [](const auto& v)
    {
        using namespace ranges;

        return all_of(views::zip(v, v | views::drop(1)), [](const auto& p) { return p.first < p.second; });
    };

    assert(is_growing(ts) && is_growing(ts_control));
    assert(samples.size() == ts.size());
    assert(ts_control.size() == ts.size());
    assert(ts_control.front() >= ts.front());
    assert(ts_control.back() <= ts.back());

    size_t                             ts_idx = 0;
    std::array<SampleType, NumSamples> ret {};

    for (const auto [i, t]: ranges::views::enumerate(ts_control))
    {
        while (!(t >= ts[ts_idx] && t <= ts[ts_idx + 1]))
        {
            ts_idx++;
            assert(ts_idx <= ts.size() - 1);
        }

        const auto sample_and_ts = [&](size_t idx) { return std::make_pair(samples[idx], ts[idx]); };
        const auto [a, t_a] = sample_and_ts(ts_idx);
        const auto [b, t_b] = sample_and_ts(ts_idx + 1);

        ret[i] = lerp(a, b, t_a, t_b, t);
    }

    return ret;
}

template<typename SampleType, size_t NumSamples, size_t NumSignals>
auto interpolate(const std::array<std::deque<SampleType>, NumSignals>& signals, const std::deque<double>& ts, const std::vector<double>& ts_control)
{
    using namespace ranges;

    assert(ranges::all_of(signals, [&](const auto& signal) { return signal.size() == ts.size(); }));
    assert(ts.size() == ts_control.size());

    // Align the latest sample ts with zero
    const auto ts_aligned = ts
                            | views::transform([&](double t) { return t - ts.back(); })
                            | to<std::vector<double>>();

    std::array<std::array<SampleType, NumSamples>, NumSignals> ret {};

    for (auto [row, signal]: ranges::views::zip(ret, signals))
    {
        const auto signal_v = ranges::views::all(signal) | ranges::to<std::vector>();
        row = interp_1d<float, NumSamples>(signal_v, ts_aligned, ts_control);
    }

    return ret;
}

//======================================================================================================================
template<size_t WindowSize, size_t NumSignals, typename SampleType>
struct WindowedModel
{
    using Sample = std::array<SampleType, NumSignals>;
    using Signals = std::array<std::deque<SampleType>, NumSignals>;

    explicit WindowedModel(std::unique_ptr<Model>&& mdl, double inference_interval)
            : model(std::move(mdl)),
              period(inference_interval),
              ts_control(linspace(period * static_cast<double>(1 - static_cast<int>(WindowSize)), 0.0, WindowSize) | ranges::to<std::vector>())
    {
        fmt::print("ts_control: {}\n", ts_control);
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

        if (get_num_samples() == WindowSize && timestamps.back() - prev_inference_ts >= period)
        {
            prev_inference_ts = timestamps.back();

            const auto interp = interpolate<SampleType, WindowSize, NumSignals>(signals, timestamps, ts_control);

            // Not really necessary since interpolate returns a 2d std::array, but if that changes...
            std::array<float, WindowSize * NumSignals> buf {};
            std::memcpy(buf.data(), interp.data(), buf.size() * sizeof(float));

            const auto end = ranges::accumulate(signals, buf.begin(), [](auto it, const auto& signal)
            {
                return std::copy(signal.begin(), signal.end(), it);
            });

            assert(end == buf.end());

            for (auto ch: ranges::views::chunk(buf, 4))
                fmt::print("{}\n", ch);

            fmt::print("\n");

            return model->infer({buf});
        }

        return {};
    }

    //==================================================================================================================
    std::unique_ptr<Model> model;

    double period;
    double prev_inference_ts {};

    // TODO: Deque with fixed length
    Signals                   signals;
    std::deque<double>        timestamps;
    const std::vector<double> ts_control;
};

} // namespace genki::ml
