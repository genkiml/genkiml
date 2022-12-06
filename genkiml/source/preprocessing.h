#pragma once

#include <cstdint>
#include <gsl/span>
#include <range/v3/algorithm.hpp>
#include <range/v3/view.hpp>

namespace genki::ml
{
template <typename Ty, typename Tx>
inline float lerp(Ty a, Ty b, Tx t_a, Tx t_b, Tx t)
{
    return static_cast<Ty>(a + (b - a) * static_cast<Ty>((t - t_a) / (t_b - t_a)));
}

template<typename SampleType, size_t NumSamples>
inline auto interp_1d(gsl::span<const SampleType> samples, gsl::span<const double> ts, gsl::span<const double> ts_control)
{
    [[maybe_unused]] const auto is_growing = [](const auto& v)
    {
        using namespace ranges;

        return all_of(views::zip(v, v | views::drop(1)), [](const auto& p) { return p.first < p.second; });
    };

    constexpr auto is_float_eq = [](double a, double b, double eps = 1e-5) { return std::abs(a - b) <= eps; };

    assert(is_growing(ts) && is_growing(ts_control));
    assert(samples.size() == ts.size());
    assert(ts_control.size() == ts.size());
    assert(ts_control.front() >= ts.front() || is_float_eq(ts_control.front(), ts.front()));
    assert(ts_control.back() <= ts.back() || is_float_eq(ts_control.front(), ts.front()));

    size_t                             ts_idx = 0;
    std::array<SampleType, NumSamples> ret {};

    for (const auto [i, t]: ranges::views::enumerate(ts_control))
    {
        constexpr auto is_between = [=](auto x, auto lo, auto hi)
        {
            return (x >= lo && x <= hi) || is_float_eq(x, lo) || is_float_eq(x, hi);
        };

        while (!is_between(t, ts[ts_idx], ts[ts_idx + 1]))
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
    const auto ts_highest = ts.back();
    const auto ts_aligned = ts
                            | views::transform([&](double t) { return t - ts_highest; })
                            | to<std::vector<double>>();

    std::array<std::array<SampleType, NumSamples>, NumSignals> ret {};

    for (auto [row, signal]: ranges::views::zip(ret, signals))
    {
        // TODO: Pass ranges directly to avoid dynamic allocation?
        const auto signal_v = ranges::views::all(signal) | ranges::to<std::vector>();
        row = interp_1d<float, NumSamples>(signal_v, ts_aligned, ts_control);
    }

    return ret;
}

template<size_t M, size_t N>
constexpr auto transpose(const std::array<std::array<float, M>, N>& in)
{
    std::array<std::array<float, N>, M> ret {};

    for (const auto [i, row]: ranges::views::enumerate(in))
        for (const auto [j, _]: ranges::views::enumerate(row))
            ret[j][i] = in[i][j];

    return ret;
}
} // namespace genki::ml
