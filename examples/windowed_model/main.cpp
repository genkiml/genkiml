#include <atomic>
#include <csignal>
#include <fmt/ranges.h>
#include <thread>

#include <genkiml/windowed_model.h>
#include "glfw.h"

std::atomic_bool is_running = true;
static void signal_handler(int) { is_running = false; }

int main()
{
    namespace sc = std::chrono;

    for (auto sig: {SIGINT, SIGTERM, SIGABRT})
        signal(sig, signal_handler);

    constexpr size_t window_size        = 200;
    constexpr size_t num_signals        = 2;
    constexpr double inference_interval = 2.0;
    constexpr double sample_rate_hz     = 100.0;

    genki::ml::WindowedModel<window_size, num_signals, float> model(genki::ml::load_model(), inference_interval, sample_rate_hz);

    sc::time_point<sc::steady_clock> prev_update {};

    while (is_running)
    {
        const auto now = sc::steady_clock::now();
        const auto dt  = sc::duration_cast<sc::milliseconds>(now - prev_update);
        prev_update = now;

        const auto [norm_x, norm_y] = GLFW::normalize(GLFW::get_mouse_pos());

        const auto ts_s = static_cast<double>(sc::duration_cast<sc::milliseconds>(now.time_since_epoch()).count()) / 1e3;

        if (const auto res = model.push_sample({static_cast<float>(norm_x), static_cast<float>(norm_y)}, ts_s); res.has_value())
        {
            fmt::print("Result: {}\n", res.value());
        }

        // Just do a busy-loop. Windows' sleep/wakeup overhead is ~15ms
        while (sc::steady_clock::now() < now + sc::milliseconds(10))
        {}
    }
}
