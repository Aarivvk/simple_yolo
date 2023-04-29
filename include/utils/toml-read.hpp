#ifndef A770C6FE_3631_4937_A669_8DB33D3076CF
#define A770C6FE_3631_4937_A669_8DB33D3076CF
#include <string_view>

#include "toml++/toml.h"

using namespace std::literals;

auto config = toml::parse_file("config/darknet.toml");

// get key-value pairs
auto training_config = config["training"];
auto layers_config = config["layers"];

#endif /* A770C6FE_3631_4937_A669_8DB33D3076CF */
