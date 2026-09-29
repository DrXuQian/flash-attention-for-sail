#include "tile_size.h"
#include <iostream>

int main() {
    for (int d : {64, 96, 112, 128, 192, 256})
    for (int dv : {64, 128, 256, 512})
    for (int bytes : {1, 2})
    for (int flags = 0; flags < 32; ++flags) {
        auto tile = tile_size_fwd_sm90(d, dv, flags & 1, flags & 2, bytes,
                                      flags & 4, flags & 8, flags & 16);
        std::cout << d << ' ' << dv << ' ' << bytes << ' ' << flags << ' '
                  << std::get<0>(tile) << ' ' << std::get<1>(tile) << ' '
                  << std::get<2>(tile) << ' ' << std::get<3>(tile) << '\n';
    }
}
