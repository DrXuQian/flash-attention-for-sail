#include "standalone_reference.hpp"
#include <iostream>

int main() {
    using namespace fa17_standalone;
    try {
        host_self_test();
        const Shape small{1, 5, 3, 4};
        const auto x = make_inputs(small);
        const auto again = make_inputs(small);
        if (x.q != again.q || x.k != again.k || x.v != again.v || x.q == x.k || x.k == x.v)
            throw std::runtime_error("fixture reproducibility/channel identity failed");
        auto ref = reference(small, x);
        if (ref.o.size() != small.elements() || ref.lse.size() != small.rows())
            throw std::runtime_error("reference denominator failed");
        for (double value : ref.o) if (!std::isfinite(value)) throw std::runtime_error("nonfinite reference");
        std::cout << "[FA17 standalone CPU] PASS: closed-form anchors + distinct channels + "
                     "last-output/NaN/extent negatives; CUDA=NOT_CALLED\n";
        return 0;
    } catch (std::exception const& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
