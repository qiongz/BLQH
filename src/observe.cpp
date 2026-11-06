#include "blqh/fqhe.hpp"

#include <cmath>
#include <complex>
#include <unordered_map>
#include <vector>

namespace blqh {
namespace {

std::vector<std::complex<double>> apply(const Csr &op, const std::vector<std::complex<double>> &psi) {
    std::vector<std::complex<double>> y(op.n, 0.0);
    for (int i = 0; i < op.n; ++i) {
        for (long p = op.row_ptr[i]; p < op.row_ptr[i + 1]; ++p)
            y[i] += op.val[p] * psi[op.col[p]];
    }
    return y;
}

double abs_expectation(const Csr &op, const std::vector<std::complex<double>> &psi) {
    const auto y = apply(op, psi);
    std::complex<double> dot = 0.0;
    for (int i = 0; i < op.n; ++i) dot += std::conj(psi[i]) * y[i];
    return std::abs(dot);
}

double mean_square(const Csr &op, const std::vector<std::complex<double>> &psi) {
    const auto y = apply(op, psi);
    double s = 0.0;
    for (const auto &v : y) s += std::norm(v);
    return s;
}

int hole_parity(std::uint64_t det, int a, int b) {
    if (a > b) std::swap(a, b);
    int nsign = 0;
    for (int i = a + 1; i < b; ++i)
        if (((det >> i) & 1ULL) == 0) ++nsign;
    return nsign & 1;
}

double exciton_weight(const System &sys, const Sector &sector, const std::vector<std::complex<double>> &psi) {
    const int nphi = sector.nphi;
    std::vector<std::pair<int, int>> pairs;
    if (sys.model == Model::Bilayer) pairs.push_back({0, 1});
    if (sys.model == Model::Spin) {
        pairs.push_back({0, 2});
        pairs.push_back({1, 3});
    }
    std::unordered_map<std::uint64_t, std::complex<double>> image;
    for (int i = 0; i < sector.nbasis; ++i) {
        const std::complex<double> amp = psi[static_cast<size_t>(i)];
        if (amp == 0.0) continue;
        const std::uint64_t det = sector.id[i];
        for (int orbital = 0; orbital < nphi; ++orbital) {
            for (const auto &pair : pairs) {
                const int n = pair.first * nphi + orbital;
                const int nt = pair.second * nphi + orbital;
                const std::uint64_t mask = (std::uint64_t{1} << n) + (std::uint64_t{1} << nt);
                const std::uint64_t kn = det & mask;
                if (kn == 0 || kn == mask) continue;
                const std::uint64_t created = det ^ mask;
                const double sign = hole_parity(det, n, nt) ? -1.0 : 1.0;
                image[created] += sign * amp;
            }
        }
    }
    double weight = 0.0;
    for (const auto &kv : image) weight += std::norm(kv.second);
    return weight;
}

double staggered_square(const Csr &upper, const Csr &lower, const std::vector<std::complex<double>> &psi) {
    const auto yu = apply(upper, psi);
    const auto yl = apply(lower, psi);
    double s = 0.0;
    for (int i = 0; i < upper.n; ++i) s += std::norm(yu[i] - yl[i]);
    return s;
}

}  // namespace

std::string classify_phase(const Order &order) {
    // S / CAF / FM: Zheng, Radtke, Das Sarma, PRL 78, 2453 (1997).
    // C / I: Champagne, Eisenstein et al., PRB 78, 205310 (2008), the coherent
    // ν_T=1 exciton condensate versus incoherent layers, including 2/3 + 1/3.
    const double az = std::abs(order.sz);
    if (order.coh_sas > 0.15 && az < 0.35) return "S";
    if ((order.fluc_af > 0.08 || order.coh_sf > 0.1) && az > 0.15 && az < 0.85) return "CAF";
    // A single Slater determinant at total filling 1 has ex_sas <= 1.
    // A coherent superposition of interlayer excitons goes above that.
    if (order.ex_sas > 1.05 && order.coh_sas < 0.05) return "C";
    if (az > 0.7 && order.ex_sas < 0.15 && order.fluc_af < 0.05) return "FM";
    if (order.ex_sas <= 1.05 && order.coh_sas < 0.05 && order.fluc_af < 0.05) return "I";
    if (az > 0.7) return "FM";
    return "mixed";
}

Order measure(const System &sys, const Sector &sector, const std::vector<std::complex<double>> &psi) {
    Order order;
    if (psi.size() != static_cast<size_t>(sector.nbasis))
        throw std::runtime_error("wavefunction size does not match the sector");
    for (int i = 0; i < sector.nbasis; ++i) {
        const double w = std::norm(psi[static_cast<size_t>(i)]);
        if (sector.nflavor > 0) order.n_uu += w * sector.count(sector.id[i], 0);
        if (sector.nflavor > 1) order.n_ud += w * sector.count(sector.id[i], 1);
        if (sector.nflavor > 2) order.n_du += w * sector.count(sector.id[i], 2);
        if (sector.nflavor > 3) order.n_dd += w * sector.count(sector.id[i], 3);
    }
    double n_up_spin = order.n_uu;
    double n_dn_spin = order.n_ud;
    double n_upper = order.n_uu;
    double n_lower = 0;
    if (sys.model == Model::Bilayer) {
        n_upper = order.n_uu;
        n_lower = order.n_ud;
        n_up_spin = 0;
        n_dn_spin = 0;
        order.n_du = 0;
        order.n_dd = 0;
    } else if (sys.model == Model::Spin) {
        n_upper = order.n_uu + order.n_ud;
        n_lower = order.n_du + order.n_dd;
        n_up_spin = order.n_uu + order.n_du;
        n_dn_spin = order.n_ud + order.n_dd;
    }
    const double nel = n_upper + n_lower;
    if (nel > 0) {
        order.sz = (n_up_spin - n_dn_spin) / nel;
        order.layer = (n_upper - n_lower) / nel;
    }
    if (sys.model == Model::Bilayer || sys.model == Model::Spin) {
        const Csr tsas = build_hop(sys, sector, HopChannel::Sas);
        order.coh_sas = abs_expectation(tsas, psi) / std::max(nel, 1.0);
        order.fluc_sas = mean_square(tsas, psi) / std::max(nel * nel, 1.0);
    }
    if (sys.model == Model::Spin) {
        const Csr tsf = build_hop(sys, sector, HopChannel::SpinFlip);
        const Csr su = build_hop(sys, sector, HopChannel::UpperSpin);
        const Csr sd = build_hop(sys, sector, HopChannel::LowerSpin);
        order.coh_sf = abs_expectation(tsf, psi) / std::max(nel, 1.0);
        order.fluc_af = staggered_square(su, sd, psi) / std::max(nel * nel, 1.0);
    }
    order.ex_sas = exciton_weight(sys, sector, psi) / std::max(nel, 1.0);
    order.phase = classify_phase(order);
    return order;
}

}  // namespace blqh
