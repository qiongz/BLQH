#include "blqh/fqhe.hpp"

#include <algorithm>
#include <cmath>

namespace blqh {
namespace {

int popcount(std::uint64_t x) { return __builtin_popcountll(x); }

std::uint64_t flavor_mask(int nphi) {
    if (nphi >= 64) return ~std::uint64_t{0};
    return (std::uint64_t{1} << nphi) - 1;
}

}  // namespace

int total_electrons(const System &sys) {
    if (!sys.conserve_flavor) {
        if (sys.nel_total < 0)
            throw std::runtime_error("nel_total required when flavor number is not fixed");
        return sys.nel_total;
    }
    int n = 0;
    for (const Flavor &f : sys.flavors) n += f.nel;
    return n;
}

Geometry make_geometry(const System &sys) {
    Geometry g;
    g.gamma = sys.gamma;
    if (sys.model == Model::Monolayer && sys.legacy_monolayer_gamma)
        g.gamma = total_electrons(sys) / 4.0;
    const double twopi = 2.0 * std::acos(-1.0);
    g.ly = std::sqrt(twopi * sys.nphi / g.gamma);
    g.lx = g.ly * g.gamma;
    return g;
}

System monolayer_system(int nphi, int nel) {
    System s;
    s.model = Model::Monolayer;
    s.nphi = nphi;
    s.flavors = {Flavor{0, 0, nel}};
    s.J = -1;
    s.K = -1;
    s.legacy_monolayer_gamma = true;
    return s;
}

System bilayer_system(int nphi, int nel, int nel_up, int J, int K, double d) {
    System s;
    s.model = Model::Bilayer;
    s.nphi = nphi;
    s.d = d;
    s.J = J;
    s.K = K;
    s.gamma = 1.0;
    s.legacy_monolayer_gamma = false;
    s.flavors = {Flavor{0, 0, nel_up}, Flavor{1, 0, nel - nel_up}};
    return s;
}

int layer_electrons(int nphi, int numer, int denom) {
    if (nphi <= 0 || denom <= 0 || numer < 0)
        throw std::runtime_error("filling needs positive nphi, denominator and a non-negative numerator");
    if ((nphi * numer) % denom != 0)
        throw std::runtime_error("nphi * numerator is not divisible by the filling denominator");
    return nphi * numer / denom;
}

namespace {

void apply_fields(System &sys, const Fields &fields) {
    sys.d = fields.distance;
    sys.delta_v = fields.bias;
    sys.delta_z = fields.zeeman;
    sys.delta_sas = fields.tunneling;
    sys.delta_sf = fields.spin_flip;
}

}  // namespace

System make_layers(const Layers &layers) {
    if (layers.n_upper < 0 || layers.n_lower < 0)
        throw std::runtime_error("layer electron numbers must be non-negative");
    const int slots = layers.spin ? 2 : 1;
    if (layers.n_upper > slots * layers.nphi || layers.n_lower > slots * layers.nphi)
        throw std::runtime_error("a layer holds at most nphi electrons per spin");

    System sys;
    sys.nphi = layers.nphi;
    sys.nLL = layers.nLL;
    sys.J = layers.J;
    sys.K = layers.K;
    sys.gamma = layers.gamma;
    sys.legacy_monolayer_gamma = false;
    sys.conserve_flavor = layers.fix_flavor;
    apply_fields(sys, layers.fields);

    if (!layers.spin) {
        sys.model = Model::Bilayer;
        sys.flavors = {Flavor{0, 0, layers.n_upper}, Flavor{1, 0, layers.n_lower}};
    } else {
        int uu = layers.upper_up;
        int ud = layers.upper_down;
        int du = layers.lower_up;
        int dd = layers.lower_down;
        if (uu < 0 && ud < 0 && du < 0 && dd < 0) {
            uu = layers.n_upper;
            ud = 0;
            du = layers.n_lower;
            dd = 0;
        } else if (uu + ud != layers.n_upper || du + dd != layers.n_lower) {
            throw std::runtime_error("spin counts must add up to the two layer numbers");
        }
        sys.model = Model::Spin;
        sys.flavors = {Flavor{0, 0, uu}, Flavor{0, 1, ud}, Flavor{1, 0, du}, Flavor{1, 1, dd}};
    }
    if (!layers.fix_flavor) {
        sys.nel_total = layers.n_upper + layers.n_lower;
        for (Flavor &f : sys.flavors) f.nel = 0;
    }
    return sys;
}

System spin_system(int nphi, int n_uu, int n_ud, int n_du, int n_dd, int J, int K, double d) {
    System s;
    s.model = Model::Spin;
    s.nphi = nphi;
    s.d = d;
    s.J = J;
    s.K = K;
    s.gamma = 1.0;
    s.legacy_monolayer_gamma = false;
    // Bit blocks: upper-up, upper-down, lower-up, lower-down.
    s.flavors = {
        Flavor{0, 0, n_uu},
        Flavor{0, 1, n_ud},
        Flavor{1, 0, n_du},
        Flavor{1, 1, n_dd},
    };
    return s;
}

int Sector::count(std::uint64_t det, int flavor) const {
    const std::uint64_t mask = flavor_mask(nphi);
    return popcount((det >> (flavor * nphi)) & mask);
}

int Sector::count_layer(std::uint64_t det, int layer) const {
    int n = 0;
    for (int f = 0; f < nflavor; ++f)
        if (flavors[f].layer == layer) n += count(det, f);
    return n;
}

int Sector::count_spin(std::uint64_t det, int spin) const {
    int n = 0;
    for (int f = 0; f < nflavor; ++f)
        if (flavors[f].spin == spin) n += count(det, f);
    return n;
}

std::uint64_t Sector::translate(std::uint64_t det, int shift, int &sign) const {
    if (shift < 0 || shift >= nphi)
        throw std::runtime_error("translation shift out of range");
    if (shift == 0) {
        sign = 1;
        return det;
    }
    const std::uint64_t mask = flavor_mask(nphi);
    const int inv = nphi - shift;
    std::uint64_t out = 0;
    int nsign = 0;
    for (int f = 0; f < nflavor; ++f) {
        const std::uint64_t block = (det >> (f * nphi)) & mask;
        const int ncross = popcount((block >> inv) & ((std::uint64_t{1} << shift) - 1));
        const int nf = popcount(block);
        nsign += ncross * (nf - ncross);
        const std::uint64_t rot = ((block << shift) | (block >> inv)) & mask;
        out |= rot << (f * nphi);
    }
    sign = (nsign % 2 == 0) ? 1 : -1;
    return out;
}

std::uint64_t Sector::inv_translate(std::uint64_t det, int shift, int &sign) const {
    if (shift == 0) {
        sign = 1;
        return det;
    }
    const std::uint64_t mask = flavor_mask(nphi);
    const int inv = nphi - shift;
    const std::uint64_t low = (std::uint64_t{1} << shift) - 1;
    std::uint64_t out = 0;
    int nsign = 0;
    for (int f = 0; f < nflavor; ++f) {
        const std::uint64_t block = (det >> (f * nphi)) & mask;
        const int ncross = popcount(block & low);
        const int nf = popcount(block);
        nsign += ncross * (nf - ncross);
        const std::uint64_t rot = ((block >> shift) | (block << inv)) & mask;
        out |= rot << (f * nphi);
    }
    sign = (nsign % 2 == 0) ? 1 : -1;
    return out;
}

namespace {

void generate_fixed(const System &sys, int flavor, int placed, int next_orb, int mom,
                     std::uint64_t bits, std::vector<std::uint64_t> &out) {
    const int nphi = sys.nphi;
    if (flavor == static_cast<int>(sys.flavors.size())) {
        if (sys.J < 0 || mom % nphi == sys.J) out.push_back(bits);
        return;
    }
    const int need = sys.flavors[flavor].nel;
    if (placed == need) {
        generate_fixed(sys, flavor + 1, 0, 0, mom, bits, out);
        return;
    }
    const int left = need - placed;
    for (int orb = next_orb; orb <= nphi - left; ++orb) {
        const std::uint64_t bit = std::uint64_t{1} << (flavor * nphi + orb);
        generate_fixed(sys, flavor, placed + 1, orb + 1, mom + orb, bits | bit, out);
    }
}

void generate_free(const System &sys, int placed, int next_bit, int mom, std::uint64_t bits,
                   std::vector<std::uint64_t> &out) {
    const int nbits = sys.nphi * static_cast<int>(sys.flavors.size());
    const int need = sys.nel_total;
    if (placed == need) {
        if (sys.J < 0 || mom % sys.nphi == sys.J) out.push_back(bits);
        return;
    }
    const int left = need - placed;
    for (int b = next_bit; b <= nbits - left; ++b) {
        generate_free(sys, placed + 1, b + 1, mom + (b % sys.nphi), bits | (std::uint64_t{1} << b), out);
    }
}

}  // namespace

Sector build_sector(const System &sys) {
    if (sys.nphi <= 0) throw std::runtime_error("nphi must be positive");
    if (sys.flavors.empty()) throw std::runtime_error("at least one flavor is required");
    const int nbits = sys.nphi * static_cast<int>(sys.flavors.size());
    if (nbits > 64) throw std::runtime_error("flavor packing exceeds 64 bits");
    if (sys.nphi > 32) throw std::runtime_error("nphi > 32 does not fit in one 64-bit rotate");

    Sector sector;
    sector.nphi = sys.nphi;
    sector.nflavor = static_cast<int>(sys.flavors.size());
    sector.flavors = sys.flavors;
    sector.J = sys.J;
    sector.K = sys.K;
    sector.conserve_flavor = sys.conserve_flavor;

    std::vector<std::uint64_t> raw;
    if (sys.conserve_flavor)
        generate_fixed(sys, 0, 0, 0, 0, 0, raw);
    else
        generate_free(sys, 0, 0, 0, 0, raw);

    if (sys.K >= 0) {
        std::vector<std::uint64_t> reps;
        reps.reserve(raw.size());
        for (std::uint64_t det : raw) {
            // One representative per magnetic-translation orbit: the smallest bitstring.
            // The spectrum does not depend on this gauge.
            std::uint64_t best = det;
            for (int k = 1; k < sys.nphi; ++k) {
                int sign = 1;
                const std::uint64_t img = sector.translate(det, k, sign);
                if (img < best) best = img;
            }
            reps.push_back(best);
        }
        std::sort(reps.begin(), reps.end());
        reps.erase(std::unique(reps.begin(), reps.end()), reps.end());
        sector.id = std::move(reps);
    } else {
        std::sort(raw.begin(), raw.end());
        sector.id = std::move(raw);
    }

    sector.nbasis = static_cast<int>(sector.id.size());
    sector.period.assign(sector.nbasis, sys.nphi);
    for (int i = 0; i < sector.nbasis; ++i) {
        sector.index.emplace(sector.id[i], i);
        if (sys.K < 0) continue;
        for (int k = 1; k < sys.nphi; ++k) {
            int sign = 1;
            if (sector.translate(sector.id[i], k, sign) == sector.id[i]) {
                sector.period[i] = k;
                break;
            }
        }
    }
    return sector;
}

}  // namespace blqh
