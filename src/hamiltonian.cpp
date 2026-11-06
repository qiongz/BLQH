#include "blqh/fqhe.hpp"

#include <algorithm>
#include <cmath>
#include <map>
#include <omp.h>

namespace blqh {
namespace {

constexpr double pi() { return std::acos(-1.0); }

double form_factor(int nLL, double q) {
    // (1 - exp(-q^2/2))^{2 nLL}. nLL = 0 is 1.
    if (nLL == 0) return 1.0;
    return std::pow(1.0 - std::exp(-0.5 * q * q), 2 * nLL);
}

double coulomb_q(double lx, double ly, int nLL, double d_factor, int qx, int qy) {
    const double q = std::hypot(qx / lx, qy / ly) * 2.0 * pi();
    if (q == 0.0) return 0.0;
    return 2.0 * pi() / q * std::exp(-0.5 * q * q - q * d_factor) * form_factor(nLL, q);
}

// Monolayer Madelung: int_1^inf t^{-1/2} exp(-z t) dt = sqrt(pi/z) erfc(sqrt(z)).
double madelung_monolayer(double lx, double ly, int nphi) {
    double e = -2.0;
    for (int i = 0; i < nphi; ++i) {
        for (int j = 0; j < nphi; ++j) {
            if (i == 0 && j == 0) continue;
            const double z = (i * i * lx / ly + j * j * ly / lx) * pi();
            e += std::sqrt(pi() / z) * std::erfc(std::sqrt(z));
        }
    }
    return e / std::sqrt(lx * ly);
}

// Bilayer / spin code replaced the quadrature by a bare erfc.
double madelung_bilayer(double lx, double ly, int nphi) {
    double e = -2.0;
    for (int i = 0; i < nphi; ++i) {
        for (int j = 0; j < nphi; ++j) {
            if (i == 0 && j == 0) continue;
            const double z = (i * i * lx / ly + j * j * ly / lx) * pi();
            e += std::erfc(std::sqrt(z));
        }
    }
    return e / std::sqrt(lx * ly);
}

std::vector<double> coulomb_table_monolayer(const Geometry &g, int nphi, int nLL) {
    std::vector<double> V(static_cast<size_t>(nphi) * nphi, 0.0);
    for (int s = 0; s < nphi; ++s) {
        for (int qy = 0; qy < nphi; ++qy) {
            double acc = 0.0;
            for (int qx = -nphi / 2; qx <= nphi / 2; ++qx) {
                if (qx == 0 && qy == 0) continue;
                acc += coulomb_q(g.lx, g.ly, nLL, 0.0, qx, qy) *
                       std::cos(2.0 * pi() * s * qx / nphi) / (2.0 * g.lx * g.ly);
            }
            V[static_cast<size_t>(s) * nphi + qy] = acc;
        }
    }
    return V;
}

// Two blocks: alpha = 0 same layer, alpha = 1 different layers.
// Layout matches the historical Coulomb_matrix[alpha * nphi^2 + s * nphi + qy].
std::vector<double> coulomb_table_bilayer(const Geometry &g, int nphi, int nLL, double d) {
    std::vector<double> V(2 * static_cast<size_t>(nphi) * nphi, 0.0);
    for (int alpha = 0; alpha < 2; ++alpha) {
        const double dist = (alpha == 0) ? 0.0 : d;
        for (int s = 0; s < nphi; ++s) {
            for (int qy = 0; qy < nphi; ++qy) {
                double acc = 0.0;
                if (alpha == 0) {
                    for (int qx = -nphi; qx <= nphi; ++qx) {
                        if (qx == 0 && qy == 0) continue;
                        acc += 2.0 * coulomb_q(g.lx, g.ly, nLL, dist, qx, qy) *
                               std::cos(2.0 * pi() * s * qx / nphi) / (2.0 * g.lx * g.ly);
                    }
                } else {
                    const int qmax = static_cast<int>(10.0 * nphi / (d + 0.01));
                    for (int qx = -qmax; qx < qmax; ++qx) {
                        if (qx == 0 && qy == 0) continue;
                        acc += 2.0 * coulomb_q(g.lx, g.ly, nLL, dist, qx, qy) *
                               std::cos(2.0 * pi() * s * qx / nphi) / (2.0 * g.lx * g.ly);
                    }
                }
                V[(static_cast<size_t>(alpha) * nphi + s) * nphi + qy] = acc;
            }
        }
    }
    return V;
}

int sign_occupied(std::uint64_t det, int n, int m, int nt, int mt) {
    const std::uint64_t b = det ^ ((std::uint64_t{1} << n) + (std::uint64_t{1} << m));
    int nsign = 0;
    auto between = [&](int a, int c) {
        const int lo = std::min(a, c);
        const int hi = std::max(a, c);
        for (int k = lo + 1; k < hi; ++k)
            if ((b >> k) & 1ULL) ++nsign;
    };
    between(n, nt);
    between(m, mt);
    if ((nt > mt && m > n) || (mt > nt && m < n)) ++nsign;
    return (nsign & 1) ? -1 : 1;
}

// Historical bilayer/spin sign: counts empty orbitals between the hops.
int sign_holes(std::uint64_t det, int n, int m, int nt, int mt) {
    const std::uint64_t b = det ^ ((std::uint64_t{1} << n) + (std::uint64_t{1} << m));
    int nsign = 0;
    auto between = [&](int a, int c) {
        const int lo = std::min(a, c);
        const int hi = std::max(a, c);
        for (int k = lo + 1; k < hi; ++k)
            if (((b >> k) & 1ULL) == 0) ++nsign;
    };
    between(n, nt);
    between(m, mt);
    if ((nt > mt && m > n) || (mt > nt && m < n)) ++nsign;
    return (nsign & 1) ? -1 : 1;
}

int sign_hop_holes(std::uint64_t det, int n, int nt) {
    int nsign = 0;
    for (int i = n + 1; i < nt; ++i)
        if (((det >> i) & 1ULL) == 0) ++nsign;
    return (nsign & 1) ? -1 : 1;
}

int wrap_local(int x, int nphi) {
    if (x >= nphi) return x - nphi;
    if (x < 0) return x + nphi;
    return x;
}

struct Builder {
    const System &sys;
    const Sector &sector;
    const Geometry &geom;
    double e_classical = 0;
    std::vector<double> Vmono;
    std::vector<double> Vbi;
    std::vector<std::complex<double>> FT;

    explicit Builder(const System &s, const Sector &sec, const Geometry &g) : sys(s), sector(sec), geom(g) {
        if (sys.model == Model::Monolayer) {
            e_classical = madelung_monolayer(geom.lx, geom.ly, sys.nphi);
            Vmono = coulomb_table_monolayer(geom, sys.nphi, sys.nLL);
        } else {
            e_classical = madelung_bilayer(geom.lx, geom.ly, sys.nphi);
            Vbi = coulomb_table_bilayer(geom, sys.nphi, sys.nLL, sys.d);
        }
        const int nphi = sys.nphi;
        FT.assign(static_cast<size_t>(nphi) * nphi, 0.0);
        const int K = sys.K;
        for (int ql = 0; ql < nphi; ++ql)
            for (int qr = 0; qr < nphi; ++qr) {
                const double phase = 2.0 * pi() * (ql - qr) * K / nphi;
                FT[static_cast<size_t>(ql) * nphi + qr] = {std::cos(phase), std::sin(phase)};
            }
    }

    std::complex<double> phase(int ql, int qr) const {
        if (sys.K < 0) return {1.0, 0.0};
        return FT[static_cast<size_t>(ql) * sys.nphi + qr];
    }
};

int orbit_dim(const Sector &sector, const System &sys, std::uint64_t det) {
    if (sys.K < 0) return 1;
    const auto it = sector.index.find(det);
    if (it != sector.index.end()) return sector.period[it->second];
    for (int D = 1; D < sys.nphi; ++D) {
        int unused = 1;
        if (sector.translate(det, D, unused) == det) return D;
    }
    return sys.nphi;
}

void scatter_monolayer_row(const Builder &b, std::uint64_t det, std::map<int, std::complex<double>> &acc) {
    const int nphi = b.sys.nphi;
    const Sector &sector = b.sector;
    for (int n = 0; n < nphi - 1; ++n) {
        for (int m = n + 1; m < nphi; ++m) {
            const std::uint64_t mask = (std::uint64_t{1} << n) + (std::uint64_t{1} << m);
            if ((det & mask) != mask) continue;
            const std::uint64_t rest = det ^ mask;
            for (int t = -nphi / 2; t <= nphi / 2; ++t) {
                const int nt = wrap_local(n + t, nphi);
                const int mt = wrap_local(m - t, nphi);
                const std::uint64_t mask_t = (std::uint64_t{1} << nt) + (std::uint64_t{1} << mt);
                if ((mask_t & rest) != 0) continue;
                const int s = std::abs(n - mt);
                const int sign = sign_occupied(det, n, m, nt, mt);
                const double amp = 2.0 * b.Vmono[static_cast<size_t>(s) * nphi + std::abs(t)] * sign;
                const std::uint64_t image = mask_t + rest;
                const auto it = sector.index.find(image);
                if (it != sector.index.end()) acc[it->second] += amp;
            }
        }
    }
}

void scatter_pair(const Builder &b, std::uint64_t det, int flavor_a, int flavor_b, int ql, int signl, int Dl,
                  bool same_layer, std::map<int, std::complex<double>> &acc) {
    const int nphi = b.sys.nphi;
    const int base_a = flavor_a * nphi;
    const int base_b = flavor_b * nphi;
    const bool intra = flavor_a == flavor_b;
    const int n_begin = base_a;
    const int n_end = base_a + (intra ? nphi - 1 : nphi);
    const int m_lo_offset = intra ? 1 : 0;

    for (int n = n_begin; n < n_end; ++n) {
        const int m_begin = intra ? n + m_lo_offset : base_b;
        const int m_end = base_b + nphi;
        for (int m = m_begin; m < m_end; ++m) {
            const std::uint64_t mask = (std::uint64_t{1} << n) + (std::uint64_t{1} << m);
            if ((det & mask) != mask) continue;
            const std::uint64_t rest = det ^ mask;
            const int n_local = n - base_a;
            const int m_local = m - base_b;
            for (int t = -nphi + 1; t < nphi; ++t) {
                const int nt = base_a + wrap_local(n_local + t, nphi);
                const int mt = base_b + wrap_local(m_local - t, nphi);
                if (intra && nt == mt) continue;
                const std::uint64_t mask_t = (std::uint64_t{1} << nt) + (std::uint64_t{1} << mt);
                if ((mask_t & rest) != 0) continue;
                const int s = std::abs((mt - base_b) - (n - base_a));
                const int pair = sign_holes(det, n, m, nt, mt);
                const size_t alpha = same_layer ? 0 : 1;
                const double element =
                    b.Vbi[(alpha * nphi + s) * static_cast<size_t>(nphi) + std::abs(t)];
                const std::uint64_t created = rest | mask_t;

                int Dr = orbit_dim(b.sector, b.sys, created);
                if (b.sys.K < 0) Dr = 1;
                const int qr_max = Dr;
                for (int qr = 0; qr < qr_max; ++qr) {
                    int signr = 1;
                    const std::uint64_t image =
                        (qr == 0) ? created : b.sector.inv_translate(created, qr, signr);
                    const auto it = b.sector.index.find(image);
                    if (it == b.sector.index.end()) continue;
                    const int sign = pair * signl * signr;
                    const double norm = std::sqrt(static_cast<double>(Dl) * Dr);
                    acc[it->second] += element * static_cast<double>(sign) * b.phase(ql, qr) / norm;
                }
            }
        }
    }
}

void scatter_hop(const Builder &b, std::uint64_t det, int fa, int fb, double amp, int ql, int signl, int Dl,
                 std::map<int, std::complex<double>> &acc) {
    if (amp == 0.0) return;
    const int nphi = b.sys.nphi;
    for (int o = 0; o < nphi; ++o) {
        const int n = fa * nphi + o;
        const int nt = fb * nphi + o;
        const std::uint64_t mask = (std::uint64_t{1} << n) + (std::uint64_t{1} << nt);
        const std::uint64_t kn = mask & det;
        if (kn == 0 || kn == mask) continue;
        const std::uint64_t created = (det ^ kn) | (kn ^ mask);
        const int pair = sign_hop_holes(det, std::min(n, nt), std::max(n, nt));
        int Dr = orbit_dim(b.sector, b.sys, created);
        if (b.sys.K < 0) Dr = 1;
        for (int qr = 0; qr < Dr; ++qr) {
            int signr = 1;
            const std::uint64_t image =
                (qr == 0) ? created : b.sector.inv_translate(created, qr, signr);
            const auto it = b.sector.index.find(image);
            if (it == b.sector.index.end()) continue;
            const int sign = pair * signl * signr;
            const double norm = std::sqrt(static_cast<double>(Dl) * Dr);
            acc[it->second] += amp * static_cast<double>(sign) * b.phase(ql, qr) / norm;
        }
    }
}

double diagonal_shift(const System &sys, const Sector &sector, std::uint64_t det, double e_classical) {
    const int nel = [&]() {
        int n = 0;
        for (int f = 0; f < sector.nflavor; ++f) n += sector.count(det, f);
        return n;
    }();
    double diag = e_classical * nel;
    if (sys.model == Model::Monolayer) return diag;

    const int n_upper = sector.count_layer(det, 0);
    const int n_lower = sector.count_layer(det, 1);
    diag += -sys.d * n_upper * n_lower / static_cast<double>(sys.nphi);
    // +bias/2 on the upper layer. +zeeman/2 on spin up.
    diag += 0.5 * sys.delta_v * (n_upper - n_lower);
    if (sys.model == Model::Spin) {
        const int n_up = sector.count_spin(det, 0);
        const int n_dn = sector.count_spin(det, 1);
        diag += 0.5 * sys.delta_z * (n_up - n_dn);
    }
    return diag;
}

}  // namespace

Csr build_hop(const System &sys, const Sector &sector, HopChannel channel) {
    Builder builder(sys, sector, make_geometry(sys));
    const int n = sector.nbasis;
    std::vector<std::map<int, std::complex<double>>> rows(n);
    for (int i = 0; i < n; ++i) {
        std::map<int, std::complex<double>> acc;
        const int Dl = (sys.K < 0) ? 1 : sector.period[i];
        for (int ql = 0; ql < Dl; ++ql) {
            int signl = 1;
            const std::uint64_t lbasis =
                (ql == 0) ? sector.id[i] : sector.translate(sector.id[i], ql, signl);
            const int nf = sector.nflavor;
            auto hop = [&](int a, int b) {
                if (a < nf && b < nf) scatter_hop(builder, lbasis, a, b, 1.0, ql, signl, Dl, acc);
            };
            if (channel == HopChannel::Sas) {
                if (sys.model == Model::Bilayer) hop(0, 1);
                if (sys.model == Model::Spin) {
                    hop(0, 2);
                    hop(1, 3);
                }
            } else if (channel == HopChannel::SpinFlip && sys.model == Model::Spin) {
                hop(0, 3);
                hop(1, 2);
            } else if (channel == HopChannel::UpperSpin && nf >= 2) {
                hop(0, 1);
            } else if (channel == HopChannel::LowerSpin && nf >= 4) {
                hop(2, 3);
            }
        }
        rows[i] = std::move(acc);
    }
    Csr H;
    H.n = n;
    H.row_ptr.assign(1, 0);
    for (int i = 0; i < n; ++i) {
        for (const auto &kv : rows[i]) {
            if (std::abs(kv.second) == 0.0) continue;
            H.col.push_back(kv.first);
            H.val.push_back(kv.second);
        }
        H.row_ptr.push_back(static_cast<long>(H.col.size()));
    }
    return H;
}

Csr build_hamiltonian(const System &sys, const Sector &sector, const Geometry &geom) {
    Builder builder(sys, sector, geom);
    const int n = sector.nbasis;
    std::vector<std::map<int, std::complex<double>>> rows(n);

    #pragma omp parallel for schedule(dynamic, 4)
    for (int i = 0; i < n; ++i) {
        std::map<int, std::complex<double>> acc;
        const int Dl = (sys.K < 0) ? 1 : sector.period[i];
        const int ql_max = Dl;
        for (int ql = 0; ql < ql_max; ++ql) {
            int signl = 1;
            const std::uint64_t lbasis =
                (ql == 0) ? sector.id[i] : sector.translate(sector.id[i], ql, signl);
            if (sys.model == Model::Monolayer) {
                scatter_monolayer_row(builder, lbasis, acc);
            } else {
                const int nf = sector.nflavor;
                for (int a = 0; a < nf; ++a) {
                    for (int b = a; b < nf; ++b) {
                        const bool same_layer = sector.flavors[a].layer == sector.flavors[b].layer;
                        scatter_pair(builder, lbasis, a, b, ql, signl, Dl, same_layer, acc);
                    }
                }
                if (sys.model == Model::Bilayer && nf >= 2)
                    scatter_hop(builder, lbasis, 0, 1, -0.5 * sys.delta_sas, ql, signl, Dl, acc);
                if (sys.model == Model::Spin && nf == 4) {
                    // Same spin across the layers, then the two spin-flip channels.
                    scatter_hop(builder, lbasis, 0, 2, -0.5 * sys.delta_sas, ql, signl, Dl, acc);
                    scatter_hop(builder, lbasis, 1, 3, -0.5 * sys.delta_sas, ql, signl, Dl, acc);
                    scatter_hop(builder, lbasis, 0, 3, -0.5 * sys.delta_sf, ql, signl, Dl, acc);
                    scatter_hop(builder, lbasis, 1, 2, -0.5 * sys.delta_sf, ql, signl, Dl, acc);
                }
            }
        }
        acc[i] += diagonal_shift(sys, sector, sector.id[i], builder.e_classical);
        rows[i] = std::move(acc);
    }

    Csr H;
    H.n = n;
    H.row_ptr.reserve(n + 1);
    H.row_ptr.push_back(0);
    for (int i = 0; i < n; ++i) {
        for (const auto &kv : rows[i]) {
            if (std::abs(kv.second) <= 1e-10) continue;
            H.col.push_back(kv.first);
            H.val.push_back(kv.second);
        }
        H.row_ptr.push_back(static_cast<long>(H.col.size()));
    }
    return H;
}

}  // namespace blqh
