#include "blqh/fqhe.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>

#include <omp.h>

namespace {

int failures = 0;

void expect(bool cond, const std::string &msg) {
    if (cond) {
        std::cout << "PASS  " << msg << "\n";
    } else {
        std::cout << "FAIL  " << msg << "\n";
        ++failures;
    }
}

int degeneracy(const std::vector<double> &eval, double tol = 1e-6) {
    int d = 0;
    for (double e : eval)
        if (std::abs(e - eval[0]) < tol) ++d;
    return d;
}

double residual(const blqh::Csr &H, const blqh::Spectrum &spec, int state) {
    const int n = H.n;
    const double energy = spec.eval[state];
    const std::complex<double> *v = spec.evec.data() + static_cast<size_t>(state) * n;
    double diff = 0.0;
    double rhs = 0.0;
    for (int i = 0; i < n; ++i) {
        std::complex<double> Hv = 0.0;
        for (long p = H.row_ptr[i]; p < H.row_ptr[i + 1]; ++p)
            Hv += H.val[p] * v[H.col[p]];
        diff += std::norm(Hv - energy * v[i]);
        rhs += std::norm(energy * v[i]);
    }
    return std::sqrt(diff / rhs);
}

void check_saved_spectrum(const blqh::Spectrum &spec, const std::string &path) {
    std::ifstream in(path);
    if (!in) {
        expect(false, "missing reference spectrum " + path);
        return;
    }
    double max_diff = 0.0;
    int n = 0;
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty() || line[0] == '#') continue;
        if (n >= static_cast<int>(spec.eval.size())) {
            max_diff = 1e9;
            break;
        }
        max_diff = std::max(max_diff, std::abs(std::stod(line) - spec.eval[n]));
        ++n;
    }
    std::cout << "  spectrum " << path << " n=" << n << " max|dE|=" << max_diff << "\n";
    expect(n == static_cast<int>(spec.eval.size()) && max_diff < 1e-8,
           "full spectrum matches " + path);
}

void check_monolayer(int nphi, int nel, double e_ref, bool vectors) {
    const blqh::System sys = blqh::monolayer_system(nphi, nel);
    const blqh::Sector sector = blqh::build_sector(sys);
    const blqh::Geometry geom = blqh::make_geometry(sys);
    const blqh::Csr H = blqh::build_hamiltonian(sys, sector, geom);
    const blqh::Spectrum spec = blqh::diagonalize(H, vectors);
    const double per = spec.eval[0] / nel;
    const int deg = degeneracy(spec.eval);
    const bool energy_ok = std::abs(per - e_ref) < 2e-6;
    const bool deg_ok = deg == 3;
    std::cout << "  nphi=" << nphi << " nel=" << nel << " dim=" << sector.nbasis
              << " E/N=" << per << " deg=" << deg << "\n";
    expect(energy_ok && deg_ok, "monolayer Laughlin nphi=" + std::to_string(nphi) +
                                    " nel=" + std::to_string(nel));
    if (vectors) {
        const double r = residual(H, spec, 0);
        std::cout << "  residual=" << r << "\n";
        expect(r < 1e-8, "monolayer residual nphi=" + std::to_string(nphi));
    }
}

void check_bilayer() {
    const blqh::System sys = blqh::bilayer_system(8, 8, 4, 4, 4, 0.0);
    const blqh::Sector sector = blqh::build_sector(sys);
    const blqh::Geometry geom = blqh::make_geometry(sys);
    const blqh::Csr H = blqh::build_hamiltonian(sys, sector, geom);
    const blqh::Spectrum spec = blqh::diagonalize(H, true);
    const double per = spec.eval[0] / 8.0;
    std::cout << "  bilayer dim=" << sector.nbasis << " E/N=" << per << "\n";
    expect(sector.nbasis == 79, "bilayer Hilbert dimension 79");
    expect(std::abs(per + 0.636707) < 2e-5, "bilayer d=0 Halperin 111 energy");
    const double r = residual(H, spec, 0);
    std::cout << "  residual=" << r << "\n";
    expect(r < 1e-8, "bilayer ground-state residual");

    const double lanc = blqh::lanczos_lowest(H, sector.nbasis, 1);
    std::cout << "  lanczos E/N=" << lanc / 8.0 << "\n";
    expect(std::abs(lanc - spec.eval[0]) < 1e-6, "Lanczos matches full ED on the same CSR");
}

void check_spin_matches_upper_layer() {
    const blqh::System two = blqh::bilayer_system(6, 2, 2, -1, -1, 0.0);
    const blqh::System four = blqh::spin_system(6, 2, 0, 0, 0, -1, -1, 0.0);
    const auto s2 = blqh::build_sector(two);
    const auto s4 = blqh::build_sector(four);
    const auto H2 = blqh::build_hamiltonian(two, s2, blqh::make_geometry(two));
    const auto H4 = blqh::build_hamiltonian(four, s4, blqh::make_geometry(four));
    const auto e2 = blqh::diagonalize(H2, false);
    const auto e4 = blqh::diagonalize(H4, false);
    double max_diff = 0.0;
    if (e2.eval.size() == e4.eval.size()) {
        for (size_t i = 0; i < e2.eval.size(); ++i)
            max_diff = std::max(max_diff, std::abs(e2.eval[i] - e4.eval[i]));
    } else {
        max_diff = 1e9;
    }
    std::cout << "  spin-reduction dim " << s2.nbasis << " vs " << s4.nbasis
              << " max|dE|=" << max_diff << "\n";
    expect(s2.nbasis == s4.nbasis && max_diff < 1e-8,
           "spinful code with one occupied flavor matches bilayer");
}

void check_bias_and_zeeman() {
    blqh::System base = blqh::bilayer_system(4, 1, 1, -1, -1, 0.0);
    blqh::System biased = base;
    biased.delta_v = 0.2;
    const auto e0 = blqh::diagonalize(blqh::build_hamiltonian(base, blqh::build_sector(base), blqh::make_geometry(base)), false);
    const auto e1 = blqh::diagonalize(
        blqh::build_hamiltonian(biased, blqh::build_sector(biased), blqh::make_geometry(biased)), false);
    double max_shift_err = 0.0;
    for (size_t i = 0; i < e0.eval.size(); ++i)
        max_shift_err = std::max(max_shift_err, std::abs((e1.eval[i] - e0.eval[i]) - 0.1));
    std::cout << "  bias shift error=" << max_shift_err << "\n";
    expect(max_shift_err < 1e-8, "bilayer bias is +1/2 Delta_V per upper electron");

    blqh::System spin = blqh::spin_system(4, 1, 0, 0, 0, -1, -1, 0.0);
    blqh::System zeeman = spin;
    zeeman.delta_z = 0.3;
    const auto z0 = blqh::diagonalize(blqh::build_hamiltonian(spin, blqh::build_sector(spin), blqh::make_geometry(spin)), false);
    const auto z1 = blqh::diagonalize(
        blqh::build_hamiltonian(zeeman, blqh::build_sector(zeeman), blqh::make_geometry(zeeman)), false);
    double zerr = 0.0;
    for (size_t i = 0; i < z0.eval.size(); ++i)
        zerr = std::max(zerr, std::abs((z1.eval[i] - z0.eval[i]) - 0.15));
    std::cout << "  zeeman shift error=" << zerr << "\n";
    expect(zerr < 1e-8, "Zeeman is +1/2 Delta_Z per spin-up electron");
}

void check_layer_parameters() {
    const int nphi = 15;
    const int n_up = blqh::layer_electrons(nphi, 1, 3);
    const int n_dn = blqh::layer_electrons(nphi, 2, 3);
    blqh::Layers layers;
    layers.nphi = nphi;
    layers.n_upper = n_up;
    layers.n_lower = n_dn;
    layers.spin = true;
    layers.J = 0;
    layers.K = 0;
    layers.fields.bias = 0.01;
    layers.fields.zeeman = 0.02;
    const blqh::System made = blqh::make_layers(layers);
    const blqh::System direct = blqh::spin_system(nphi, n_up, 0, n_dn, 0, 0, 0, 0.0);
    const auto sector = blqh::build_sector(made);
    const auto reference = blqh::build_sector(direct);
    std::cout << "  (1/3,2/3) spin-polarized dim=" << sector.nbasis << "\n";
    expect(n_up == 5 && n_dn == 10, "fillings 1/3 and 2/3 at 15 orbitals");
    expect(sector.nbasis == reference.nbasis && sector.nbasis == 40081,
           "spin-polarized 5+10 sector");

    blqh::Layers plain = layers;
    plain.spin = false;
    plain.fields = {};
    const blqh::System spinless = blqh::make_layers(plain);
    expect(spinless.model == blqh::Model::Bilayer && spinless.flavors.size() == 2,
           "spin off keeps two flavors");

    blqh::System flip;
    flip.model = blqh::Model::Spin;
    flip.nphi = 1;
    flip.gamma = 1;
    flip.legacy_monolayer_gamma = false;
    flip.conserve_flavor = false;
    flip.nel_total = 1;
    flip.flavors = {blqh::Flavor{0, 0, 0}, blqh::Flavor{0, 1, 0}, blqh::Flavor{1, 0, 0},
                    blqh::Flavor{1, 1, 0}};
    flip.delta_sf = 0.4;
    const auto spec = blqh::diagonalize(
        blqh::build_hamiltonian(flip, blqh::build_sector(flip), blqh::make_geometry(flip)), false);
    const double gap = spec.eval.back() - spec.eval.front();
    std::cout << "  spin-flip dim=" << spec.eval.size() << " gap=" << gap << "\n";
    expect(spec.eval.size() == 4, "one electron over four spin-layer orbitals");
    expect(std::abs(gap - flip.delta_sf) < 1e-8, "spin-flip tunneling splitting");
}

void check_tunneling() {
    blqh::System sys;
    sys.model = blqh::Model::Bilayer;
    sys.nphi = 1;
    sys.gamma = 1.0;
    sys.legacy_monolayer_gamma = false;
    sys.conserve_flavor = false;
    sys.nel_total = 1;
    sys.flavors = {blqh::Flavor{0, 0, 0}, blqh::Flavor{1, 0, 0}};
    sys.delta_sas = 0.4;
    sys.J = -1;
    sys.K = -1;
    const auto sector = blqh::build_sector(sys);
    const auto spec = blqh::diagonalize(blqh::build_hamiltonian(sys, sector, blqh::make_geometry(sys)), false);
    const double gap = spec.eval.back() - spec.eval.front();
    std::cout << "  tunneling dim=" << sector.nbasis << " gap=" << gap << "\n";
    expect(sector.nbasis == 2, "one electron, two layers, one orbital");
    expect(std::abs(gap - sys.delta_sas) < 1e-8, "SAS splitting equals Delta_SAS");
}

}  // namespace

namespace blqh {

int run_checks(int threads, bool full) {
    omp_set_num_threads(threads);
    failures = 0;
    std::cout << "threads=" << threads << "\n";
    {
        const blqh::System sys = blqh::monolayer_system(12, 4);
        const auto sector = blqh::build_sector(sys);
        const auto spec = blqh::diagonalize(blqh::build_hamiltonian(sys, sector, blqh::make_geometry(sys)), false);
        check_saved_spectrum(spec, std::string(BLQH_SOURCE_DIR) + "/tests/ref/mono_nphi12_nel4_evals.txt");
    }
    {
        const blqh::System sys = blqh::monolayer_system(15, 5);
        const auto sector = blqh::build_sector(sys);
        const auto spec = blqh::diagonalize(blqh::build_hamiltonian(sys, sector, blqh::make_geometry(sys)), false);
        check_saved_spectrum(spec, std::string(BLQH_SOURCE_DIR) + "/tests/ref/mono_nphi15_nel5_evals.txt");
    }
    check_monolayer(12, 4, -0.4180946966, true);
    check_monolayer(15, 5, -0.4160833802, true);
    if (full) check_monolayer(18, 6, -0.4178208128, false);
    check_bilayer();
    check_spin_matches_upper_layer();
    check_bias_and_zeeman();
    check_tunneling();
    check_layer_parameters();

    std::cout << (failures ? "FAILED " : "OK ") << failures << "\n";
    return failures ? 1 : 0;
}

}  // namespace blqh
