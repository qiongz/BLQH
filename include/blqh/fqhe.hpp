#pragma once

// One Fock-space exact-diagonalization library for torus quantum Hall systems.
// Monolayer, spinless bilayer and spinful bilayer are the same pipeline:
//   flavors -> sector -> one CSR operator -> dense or Lanczos solver.
// Numerical conventions of the Coulomb sum follow the historical code that
// produced the reference energies (see hamiltonian.cpp).

#include <complex>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace blqh {

enum class Model { Monolayer, Bilayer, Spin };

struct Flavor {
    int layer = 0;
    int spin = 0;
    int nel = 0;
};

// One-body fields. Signs are the same for spinless and spinful bilayers:
//   bias   > 0 raises the upper layer by bias/2 per electron
//   zeeman > 0 raises spin up by zeeman/2 per electron
//   tunneling is spin-conserving interlayer hopping (upper σ ↔ lower σ)
//   spin_flip is spin-changing interlayer hopping (upper ↑ ↔ lower ↓ and upper ↓ ↔ lower ↑)
// A hop only appears inside a sector that contains both ends. Fixed flavor
// numbers therefore drop tunneling; fix only the total electron number to keep it.
struct Fields {
    double distance = 0.0;
    double bias = 0.0;
    double zeeman = 0.0;
    double tunneling = 0.0;
    double spin_flip = 0.0;
};

// Layer populations, in electrons. Fillings ν = p/q are converted with
// layer_electrons(nphi, p, q), which requires nphi * p divisible by q.
struct Layers {
    int nphi = 0;
    int n_upper = 0;
    int n_lower = 0;
    bool spin = false;
    // Negative: that layer is spin-up polarized. Otherwise the four counts
    // must add up to (n_upper, n_lower).
    int upper_up = -1;
    int upper_down = -1;
    int lower_up = -1;
    int lower_down = -1;
    // False keeps only n_upper + n_lower, so electrons may change layer and spin.
    bool fix_flavor = true;
    int J = -1;
    int K = -1;
    int nLL = 0;
    double gamma = 1.0;
    Fields fields;
};

struct System {
    Model model = Model::Monolayer;
    int nphi = 0;
    int nLL = 0;
    std::vector<Flavor> flavors;
    int J = -1;
    int K = -1;
    double gamma = 1.0;
    double d = 0.0;
    double delta_sas = 0.0;
    double delta_sf = 0.0;
    double delta_v = 0.0;
    double delta_z = 0.0;
    // Monolayer's original main() overwrites gamma with nel/4.
    bool legacy_monolayer_gamma = true;
    // If false, only nel_total is fixed and electrons may sit in any flavor.
    // Layer tunneling needs that sector.
    bool conserve_flavor = true;
    int nel_total = -1;
};

struct Geometry {
    double lx = 0;
    double ly = 0;
    double gamma = 1;
};

struct Sector {
    int nphi = 0;
    int nflavor = 0;
    std::vector<Flavor> flavors;
    int J = -1;
    int K = -1;
    bool conserve_flavor = true;
    std::vector<std::uint64_t> id;
    std::vector<int> period;
    std::unordered_map<std::uint64_t, int> index;
    int nbasis = 0;

    int count(std::uint64_t det, int flavor) const;
    int count_layer(std::uint64_t det, int layer) const;
    int count_spin(std::uint64_t det, int spin) const;
    // Left rotation of every flavor block. sign is the fermion sign of the wrap.
    std::uint64_t translate(std::uint64_t det, int shift, int &sign) const;
    // Right rotation. This is what the historical code calls inv_translate.
    std::uint64_t inv_translate(std::uint64_t det, int shift, int &sign) const;
};

struct Csr {
    int n = 0;
    std::vector<long> row_ptr;
    std::vector<int> col;
    std::vector<std::complex<double>> val;
};

struct Spectrum {
    std::vector<double> eval;
    // Column-major eigenvectors, length n*n, when requested.
    // Column j is state j, component i at index i + j*n.
    std::vector<std::complex<double>> evec;
};

enum class HopChannel { Sas, SpinFlip, UpperSpin, LowerSpin };

struct GroundState {
    double energy = 0;
    std::vector<double> lowest;
    std::vector<std::complex<double>> psi;
    bool dense = false;
};

struct Order {
    double sz = 0;
    double layer = 0;
    double n_uu = 0, n_ud = 0, n_du = 0, n_dd = 0;
    double coh_sas = 0;
    double coh_sf = 0;
    double fluc_sas = 0;
    double fluc_af = 0;
    // ||T_sas ψ||^2 / N, including images outside this charge sector.
    // This is the exciton weight Champagne measures through interlayer tunneling.
    double ex_sas = 0;
    std::string phase;
};

Geometry make_geometry(const System &sys);
int total_electrons(const System &sys);
Sector build_sector(const System &sys);
Csr build_hamiltonian(const System &sys, const Sector &sector, const Geometry &geom);
Csr build_hop(const System &sys, const Sector &sector, HopChannel channel);
Spectrum diagonalize(const Csr &H, bool vectors);
double lanczos_lowest(const Csr &H, int steps, unsigned seed);
GroundState lowest_state(const Csr &H, bool dense, int steps, unsigned seed);
Order measure(const System &sys, const Sector &sector, const std::vector<std::complex<double>> &psi);
std::string classify_phase(const Order &order);
int run_checks(int threads, bool full);

int layer_electrons(int nphi, int numer, int denom);
System monolayer_system(int nphi, int nel);
System bilayer_system(int nphi, int nel, int nel_up, int J, int K, double d);
System spin_system(int nphi, int n_uu, int n_ud, int n_du, int n_dd, int J, int K, double d);
System make_layers(const Layers &layers);

}  // namespace blqh
