#include "blqh/fqhe.hpp"

#include <algorithm>
#include <random>

#define MKL_Complex16 std::complex<double>
#include <mkl.h>

namespace blqh {
namespace {

std::vector<std::complex<double>> matvec(const Csr &H, const std::vector<std::complex<double>> &v) {
    std::vector<std::complex<double>> y(H.n, 0.0);
    for (int i = 0; i < H.n; ++i) {
        std::complex<double> s = 0.0;
        for (long p = H.row_ptr[i]; p < H.row_ptr[i + 1]; ++p)
            s += H.val[p] * v[H.col[p]];
        y[i] = s;
    }
    return y;
}

double norm(const std::vector<std::complex<double>> &v) {
    double s = 0.0;
    for (const auto &x : v) s += std::norm(x);
    return std::sqrt(s);
}

}  // namespace

Spectrum diagonalize(const Csr &H, bool vectors) {
    const int n = H.n;
    if (n == 0) throw std::runtime_error("empty Hamiltonian");
    std::vector<std::complex<double>> A(static_cast<size_t>(n) * n, 0.0);
    for (int i = 0; i < n; ++i) {
        for (long p = H.row_ptr[i]; p < H.row_ptr[i + 1]; ++p)
            A[static_cast<size_t>(i) + static_cast<size_t>(H.col[p]) * n] += H.val[p];
    }
    Spectrum spec;
    spec.eval.assign(n, 0.0);
    const lapack_int info =
        LAPACKE_zheevd(LAPACK_COL_MAJOR, vectors ? 'V' : 'N', 'U', n, A.data(), n, spec.eval.data());
    if (info != 0) throw std::runtime_error("LAPACKE_zheevd failed, info=" + std::to_string(info));
    if (vectors) spec.evec = std::move(A);
    return spec;
}

double lanczos_lowest(const Csr &H, int steps, unsigned seed) {
    return lowest_state(H, false, steps, seed).energy;
}

GroundState lowest_state(const Csr &H, bool dense, int steps, unsigned seed) {
    const int n = H.n;
    if (n == 0) throw std::runtime_error("empty Hamiltonian");
    if (dense) {
        const Spectrum spec = diagonalize(H, true);
        GroundState g;
        g.energy = spec.eval[0];
        g.dense = true;
        const int keep = std::min(n, 6);
        g.lowest.assign(spec.eval.begin(), spec.eval.begin() + keep);
        g.psi.resize(n);
        for (int i = 0; i < n; ++i) g.psi[i] = spec.evec[static_cast<size_t>(i)];
        return g;
    }
    if (steps > n) steps = n;
    if (steps < 1) throw std::runtime_error("Lanczos steps must be positive");

    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> dist(-0.5, 0.5);
    std::vector<std::vector<std::complex<double>>> basis(steps);
    basis[0].resize(n);
    for (int i = 0; i < n; ++i) basis[0][i] = {dist(rng), dist(rng)};
    const double n0 = norm(basis[0]);
    for (auto &x : basis[0]) x /= n0;

    for (int j = 0; j < steps - 1; ++j) {
        auto w = matvec(H, basis[j]);
        for (int pass = 0; pass < 2; ++pass) {
            for (int i = 0; i <= j; ++i) {
                std::complex<double> dot = 0.0;
                for (int k = 0; k < n; ++k) dot += std::conj(basis[i][k]) * w[k];
                for (int k = 0; k < n; ++k) w[k] -= dot * basis[i][k];
            }
        }
        const double nw = norm(w);
        if (nw < 1e-14) {
            steps = j + 1;
            break;
        }
        basis[j + 1].resize(n);
        for (int k = 0; k < n; ++k) basis[j + 1][k] = w[k] / nw;
    }

    std::vector<std::complex<double>> K(static_cast<size_t>(steps) * steps, 0.0);
    for (int j = 0; j < steps; ++j) {
        const auto Hv = matvec(H, basis[j]);
        for (int i = 0; i < steps; ++i) {
            std::complex<double> dot = 0.0;
            for (int k = 0; k < n; ++k) dot += std::conj(basis[i][k]) * Hv[k];
            K[static_cast<size_t>(i) + static_cast<size_t>(j) * steps] = dot;
        }
    }
    std::vector<double> eval(steps, 0.0);
    const lapack_int info =
        LAPACKE_zheevd(LAPACK_COL_MAJOR, 'V', 'U', steps, K.data(), steps, eval.data());
    if (info != 0) throw std::runtime_error("Lanczos zheevd failed");
    GroundState g;
    g.energy = eval[0];
    g.dense = false;
    g.lowest.push_back(eval[0]);
    g.psi.assign(n, 0.0);
    for (int j = 0; j < steps; ++j) {
        const std::complex<double> weight = K[static_cast<size_t>(j)];
        for (int k = 0; k < n; ++k) g.psi[k] += weight * basis[j][k];
    }
    const double npsi = norm(g.psi);
    if (npsi > 0) {
        for (auto &x : g.psi) x /= npsi;
    }
    return g;
}

}  // namespace blqh
