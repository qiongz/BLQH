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

std::vector<std::complex<double>> lanczos_start(int n, unsigned seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> dist(-0.5, 0.5);
    std::vector<std::complex<double>> q(n);
    for (int i = 0; i < n; ++i) q[i] = {dist(rng), dist(rng)};
    const double n0 = norm(q);
    if (n0 == 0.0) throw std::runtime_error("Lanczos start vector is zero");
    for (auto &x : q) x /= n0;
    return q;
}

const char *lanczos_kernel_name(LanczosKernel kernel) {
    switch (kernel) {
    case LanczosKernel::Scalar: return "scalar";
    case LanczosKernel::Mkl: return "mkl";
    case LanczosKernel::Gluon: return "gluon";
    }
    return "scalar";
}

LanczosKernel lanczos_kernel_from_name(const std::string &name) {
    if (name == "scalar") return LanczosKernel::Scalar;
    if (name == "mkl") return LanczosKernel::Mkl;
    if (name == "gluon") return LanczosKernel::Gluon;
    throw std::runtime_error("lanczos kernel must be scalar, mkl, or gluon, got " + name);
}

GroundState lanczos_mkl(const Csr &H, int steps, unsigned seed, int threads);
GroundState lanczos_gluon(const Csr &H, int steps, unsigned seed);

double lanczos_lowest(const Csr &H, int steps, unsigned seed) {
    return lowest_state(H, false, steps, seed).energy;
}

GroundState lanczos_ground(const Csr &H, int steps, unsigned seed, LanczosKernel kernel, int threads) {
    if (kernel == LanczosKernel::Mkl) return lanczos_mkl(H, steps, seed, threads);
    if (kernel == LanczosKernel::Gluon) return lanczos_gluon(H, steps, seed);
    return lowest_state(H, false, steps, seed, LanczosKernel::Scalar, threads);
}

GroundState lowest_state(const Csr &H, bool dense, int steps, unsigned seed, LanczosKernel kernel, int threads) {
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
    if (kernel != LanczosKernel::Scalar) return lanczos_ground(H, steps, seed, kernel, threads);
    if (steps > n) steps = n;
    if (steps < 1) throw std::runtime_error("Lanczos steps must be positive");

    std::vector<std::vector<std::complex<double>>> basis(steps);
    basis[0] = lanczos_start(n, seed);

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
    g.lowest.assign(eval.begin(), eval.end());
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
