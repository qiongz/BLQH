#include "blqh/fqhe.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

#define MKL_Complex16 std::complex<double>
#include <mkl.h>

namespace blqh {
namespace {

struct LocalMklThreads {
    explicit LocalMklThreads(int threads) { mkl_set_num_threads_local(threads > 0 ? threads : 64); }
    ~LocalMklThreads() { mkl_set_num_threads_local(0); }
};

}  // namespace

GroundState lanczos_mkl(const Csr &H, int steps, unsigned seed, int threads) {
    const int n = H.n;
    if (n == 0) throw std::runtime_error("empty Hamiltonian");
    if (steps < 1) throw std::runtime_error("Lanczos steps must be positive");
    if (steps > n) steps = n;
    if (H.row_ptr.size() != static_cast<size_t>(n) + 1)
        throw std::runtime_error("CSR row_ptr has the wrong length");
    if (H.row_ptr.back() > static_cast<long>(std::numeric_limits<MKL_INT>::max()))
        throw std::runtime_error("CSR nnz does not fit MKL LP64 indices");

    LocalMklThreads mkl_threads(threads);
    std::vector<MKL_INT> rows_start(n), rows_end(n);
    for (int i = 0; i < n; ++i) {
        rows_start[i] = static_cast<MKL_INT>(H.row_ptr[i]);
        rows_end[i] = static_cast<MKL_INT>(H.row_ptr[i + 1]);
    }
    sparse_matrix_t A = nullptr;
    sparse_status_t st = mkl_sparse_z_create_csr(
        &A, SPARSE_INDEX_BASE_ZERO, n, n, rows_start.data(), rows_end.data(),
        const_cast<MKL_INT *>(H.col.data()),
        const_cast<MKL_Complex16 *>(H.val.data()));
    if (st != SPARSE_STATUS_SUCCESS || A == nullptr)
        throw std::runtime_error("mkl_sparse_z_create_csr failed");

    matrix_descr descr{};
    descr.type = SPARSE_MATRIX_TYPE_GENERAL;
    descr.mode = SPARSE_FILL_MODE_FULL;
    descr.diag = SPARSE_DIAG_NON_UNIT;
    mkl_sparse_set_mv_hint(A, SPARSE_OPERATION_NON_TRANSPOSE, descr, steps + 2);
    st = mkl_sparse_optimize(A);
    if (st != SPARSE_STATUS_SUCCESS) {
        mkl_sparse_destroy(A);
        throw std::runtime_error("mkl_sparse_optimize failed");
    }

    const std::complex<double> one(1.0, 0.0);
    const std::complex<double> zero(0.0, 0.0);
    const auto q0 = lanczos_start(n, seed);
    std::vector<std::complex<double>> Q(static_cast<size_t>(steps) * n, 0.0);
    std::vector<std::complex<double>> K(static_cast<size_t>(steps) * steps, 0.0);
    std::copy(q0.begin(), q0.end(), Q.begin());
    std::vector<std::complex<double>> w(n), h1(steps), h2(steps);

    // Two-pass classical Gram-Schmidt. The first pass of dots is the Ritz column.
    for (int j = 0; j < steps; ++j) {
        st = mkl_sparse_z_mv(SPARSE_OPERATION_NON_TRANSPOSE, one, A, descr,
                             Q.data() + static_cast<size_t>(j) * n, zero, w.data());
        if (st != SPARSE_STATUS_SUCCESS) {
            mkl_sparse_destroy(A);
            throw std::runtime_error("mkl_sparse_z_mv failed");
        }
        const int m = j + 1;
        for (int i = 0; i < m; ++i) {
            cblas_zdotc_sub(n, Q.data() + static_cast<size_t>(i) * n, 1, w.data(), 1, &h1[i]);
            K[static_cast<size_t>(i) + static_cast<size_t>(j) * steps] = h1[i];
        }
        if (j + 1 == steps) break;
        for (int i = 0; i < m; ++i) {
            const std::complex<double> alpha = -h1[i];
            cblas_zaxpy(n, &alpha, Q.data() + static_cast<size_t>(i) * n, 1, w.data(), 1);
        }
        for (int i = 0; i < m; ++i)
            cblas_zdotc_sub(n, Q.data() + static_cast<size_t>(i) * n, 1, w.data(), 1, &h2[i]);
        for (int i = 0; i < m; ++i) {
            const std::complex<double> alpha = -h2[i];
            cblas_zaxpy(n, &alpha, Q.data() + static_cast<size_t>(i) * n, 1, w.data(), 1);
        }
        const double nw = cblas_dznrm2(n, w.data(), 1);
        if (nw < 1e-14) {
            steps = j + 1;
            break;
        }
        std::copy(w.begin(), w.end(), Q.begin() + static_cast<size_t>(j + 1) * n);
        cblas_zdscal(n, 1.0 / nw, Q.data() + static_cast<size_t>(j + 1) * n, 1);
    }
    mkl_sparse_destroy(A);

    for (int j = 0; j < steps; ++j) {
        auto &diag = K[static_cast<size_t>(j) + static_cast<size_t>(j) * steps];
        diag = std::real(diag);
    }
    std::vector<double> eval(steps, 0.0);
    const lapack_int info =
        LAPACKE_zheevd(LAPACK_COL_MAJOR, 'V', 'U', steps, K.data(), steps, eval.data());
    if (info != 0) throw std::runtime_error("MKL Lanczos zheevd failed, info=" + std::to_string(info));

    GroundState g;
    g.energy = eval[0];
    g.dense = false;
    g.lowest = std::move(eval);
    g.psi.assign(n, 0.0);
    for (int j = 0; j < steps; ++j) {
        const std::complex<double> weight = K[static_cast<size_t>(j)];
        cblas_zaxpy(n, &weight, Q.data() + static_cast<size_t>(j) * n, 1, g.psi.data(), 1);
    }
    const double npsi = cblas_dznrm2(n, g.psi.data(), 1);
    if (npsi > 0) cblas_zdscal(n, 1.0 / npsi, g.psi.data(), 1);
    return g;
}

}  // namespace blqh
