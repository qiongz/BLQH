#include "blqh/fqhe.hpp"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <unistd.h>
#include <vector>

namespace blqh {
namespace {

struct TempFile {
    char path[64] = "/tmp/blqh-gl-XXXXXX";
    TempFile() {
        const int fd = mkstemp(path);
        if (fd < 0) throw std::runtime_error("cannot create a temporary file for the Gluon Lanczos");
        close(fd);
    }
    ~TempFile() { std::remove(path); }
};

void write_request(const std::string &path, const Csr &H, const std::vector<std::complex<double>> &q0, int steps) {
    std::ofstream out(path, std::ios::binary);
    if (!out) throw std::runtime_error("cannot write " + path);
    const int n = H.n;
    const std::int64_t nnz = H.row_ptr.back();
    out.write("BLQHGL01", 8);
    out.write(reinterpret_cast<const char *>(&n), sizeof(n));
    out.write(reinterpret_cast<const char *>(&steps), sizeof(steps));
    out.write(reinterpret_cast<const char *>(&nnz), sizeof(nnz));
    out.write(reinterpret_cast<const char *>(H.row_ptr.data()), static_cast<std::streamsize>((n + 1) * sizeof(long)));
    out.write(reinterpret_cast<const char *>(H.col.data()), static_cast<std::streamsize>(nnz * sizeof(int)));
    out.write(reinterpret_cast<const char *>(H.val.data()), static_cast<std::streamsize>(nnz * sizeof(std::complex<double>)));
    out.write(reinterpret_cast<const char *>(q0.data()), static_cast<std::streamsize>(n * sizeof(std::complex<double>)));
    if (!out) throw std::runtime_error("short write of the Gluon Lanczos request");
}

GroundState read_result(const std::string &path, int n) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("Gluon Lanczos wrote no result");
    char magic[8];
    in.read(magic, 8);
    if (std::string(magic, 8) != "BLQHGO01") throw std::runtime_error("bad Gluon Lanczos result");
    int rn = 0, nval = 0;
    in.read(reinterpret_cast<char *>(&rn), sizeof(rn));
    in.read(reinterpret_cast<char *>(&nval), sizeof(nval));
    if (rn != n || nval < 1) throw std::runtime_error("Gluon Lanczos result has the wrong shape");
    GroundState g;
    g.dense = false;
    double energy = 0;
    in.read(reinterpret_cast<char *>(&energy), sizeof(energy));
    g.energy = energy;
    g.lowest.resize(nval);
    in.read(reinterpret_cast<char *>(g.lowest.data()), static_cast<std::streamsize>(nval * sizeof(double)));
    g.psi.resize(n);
    in.read(reinterpret_cast<char *>(g.psi.data()), static_cast<std::streamsize>(n * sizeof(std::complex<double>)));
    if (!in) throw std::runtime_error("truncated Gluon Lanczos result");
    return g;
}

}  // namespace

GroundState lanczos_gluon(const Csr &H, int steps, unsigned seed) {
    const int n = H.n;
    if (n == 0) throw std::runtime_error("empty Hamiltonian");
    if (steps < 1) throw std::runtime_error("Lanczos steps must be positive");
    if (steps > 40) throw std::runtime_error("Gluon Lanczos supports at most 40 steps");
    if (steps > n) steps = n;
    if (sizeof(long) != sizeof(std::int64_t))
        throw std::runtime_error("Gluon Lanczos request expects 64-bit row pointers");

    const auto q0 = lanczos_start(n, seed);
    TempFile request;
    TempFile result;
    TempFile log;
    write_request(request.path, H, q0, steps);

    const char *python = std::getenv("BLQH_PYTHON");
    if (python == nullptr || python[0] == '\0') python = "python3";
    const std::string source = BLQH_SOURCE_DIR;
    std::ostringstream cmd;
    cmd << "PYTHONPATH=\"" << source << "/python${PYTHONPATH:+:$PYTHONPATH}\" " << python
        << " -m blqh.lanczos_gluon --bin " << request.path << " --result " << result.path
        << " >" << log.path << " 2>&1";
    const int rc = std::system(cmd.str().c_str());
    if (rc != 0) {
        std::ifstream err(log.path);
        std::stringstream text;
        text << err.rdbuf();
        throw std::runtime_error("Gluon Lanczos failed\n" + text.str());
    }
    return read_result(result.path, n);
}

}  // namespace blqh
