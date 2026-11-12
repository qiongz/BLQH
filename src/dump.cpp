#include "blqh/fqhe.hpp"

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include <omp.h>

namespace blqh {
namespace {

struct DumpOpt {
    int nel = 0;
    int J = -1;
    int threads = 64;
    int samples = 16;
    bool dims = false;
    bool estimate = false;
    std::string out;
    std::uint64_t max_csr_bytes = 28ull << 30;
    std::uint64_t max_build_bytes = 100ull << 30;
};

void dump_usage() {
    std::cerr
        << "fqhe dump-ham --nel N [--J j] [--dims] [--estimate] [--out ham.npz]\n"
        << "           [--samples 16] [--threads 64]\n"
        << "           [--max-csr-bytes N] [--max-build-bytes N]\n"
        << "Square-torus Laughlin nu=1/3: Nphi=3N, gamma=1, legacy aspect off.\n"
        << "J < 0 is the full Fock space. Otherwise one orbital-sum block.\n";
}

DumpOpt parse_dump(int argc, char **argv) {
    DumpOpt opt;
    for (int i = 2; i < argc; ++i) {
        const std::string flag = argv[i];
        auto need = [&](const char *name) {
            if (i + 1 >= argc) throw std::runtime_error(std::string("missing value for ") + name);
            return std::string(argv[++i]);
        };
        if (flag == "--nel") opt.nel = std::stoi(need("--nel"));
        else if (flag == "--J") opt.J = std::stoi(need("--J"));
        else if (flag == "--threads") opt.threads = std::stoi(need("--threads"));
        else if (flag == "--samples") opt.samples = std::stoi(need("--samples"));
        else if (flag == "--out") opt.out = need("--out");
        else if (flag == "--dims") opt.dims = true;
        else if (flag == "--estimate") opt.estimate = true;
        else if (flag == "--max-csr-bytes") opt.max_csr_bytes = std::stoull(need("--max-csr-bytes"));
        else if (flag == "--max-build-bytes") opt.max_build_bytes = std::stoull(need("--max-build-bytes"));
        else throw std::runtime_error("unknown argument " + flag);
    }
    if (opt.nel <= 0) throw std::runtime_error("--nel is required");
    if (!opt.dims && !opt.estimate && opt.out.empty())
        throw std::runtime_error("dump-ham needs --dims, --estimate, or --out");
    return opt;
}

System laughlin_square(int nel, int J) {
    System sys;
    sys.model = Model::Monolayer;
    sys.nphi = 3 * nel;
    sys.nLL = 0;
    sys.flavors = {Flavor{0, 0, nel}};
    sys.J = J;
    sys.K = -1;
    sys.gamma = 1.0;
    sys.legacy_monolayer_gamma = false;
    sys.conserve_flavor = true;
    return sys;
}

long long binom(int n, int k) {
    if (k < 0 || k > n) return 0;
    if (k > n - k) k = n - k;
    __int128 r = 1;
    for (int i = 1; i <= k; ++i) {
        r = r * (n - k + i) / i;
    }
    return static_cast<long long>(r);
}

std::vector<long long> orbital_block_dims(int nphi, int nel) {
    std::vector<long long> dp(static_cast<size_t>(nel + 1) * nphi, 0);
    dp[0] = 1;
    for (int orb = 0; orb < nphi; ++orb) {
        for (int j = nel - 1; j >= 0; --j) {
            for (int s = 0; s < nphi; ++s) {
                const long long v = dp[static_cast<size_t>(j) * nphi + s];
                if (v == 0) continue;
                int ns = s + orb;
                if (ns >= nphi) ns -= nphi;
                dp[static_cast<size_t>(j + 1) * nphi + ns] += v;
            }
        }
    }
    std::vector<long long> dims(nphi);
    for (int s = 0; s < nphi; ++s) dims[s] = dp[static_cast<size_t>(nel) * nphi + s];
    return dims;
}

std::uint64_t csr_bytes(long long n, long long nnz) {
    return static_cast<std::uint64_t>(n + 1) * 8ull + static_cast<std::uint64_t>(nnz) * 4ull +
           static_cast<std::uint64_t>(nnz) * 16ull;
}

std::uint64_t dense_bytes(long long n) { return 16ull * static_cast<std::uint64_t>(n) * static_cast<std::uint64_t>(n); }

int t_count(int nphi) {
    // t runs over [-nphi/2, nphi/2] with C++ integer division.
    return (nphi / 2) * 2 + 1;
}

long long nnz_upper(long long n, int nel, int nphi) {
    const long long pairs = static_cast<long long>(nel) * (nel - 1) / 2;
    return n * (pairs * t_count(nphi) + 1);
}

std::string loadavg() {
    std::ifstream in("/proc/loadavg");
    std::string line;
    std::getline(in, line);
    return line;
}

void print_dims(int nel) {
    const int nphi = 3 * nel;
    const auto dims = orbital_block_dims(nphi, nel);
    long long total = 0;
    long long max_dim = -1;
    int max_J = 0;
    for (int J = 0; J < nphi; ++J) {
        total += dims[J];
        if (dims[J] > max_dim) {
            max_dim = dims[J];
            max_J = J;
        }
        std::cout << "BLQH_BLOCK nel=" << nel << " nphi=" << nphi << " J=" << J << " dim=" << dims[J]
                  << " dense_bytes=" << dense_bytes(dims[J]) << "\n";
    }
    const long long expect = binom(nphi, nel);
    std::cout << "BLQH_DIMS nel=" << nel << " nphi=" << nphi << " full=" << total << " expect=" << expect
              << " max_J=" << max_J << " max_dim=" << max_dim << " dense_bytes=" << dense_bytes(max_dim)
              << " full_dense_bytes=" << dense_bytes(total) << " loadavg=" << loadavg() << "\n";
    if (total != expect) throw std::runtime_error("orbital blocks do not sum to the Fock dimension");
}

uint32_t crc_update(uint32_t crc, const void *data, size_t n) {
    static uint32_t table[256];
    static bool ready = false;
    if (!ready) {
        for (uint32_t i = 0; i < 256; ++i) {
            uint32_t c = i;
            for (int k = 0; k < 8; ++k) c = (c & 1u) ? (0xEDB88320u ^ (c >> 1)) : (c >> 1);
            table[i] = c;
        }
        ready = true;
    }
    const auto *p = static_cast<const unsigned char *>(data);
    for (size_t i = 0; i < n; ++i) crc = table[(crc ^ p[i]) & 0xFFu] ^ (crc >> 8);
    return crc;
}

uint32_t crc_finish(uint32_t crc) { return crc ^ 0xFFFFFFFFu; }

void write_u16(std::ostream &out, unsigned v) {
    const unsigned char b[2] = {static_cast<unsigned char>(v), static_cast<unsigned char>(v >> 8)};
    out.write(reinterpret_cast<const char *>(b), 2);
}

void write_u32(std::ostream &out, std::uint32_t v) {
    const unsigned char b[4] = {static_cast<unsigned char>(v), static_cast<unsigned char>(v >> 8),
                                static_cast<unsigned char>(v >> 16), static_cast<unsigned char>(v >> 24)};
    out.write(reinterpret_cast<const char *>(b), 4);
}

void write_u64(std::ostream &out, std::uint64_t v) {
    for (int i = 0; i < 8; ++i) {
        const unsigned char b = static_cast<unsigned char>(v >> (8 * i));
        out.write(reinterpret_cast<const char *>(&b), 1);
    }
}

std::string npy_prefix(const std::string &descr, const std::string &shape) {
    std::string dict = "{'descr': '" + descr + "', 'fortran_order': False, 'shape': " + shape + ", }";
    const int prefix = 10;
    int header_len = static_cast<int>(dict.size()) + 1;
    int pad = (64 - ((prefix + header_len) % 64)) % 64;
    dict.append(pad, ' ');
    dict.push_back('\n');
    std::string out;
    out.push_back(static_cast<char>(0x93));
    out += "NUMPY";
    out.push_back(1);
    out.push_back(0);
    const unsigned len = static_cast<unsigned>(dict.size());
    out.push_back(static_cast<char>(len & 0xFF));
    out.push_back(static_cast<char>((len >> 8) & 0xFF));
    out += dict;
    return out;
}

struct ZipItem {
    std::string name;
    uint32_t crc = 0;
    std::uint64_t size = 0;
    std::uint64_t offset = 0;
};

class NpyZip {
   public:
    explicit NpyZip(const std::string &path) : out_(open_out(path)) {
        if (!out_) throw std::runtime_error("cannot open " + path);
    }

    void add(const std::string &name, const std::string &descr, const std::string &shape, const void *data,
             std::uint64_t nbytes) {
        const std::string prefix = npy_prefix(descr, shape);
        const std::uint64_t size = prefix.size() + nbytes;
        uint32_t crc = crc_update(0xFFFFFFFFu, prefix.data(), prefix.size());
        crc = crc_finish(crc_update(crc, data, static_cast<size_t>(nbytes)));
        ZipItem item;
        item.name = name;
        item.crc = crc;
        item.size = size;
        item.offset = static_cast<std::uint64_t>(out_.tellp());
        write_local(item);
        out_.write(prefix.data(), static_cast<std::streamsize>(prefix.size()));
        if (nbytes > 0) {
            if (data == nullptr) throw std::runtime_error("null array");
            out_.write(static_cast<const char *>(data), static_cast<std::streamsize>(nbytes));
        }
        items_.push_back(item);
    }

    void finish() {
        const std::uint64_t cd = static_cast<std::uint64_t>(out_.tellp());
        for (const ZipItem &item : items_) write_central(item);
        const std::uint64_t cd_end = static_cast<std::uint64_t>(out_.tellp());
        const std::uint64_t cd_size = cd_end - cd;
        const bool zip64 = cd >= 0xFFFFFFFFull || cd_size >= 0xFFFFFFFFull || items_.size() >= 0xFFFFull ||
                           any_big();
        if (zip64) {
            const std::uint64_t eocd64 = cd_end;
            write_u32(out_, 0x06064b50u);
            write_u64(out_, 44);
            write_u16(out_, 45);
            write_u16(out_, 45);
            write_u32(out_, 0);
            write_u32(out_, 0);
            write_u64(out_, items_.size());
            write_u64(out_, items_.size());
            write_u64(out_, cd_size);
            write_u64(out_, cd);
            write_u32(out_, 0x07064b50u);
            write_u32(out_, 0);
            write_u64(out_, eocd64);
            write_u32(out_, 1);
        }
        write_u32(out_, 0x06054b50u);
        write_u16(out_, 0);
        write_u16(out_, 0);
        write_u16(out_, zip64 ? 0xFFFFu : static_cast<unsigned>(items_.size()));
        write_u16(out_, zip64 ? 0xFFFFu : static_cast<unsigned>(items_.size()));
        write_u32(out_, zip64 || cd_size >= 0xFFFFFFFFull ? 0xFFFFFFFFu : static_cast<std::uint32_t>(cd_size));
        write_u32(out_, zip64 || cd >= 0xFFFFFFFFull ? 0xFFFFFFFFu : static_cast<std::uint32_t>(cd));
        write_u16(out_, 0);
        if (!out_) throw std::runtime_error("write failed");
    }

   private:
    bool any_big() const {
        for (const ZipItem &item : items_)
            if (item.size >= 0xFFFFFFFFull || item.offset >= 0xFFFFFFFFull) return true;
        return false;
    }

    void write_local(const ZipItem &item) {
        const bool big = item.size >= 0xFFFFFFFFull;
        write_u32(out_, 0x04034b50u);
        write_u16(out_, big ? 45 : 20);
        write_u16(out_, 0);
        write_u16(out_, 0);
        write_u16(out_, 0);
        write_u16(out_, 0);
        write_u32(out_, item.crc);
        write_u32(out_, big ? 0xFFFFFFFFu : static_cast<std::uint32_t>(item.size));
        write_u32(out_, big ? 0xFFFFFFFFu : static_cast<std::uint32_t>(item.size));
        write_u16(out_, static_cast<unsigned>(item.name.size()));
        write_u16(out_, big ? 20 : 0);
        out_.write(item.name.data(), static_cast<std::streamsize>(item.name.size()));
        if (big) {
            write_u16(out_, 0x0001);
            write_u16(out_, 16);
            write_u64(out_, item.size);
            write_u64(out_, item.size);
        }
    }

    void write_central(const ZipItem &item) {
        const bool big_size = item.size >= 0xFFFFFFFFull;
        const bool big_off = item.offset >= 0xFFFFFFFFull;
        const unsigned extra = (big_size ? 16u : 0u) + (big_off ? 8u : 0u);
        write_u32(out_, 0x02014b50u);
        write_u16(out_, 45);
        write_u16(out_, (big_size || big_off) ? 45 : 20);
        write_u16(out_, 0);
        write_u16(out_, 0);
        write_u16(out_, 0);
        write_u16(out_, 0);
        write_u32(out_, item.crc);
        write_u32(out_, big_size ? 0xFFFFFFFFu : static_cast<std::uint32_t>(item.size));
        write_u32(out_, big_size ? 0xFFFFFFFFu : static_cast<std::uint32_t>(item.size));
        write_u16(out_, static_cast<unsigned>(item.name.size()));
        write_u16(out_, extra == 0 ? 0 : extra + 4);
        write_u16(out_, 0);
        write_u16(out_, 0);
        write_u16(out_, 0);
        write_u32(out_, 0);
        write_u32(out_, big_off ? 0xFFFFFFFFu : static_cast<std::uint32_t>(item.offset));
        out_.write(item.name.data(), static_cast<std::streamsize>(item.name.size()));
        if (extra) {
            write_u16(out_, 0x0001);
            write_u16(out_, extra);
            if (big_size) {
                write_u64(out_, item.size);
                write_u64(out_, item.size);
            }
            if (big_off) write_u64(out_, item.offset);
        }
    }

    static std::ofstream open_out(const std::string &path) {
        const auto parent = std::filesystem::path(path).parent_path();
        if (!parent.empty()) std::filesystem::create_directories(parent);
        return std::ofstream(path, std::ios::binary);
    }

    std::ofstream out_;
    std::vector<ZipItem> items_;
};

void write_csr_npz(const std::string &path, int nel, int nphi, int J, const Csr &H) {
    static_assert(sizeof(long) == 8, "row_ptr is stored as int64");
    static_assert(sizeof(int) == 4, "column index is stored as int32");
    static_assert(sizeof(std::complex<double>) == 16, "values are stored as complex128");
    const long long nnz = static_cast<long long>(H.val.size());
    std::vector<std::int64_t> meta = {nel, nphi, J, H.n, nnz, 1, 0, 0};
    NpyZip zip(path);
    zip.add("row_ptr.npy", "<i8", "(" + std::to_string(H.row_ptr.size()) + ",)", H.row_ptr.data(),
            H.row_ptr.size() * sizeof(long));
    zip.add("col.npy", "<i4", "(" + std::to_string(H.col.size()) + ",)", H.col.data(),
            H.col.size() * sizeof(int));
    zip.add("val.npy", "<c16", "(" + std::to_string(H.val.size()) + ",)", H.val.data(),
            H.val.size() * sizeof(std::complex<double>));
    zip.add("meta.npy", "<i8", "(8,)", meta.data(), meta.size() * sizeof(std::int64_t));
    zip.finish();
}

int refuse_size(int nel, int nphi, int J, long long n, long long nnz, const std::string &why) {
    std::cout << "BLQH_SKIP nel=" << nel << " nphi=" << nphi << " J=" << J << " n=" << n << " nnz=" << nnz
              << " csr_bytes=" << csr_bytes(n, nnz) << " dense_bytes=" << dense_bytes(n) << " why=" << why
              << " loadavg=" << loadavg() << "\n";
    return 2;
}

}  // namespace

int dump_ham_main(int argc, char **argv) {
    try {
        const DumpOpt opt = parse_dump(argc, argv);
        std::cout.setf(std::ios::unitbuf);
        std::cerr.setf(std::ios::unitbuf);
        omp_set_num_threads(opt.threads);
        if (opt.dims) print_dims(opt.nel);
        if (!opt.estimate && opt.out.empty()) return 0;

        const System sys = laughlin_square(opt.nel, opt.J);
        const int nphi = sys.nphi;
        if (opt.J < -1 || opt.J >= nphi) throw std::runtime_error("--J is outside -1 .. Nphi-1");
        std::cerr << "building sector nel=" << opt.nel << " nphi=" << nphi << " J=" << opt.J << "\n";
        const Sector sector = build_sector(sys);
        const long long n = sector.nbasis;
        const long long upper = nnz_upper(n, opt.nel, nphi);
        const std::uint64_t upper_csr = csr_bytes(n, upper);
        const std::uint64_t upper_build = upper * 96ull;
        long long nnz_est = upper;
        bool estimated = false;
        if (opt.estimate || upper_csr > opt.max_csr_bytes || upper_build > opt.max_build_bytes) {
            std::cerr << "sampling " << opt.samples << " rows (upper nnz=" << upper << ")\n";
            const double per = sample_hamiltonian_nnz_per_row(sys, sector, make_geometry(sys), opt.samples);
            nnz_est = static_cast<long long>(per * static_cast<double>(n) + 0.5);
            estimated = true;
            std::cout << "BLQH_EST nel=" << opt.nel << " nphi=" << nphi << " J=" << opt.J << " n=" << n
                      << " nnz_per_row=" << per << " nnz=" << nnz_est << " csr_bytes=" << csr_bytes(n, nnz_est)
                      << " dense_bytes=" << dense_bytes(n) << " samples=" << opt.samples
                      << " upper_nnz=" << upper << " loadavg=" << loadavg() << "\n";
        }
        if (opt.out.empty()) return 0;
        if (csr_bytes(n, nnz_est) > opt.max_csr_bytes)
            return refuse_size(opt.nel, nphi, opt.J, n, nnz_est, "csr");
        if (nnz_est * 96ull > opt.max_build_bytes)
            return refuse_size(opt.nel, nphi, opt.J, n, nnz_est, "build");
        std::cerr << "building Hamiltonian n=" << n << (estimated ? " (estimate accepted)" : "") << "\n";
        const Csr H = build_hamiltonian(sys, sector, make_geometry(sys));
        const long long nnz = static_cast<long long>(H.val.size());
        if (csr_bytes(n, nnz) > opt.max_csr_bytes) {
            return refuse_size(opt.nel, nphi, opt.J, n, nnz, "csr-actual");
        }
        write_csr_npz(opt.out, opt.nel, nphi, opt.J, H);
        std::cout << "BLQH_CSR nel=" << opt.nel << " nphi=" << nphi << " J=" << opt.J << " n=" << H.n
                  << " nnz=" << nnz << " csr_bytes=" << csr_bytes(n, nnz) << " dense_bytes=" << dense_bytes(n)
                  << " out=" << opt.out << " loadavg=" << loadavg() << "\n";
        return 0;
    } catch (const std::exception &err) {
        std::cerr << err.what() << "\n";
        dump_usage();
        return 1;
    }
}

}  // namespace blqh
