#include "blqh/fqhe.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include <omp.h>

namespace {

struct Axis {
    std::vector<double> values;
};

Axis parse_axis(const std::string &text) {
    Axis axis;
    const auto c1 = text.find(':');
    if (c1 == std::string::npos) {
        axis.values.push_back(std::stod(text));
        return axis;
    }
    const auto c2 = text.find(':', c1 + 1);
    if (c2 == std::string::npos)
        throw std::runtime_error("range must be start:stop:step, got " + text);
    const double start = std::stod(text.substr(0, c1));
    const double stop = std::stod(text.substr(c1 + 1, c2 - c1 - 1));
    const double step = std::stod(text.substr(c2 + 1));
    if (step == 0.0) throw std::runtime_error("range step is zero");
    const int n = static_cast<int>(std::floor((stop - start) / step + 0.5));
    if ((stop - start) * step < 0) throw std::runtime_error("range step has the wrong sign");
    for (int i = 0; i <= std::max(n, 0); ++i) axis.values.push_back(start + i * step);
    return axis;
}

int parse_count(int nphi, const std::string &text) {
    const auto slash = text.find('/');
    if (slash == std::string::npos) return std::stoi(text);
    return blqh::layer_electrons(nphi, std::stoi(text.substr(0, slash)), std::stoi(text.substr(slash + 1)));
}

struct Opt {
    std::string command = "help";
    blqh::Layers layers;
    std::string upper = "1/2";
    std::string lower = "1/2";
    std::string distance = "0";
    std::string bias = "0";
    std::string zeeman = "0";
    std::string tunnel = "0";
    std::string spin_flip = "0";
    std::string nu_total = "1";
    std::string delta_nu = "0:1";
    std::string xname = "bias";
    std::string yname = "tunnel";
    bool dense = false;
    bool lanczos = false;
    int steps = 80;
    int threads = 80;
    unsigned seed = 1;
    int wf = 8;
    std::string out;
    bool full_check = false;
};

std::string need(int &i, int argc, char **argv, const std::string &flag) {
    if (i + 1 >= argc) throw std::runtime_error("missing value for " + flag);
    return argv[++i];
}

Opt parse(int argc, char **argv) {
    Opt opt;
    if (argc < 2) return opt;
    opt.command = argv[1];
    for (int i = 2; i < argc; ++i) {
        const std::string flag = argv[i];
        if (flag == "--spin") opt.layers.spin = true;
        else if (flag == "--free") opt.layers.fix_flavor = false;
        else if (flag == "--dense") opt.dense = true;
        else if (flag == "--lanczos") opt.lanczos = true;
        else if (flag == "--nphi") opt.layers.nphi = std::stoi(need(i, argc, argv, flag));
        else if (flag == "--upper") opt.upper = need(i, argc, argv, flag);
        else if (flag == "--lower") opt.lower = need(i, argc, argv, flag);
        else if (flag == "--upper-up") opt.layers.upper_up = std::stoi(need(i, argc, argv, flag));
        else if (flag == "--upper-down") opt.layers.upper_down = std::stoi(need(i, argc, argv, flag));
        else if (flag == "--lower-up") opt.layers.lower_up = std::stoi(need(i, argc, argv, flag));
        else if (flag == "--lower-down") opt.layers.lower_down = std::stoi(need(i, argc, argv, flag));
        else if (flag == "--J") opt.layers.J = std::stoi(need(i, argc, argv, flag));
        else if (flag == "--K") opt.layers.K = std::stoi(need(i, argc, argv, flag));
        else if (flag == "--gamma") opt.layers.gamma = std::stod(need(i, argc, argv, flag));
        else if (flag == "--nLL") opt.layers.nLL = std::stoi(need(i, argc, argv, flag));
        else if (flag == "--d") opt.distance = need(i, argc, argv, flag);
        else if (flag == "--bias") opt.bias = need(i, argc, argv, flag);
        else if (flag == "--zeeman") opt.zeeman = need(i, argc, argv, flag);
        else if (flag == "--tunnel") opt.tunnel = need(i, argc, argv, flag);
        else if (flag == "--spin-flip") opt.spin_flip = need(i, argc, argv, flag);
        else if (flag == "--nu-total") opt.nu_total = need(i, argc, argv, flag);
        else if (flag == "--delta-nu") opt.delta_nu = need(i, argc, argv, flag);
        else if (flag == "--x") opt.xname = need(i, argc, argv, flag);
        else if (flag == "--y") opt.yname = need(i, argc, argv, flag);
        else if (flag == "--steps") opt.steps = std::stoi(need(i, argc, argv, flag));
        else if (flag == "--threads") opt.threads = std::stoi(need(i, argc, argv, flag));
        else if (flag == "--seed") opt.seed = static_cast<unsigned>(std::stoul(need(i, argc, argv, flag)));
        else if (flag == "--wf") opt.wf = std::stoi(need(i, argc, argv, flag));
        else if (flag == "--out") opt.out = need(i, argc, argv, flag);
        else if (flag == "full" && opt.command == "check") opt.full_check = true;
        else if (opt.command == "check" && flag.find_first_not_of("0123456789") == std::string::npos)
            opt.threads = std::stoi(flag);
        else
            throw std::runtime_error("unknown argument " + flag);
    }
    return opt;
}

void resolve_counts(Opt &opt) {
    if (opt.layers.nphi <= 0) throw std::runtime_error("--nphi is required");
    if (opt.command != "scan-imbalance") {
        opt.layers.n_upper = parse_count(opt.layers.nphi, opt.upper);
        opt.layers.n_lower = parse_count(opt.layers.nphi, opt.lower);
    }
    opt.layers.fields.distance = parse_axis(opt.distance).values.front();
    opt.layers.fields.bias = parse_axis(opt.bias).values.front();
    opt.layers.fields.zeeman = parse_axis(opt.zeeman).values.front();
    opt.layers.fields.tunneling = parse_axis(opt.tunnel).values.front();
    opt.layers.fields.spin_flip = parse_axis(opt.spin_flip).values.front();
}

std::string occupation(std::uint64_t det, int nphi, int nflavor) {
    std::string text;
    for (int f = 0; f < nflavor; ++f) {
        if (f) text += ' ';
        for (int o = 0; o < nphi; ++o) text += ((det >> (f * nphi + o)) & 1ULL) ? '1' : '0';
    }
    return text;
}

void write_wavefunction(const blqh::Sector &sector, const std::vector<std::complex<double>> &psi, int how_many) {
    std::vector<int> order(psi.size());
    for (int i = 0; i < static_cast<int>(psi.size()); ++i) order[i] = i;
    std::sort(order.begin(), order.end(), [&](int a, int b) { return std::norm(psi[a]) > std::norm(psi[b]); });
    const int n = std::min(how_many, static_cast<int>(order.size()));
    std::cerr << "largest wavefunction components\n";
    for (int k = 0; k < n; ++k) {
        const int i = order[k];
        std::cerr << "  " << std::norm(psi[i]) << "  "
                  << occupation(sector.id[i], sector.nphi, sector.nflavor) << "\n";
    }
}

struct Sample {
    blqh::GroundState state;
    blqh::Order order;
    int dim = 0;
};

Sample evaluate(const blqh::Layers &layers, const Opt &opt) {
    const blqh::System sys = blqh::make_layers(layers);
    const blqh::Sector sector = blqh::build_sector(sys);
    const blqh::Csr H = blqh::build_hamiltonian(sys, sector, blqh::make_geometry(sys));
    const bool dense = opt.dense || (!opt.lanczos && sector.nbasis <= 800);
    Sample sample;
    sample.state = blqh::lowest_state(H, dense, opt.steps, opt.seed);
    sample.order = blqh::measure(sys, sector, sample.state.psi);
    sample.dim = sector.nbasis;
    return sample;
}

const char *kComment =
    "# S/CAF/FM: Zheng, Radtke, Das Sarma, PRL 78, 2453 (1997);\n"
    "# Das Sarma, Sachdev, Zheng, PRB 58, 4672 (1998).\n"
    "# C/I versus d/l and Delta nu: Champagne, Eisenstein et al.,\n"
    "# PRL 100, 096801 (2008) and PRB 78, 205310 (2008).\n"
    "# Delta nu = nu_upper - nu_lower, nu_T = nu_upper + nu_lower.\n"
    "# ex_sas is the interlayer exciton weight ||T_sas psi||^2/N.\n"
    "# At fixed layer number coh_sas is zero; ex_sas still counts excitons.\n"
    "# C marks a coherent nu_T=1 exciton condensate, I the incoherent layers\n"
    "# (at Delta nu=1/3 those are the single-layer 2/3 and 1/3 states).\n";

const char *kHeader =
    "d,delta_nu,nu_upper,nu_lower,n_upper,n_lower,bias,zeeman,tunnel,spin_flip,dim,solver,"
    "E0,E0_per_N,gap,sz,layer,n0,n1,n2,n3,coh_sas,coh_sf,fluc_sas,fluc_af,ex_sas,phase\n";

void write_row(std::ostream &out, const blqh::Layers &layers, const Sample &sample) {
    const int nel = layers.n_upper + layers.n_lower;
    const double gap = sample.state.lowest.size() > 1 ? sample.state.lowest[1] - sample.state.lowest[0]
                                                       : std::nan("");
    const auto &o = sample.order;
    const double nu_u = static_cast<double>(layers.n_upper) / layers.nphi;
    const double nu_l = static_cast<double>(layers.n_lower) / layers.nphi;
    out << layers.fields.distance << ',' << (nu_u - nu_l) << ',' << nu_u << ',' << nu_l << ','
        << layers.n_upper << ',' << layers.n_lower << ',' << layers.fields.bias << ',' << layers.fields.zeeman
        << ',' << layers.fields.tunneling << ',' << layers.fields.spin_flip << ',' << sample.dim << ','
        << (sample.state.dense ? "dense" : "lanczos") << ',' << sample.state.energy << ','
        << sample.state.energy / nel << ',' << gap << ',' << o.sz << ',' << o.layer << ',' << o.n_uu << ','
        << o.n_ud << ',' << o.n_du << ',' << o.n_dd << ',' << o.coh_sas << ',' << o.coh_sf << ','
        << o.fluc_sas << ',' << o.fluc_af << ',' << o.ex_sas << ',' << o.phase << '\n';
}

void usage() {
    std::cerr
        << "fqhe check [threads] [full]\n"
        << "fqhe point --nphi N --upper 1/3 --lower 2/3 [--spin] [--free]\n"
        << "           [--d 1] [--bias 0] [--zeeman 0] [--tunnel 0] [--spin-flip 0]\n"
        << "           [--J -1 --K -1] [--wf 8] [--out point.csv]\n"
        << "fqhe scan-d --nphi N --upper 1/3 --lower 2/3 --spin --free --d 0:2:0.25 [--out scan.csv]\n"
        << "fqhe scan-phase --nphi N --upper 1/2 --lower 1/2 --spin --free\n"
        << "           --x bias --y tunnel --bias -0.2:0.2:0.1 --tunnel 0:0.2:0.1 [--out phase.csv]\n"
        << "           --y spin-flip sweeps the spin-changing interlayer tunneling\n"
        << "fqhe scan-imbalance --nphi N --nu-total 1 --d 0:2:0.25 --delta-nu 0:1\n"
        << "           [--spin] [--tunnel 0] [--out imbalance.csv]\n"
        << "\n"
        << "scan-imbalance follows Champagne et al., PRB 78, 205310 (2008):\n"
        << "total filling nu_T fixed, Delta nu = nu_upper - nu_lower, axis d/l.\n"
        << "Delta nu = 0 is balanced; Delta nu = 1/3 is (2/3, 1/3).\n"
        << "Only integer layer occupations inside the Delta-nu window are kept.\n"
        << "\n"
        << "--upper and --lower take an electron number or a filling p/q.\n"
        << "Positive bias raises the upper layer. Positive zeeman raises spin up.\n";
}

}  // namespace

int main(int argc, char **argv) {
    try {
        Opt opt = parse(argc, argv);
        if (opt.command == "help" || opt.command == "-h" || opt.command == "--help") {
            usage();
            return 0;
        }
        if (opt.command == "check") return blqh::run_checks(opt.threads, opt.full_check);

        resolve_counts(opt);
        omp_set_num_threads(opt.threads);
        std::ofstream file;
        std::ostream *out = &std::cout;
        if (!opt.out.empty()) {
            file.open(opt.out);
            if (!file) throw std::runtime_error("cannot open " + opt.out);
            out = &file;
        }
        out->setf(std::ios::scientific);
        out->precision(8);
        *out << kComment << kHeader;

        if (opt.command == "point") {
            const Sample sample = evaluate(opt.layers, opt);
            write_row(*out, opt.layers, sample);
            const int nel = opt.layers.n_upper + opt.layers.n_lower;
            std::cerr << "dim=" << sample.dim << " E0/N=" << sample.state.energy / nel
                      << " phase=" << sample.order.phase << " sz=" << sample.order.sz
                      << " layer=" << sample.order.layer << " coh_sas=" << sample.order.coh_sas
                      << " fluc_af=" << sample.order.fluc_af << "\n";
            if (opt.wf > 0)
                write_wavefunction(blqh::build_sector(blqh::make_layers(opt.layers)), sample.state.psi, opt.wf);
            return 0;
        }

        if (opt.command == "scan-d") {
            for (double d : parse_axis(opt.distance).values) {
                opt.layers.fields.distance = d;
                const Sample sample = evaluate(opt.layers, opt);
                write_row(*out, opt.layers, sample);
                std::cerr << "d=" << d << " E0/N=" << sample.state.energy / (opt.layers.n_upper + opt.layers.n_lower)
                          << " phase=" << sample.order.phase << "\n";
            }
            return 0;
        }

        if (opt.command == "scan-phase") {
            const Axis xs = parse_axis(opt.xname == "bias" ? opt.bias : opt.xname == "zeeman" ? opt.zeeman
                                                           : opt.xname == "d"     ? opt.distance
                                                           : opt.xname == "tunnel" ? opt.tunnel
                                                                                   : opt.spin_flip);
            const Axis ys = parse_axis(opt.yname == "bias" ? opt.bias : opt.yname == "zeeman" ? opt.zeeman
                                                           : opt.yname == "d"     ? opt.distance
                                                           : opt.yname == "tunnel" ? opt.tunnel
                                                                                   : opt.spin_flip);
            if (opt.xname != "bias" && opt.xname != "zeeman" && opt.xname != "d" && opt.xname != "tunnel" &&
                opt.xname != "spin-flip")
                throw std::runtime_error("unknown --x " + opt.xname);
            if (opt.yname != "bias" && opt.yname != "zeeman" && opt.yname != "d" && opt.yname != "tunnel" &&
                opt.yname != "spin-flip")
                throw std::runtime_error("unknown --y " + opt.yname);
            auto set_axis = [](blqh::Fields &fields, const std::string &name, double value) {
                if (name == "d") fields.distance = value;
                else if (name == "bias") fields.bias = value;
                else if (name == "zeeman") fields.zeeman = value;
                else if (name == "tunnel") fields.tunneling = value;
                else fields.spin_flip = value;
            };
            for (double y : ys.values) {
                for (double x : xs.values) {
                    set_axis(opt.layers.fields, opt.xname, x);
                    set_axis(opt.layers.fields, opt.yname, y);
                    const Sample sample = evaluate(opt.layers, opt);
                    write_row(*out, opt.layers, sample);
                    std::cerr << opt.xname << "=" << x << " " << opt.yname << "=" << y
                              << " phase=" << sample.order.phase << "\n";
                }
            }
            return 0;
        }

        if (opt.command == "scan-imbalance") {
            const double nu_t = [&]() {
                const auto slash = opt.nu_total.find('/');
                if (slash == std::string::npos) return std::stod(opt.nu_total);
                return std::stod(opt.nu_total.substr(0, slash)) / std::stod(opt.nu_total.substr(slash + 1));
            }();
            const double n_tot_exact = nu_t * opt.layers.nphi;
            const int n_tot = static_cast<int>(std::lround(n_tot_exact));
            if (std::abs(n_tot_exact - n_tot) > 1e-8)
                throw std::runtime_error("nu_total * nphi is not an integer electron number");
            const Axis distances = parse_axis(opt.distance);
            const auto slash = opt.delta_nu.find(':');
            const bool window = slash != std::string::npos && opt.delta_nu.find(':', slash + 1) == std::string::npos;
            double dnu_lo = 0;
            double dnu_hi = 0;
            Axis dnu_grid;
            if (window) {
                dnu_lo = std::stod(opt.delta_nu.substr(0, slash));
                dnu_hi = std::stod(opt.delta_nu.substr(slash + 1));
            } else {
                dnu_grid = parse_axis(opt.delta_nu);
            }
            const int slots = opt.layers.spin ? 2 : 1;
            bool any = false;
            for (double d : distances.values) {
                opt.layers.fields.distance = d;
                for (int n_u = 0; n_u <= n_tot; ++n_u) {
                    const int n_l = n_tot - n_u;
                    if (n_u > slots * opt.layers.nphi || n_l > slots * opt.layers.nphi) continue;
                    const double dnu = static_cast<double>(n_u - n_l) / opt.layers.nphi;
                    if (window) {
                        if (dnu < std::min(dnu_lo, dnu_hi) - 1e-8 || dnu > std::max(dnu_lo, dnu_hi) + 1e-8)
                            continue;
                    } else {
                        bool hit = false;
                        for (double target : dnu_grid.values)
                            if (std::abs(dnu - target) < 1e-8) hit = true;
                        if (!hit) continue;
                    }
                    opt.layers.n_upper = n_u;
                    opt.layers.n_lower = n_l;
                    opt.layers.upper_up = opt.layers.upper_down = opt.layers.lower_up = opt.layers.lower_down = -1;
                    const Sample sample = evaluate(opt.layers, opt);
                    write_row(*out, opt.layers, sample);
                    std::cerr << "d/l=" << d << " Delta_nu=" << dnu << " (" << n_u << "+" << n_l
                              << ") E0/N=" << sample.state.energy / n_tot << " ex_sas=" << sample.order.ex_sas
                              << " phase=" << sample.order.phase << "\n";
                    any = true;
                }
            }
            if (!any) throw std::runtime_error("no integer layer occupations in that Delta-nu window");
            return 0;
        }
        throw std::runtime_error("unknown command " + opt.command);
    } catch (const std::exception &err) {
        std::cerr << err.what() << "\n";
        usage();
        return 1;
    }
}
