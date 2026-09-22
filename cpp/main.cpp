// main.cpp — Winfree ER network raw trial data (Ariaratnam & Strogatz 2001)
// Build with ./build.sh. Do not compile this file alone.
//
// Dynamics (matches Ariaratnam2001.py):
//   dθ/dt = ω - (K/N) * sin(θ) ⊙ (A @ (1 + cos(θ)))
// where A is an undirected ER adjacency matrix.

#include "third_party/cnpy/cnpy.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include <sys/stat.h>

#ifdef USE_OPENMP
#include <omp.h>
#endif

static bool path_exists(const std::string& path) {
  struct stat st;
  return stat(path.c_str(), &st) == 0;
}

static void create_directories(const std::string& path) {
  if (path.empty() || path_exists(path)) {
    return;
  }
  const std::size_t pos = path.find_last_of('/');
  if (pos != std::string::npos) {
    create_directories(path.substr(0, pos));
  }
  mkdir(path.c_str(), 0755);
}

constexpr int kDefaultN = 1000;
constexpr double kDefaultT = 10000.0;
constexpr double kDefaultT0 = 8000.0;
constexpr double kDefaultP = 0.8;
constexpr double kRtol = 1e-6;
constexpr double kAtol = 1e-8;
constexpr double kPi = 3.14159265358979323846;

enum class Integrator { RK4, RK45 };

struct Params {
  double k_min = 0.0;
  double k_max = 2.0;
  double k_step = 0.1;
  double delta_min = 0.0;
  double delta_max = 1.0;
  double delta_step = 0.05;
  double p = kDefaultP;
  int trials = 1;
  int n = kDefaultN;
  double T = kDefaultT;
  double T0 = kDefaultT0;
  int workers = 0;  // 0 => OpenMP default
  uint64_t seed = 0;
  std::string output_dir = "raw_trials";
  bool skip_existing = false;
  bool benchmark = false;
  Integrator integrator = Integrator::RK4;
  double dt = 0.0;  // fixed step for RK4; 0 => auto from omega range
};

struct GridCell {
  double K;
  double delta;
  int trial;
};

struct SimBuffers {
  std::vector<double> theta;
  std::vector<double> k1, k2, k3, k4, k5, k6, k7;
  std::vector<double> ystage;
  std::vector<double> y_next;
  std::vector<double> one_plus_cos;
  std::vector<double> interaction;
  std::vector<std::vector<int>> neighbors;
};

static std::vector<double> arange_inclusive(double vmin, double vmax, double step) {
  std::vector<double> vals;
  if (step <= 0.0) {
    throw std::runtime_error("grid step must be positive");
  }
  const int n = static_cast<int>(std::lround((vmax - vmin) / step)) + 1;
  vals.reserve(static_cast<size_t>(n));
  for (int i = 0; i < n; ++i) {
    const double v = vmin + i * step;
    vals.push_back(std::round(v * 100.0) / 100.0);
  }
  return vals;
}

static uint32_t trial_seed(double K, double delta, double p, int trial, uint64_t global_seed) {
  auto float_bits = [](double x) -> uint64_t {
    uint64_t u;
    std::memcpy(&u, &x, sizeof(double));
    return u;
  };
  uint64_t h = global_seed;
  h ^= float_bits(K) + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
  h ^= float_bits(delta) + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
  h ^= float_bits(p) + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
  h ^= static_cast<uint64_t>(trial) + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
  h ^= h >> 33;
  h *= 0xff51afd7ed558ccdULL;
  h ^= h >> 33;
  h *= 0xc4ceb9fe1a85ec53ULL;
  h ^= h >> 33;
  return static_cast<uint32_t>(h & 0xFFFFFFFFULL);
}

static void natural_frequencies(double delta, int n, std::vector<double>& omega) {
  omega.resize(static_cast<size_t>(n));
  if (n == 1) {
    omega[0] = 1.0;
    return;
  }
  const double lo = 1.0 - delta;
  const double hi = 1.0 + delta;
  for (int i = 0; i < n; ++i) {
    omega[static_cast<size_t>(i)] = lo + (hi - lo) * static_cast<double>(i) / static_cast<double>(n - 1);
  }
}

static void make_er_neighbors(int n, double p, std::mt19937& rng, std::vector<std::vector<int>>& neighbors) {
  neighbors.assign(static_cast<size_t>(n), {});
  std::uniform_real_distribution<double> uni(0.0, 1.0);
  for (int i = 0; i < n; ++i) {
    for (int j = i + 1; j < n; ++j) {
      if (uni(rng) < p) {
        neighbors[static_cast<size_t>(i)].push_back(j);
        neighbors[static_cast<size_t>(j)].push_back(i);
      }
    }
  }
}

static void winfree_rhs(const std::vector<double>& theta,
                        const std::vector<double>& omega,
                        const std::vector<std::vector<int>>& neighbors,
                        double K,
                        int n,
                        std::vector<double>& one_plus_cos,
                        std::vector<double>& interaction,
                        std::vector<double>& dtheta) {
  const double scale = K / static_cast<double>(n);
  const double* theta_data = theta.data();
  double* opc = one_plus_cos.data();
  double* inter = interaction.data();
  const double* omega_data = omega.data();
  double* dtheta_data = dtheta.data();

  for (int j = 0; j < n; ++j) {
    opc[j] = 1.0 + std::cos(theta_data[j]);
  }
  std::fill(inter, inter + n, 0.0);
  for (int i = 0; i < n; ++i) {
    double sum = 0.0;
    for (int j : neighbors[static_cast<size_t>(i)]) {
      sum += opc[j];
    }
    inter[i] = sum;
  }
  for (int i = 0; i < n; ++i) {
    dtheta_data[i] = omega_data[i] - scale * std::sin(theta_data[i]) * inter[i];
  }
}

static double max_omega_magnitude(const std::vector<double>& omega) {
  double m = 0.0;
  for (double w : omega) {
    m = std::max(m, std::abs(w));
  }
  return m;
}

static double default_rk4_dt(const std::vector<double>& omega) {
  const double wmax = std::max(max_omega_magnitude(omega), 1.0);
  // ~20 samples per fastest natural period 2π/ω
  return std::min(0.1, (2.0 * kPi) / (20.0 * wmax));
}

static void blend_toward_target(const std::vector<double>& y0,
                                const std::vector<double>& y1,
                                double alpha,
                                int n,
                                std::vector<double>& out) {
  for (int i = 0; i < n; ++i) {
    out[static_cast<size_t>(i)] = y0[static_cast<size_t>(i)] +
                                   alpha * (y1[static_cast<size_t>(i)] - y0[static_cast<size_t>(i)]);
  }
}

static void capture_crossing(double t,
                             double t_next,
                             double target,
                             bool& got,
                             const std::vector<double>& y0,
                             const std::vector<double>& y1,
                             int n,
                             std::vector<double>& out) {
  if (!got && t < target && t_next >= target) {
    blend_toward_target(y0, y1, (target - t) / (t_next - t), n, out);
    got = true;
  }
}

// Dormand–Prince RK5(4) — same family as SciPy solve_ivp(method="RK45")
static bool rk45_step(const std::vector<double>& y,
                      double h,
                      const std::vector<double>& omega,
                      const std::vector<std::vector<int>>& neighbors,
                      double K,
                      int n,
                      SimBuffers& buf,
                      double& err_norm) {
  auto& k1 = buf.k1;
  auto& k2 = buf.k2;
  auto& k3 = buf.k3;
  auto& k4 = buf.k4;
  auto& k5 = buf.k5;
  auto& k6 = buf.k6;
  auto& k7 = buf.k7;
  auto& ys = buf.ystage;
  auto& y_next = buf.y_next;

  winfree_rhs(y, omega, neighbors, K, n, buf.one_plus_cos, buf.interaction, k1);

  for (int i = 0; i < n; ++i) {
    ys[static_cast<size_t>(i)] = y[static_cast<size_t>(i)] + h * (1.0 / 5.0) * k1[static_cast<size_t>(i)];
  }
  winfree_rhs(ys, omega, neighbors, K, n, buf.one_plus_cos, buf.interaction, k2);

  for (int i = 0; i < n; ++i) {
    ys[static_cast<size_t>(i)] = y[static_cast<size_t>(i)] +
                                 h * (3.0 / 40.0 * k1[static_cast<size_t>(i)] + 9.0 / 40.0 * k2[static_cast<size_t>(i)]);
  }
  winfree_rhs(ys, omega, neighbors, K, n, buf.one_plus_cos, buf.interaction, k3);

  for (int i = 0; i < n; ++i) {
    ys[static_cast<size_t>(i)] = y[static_cast<size_t>(i)] +
                                 h * (44.0 / 45.0 * k1[static_cast<size_t>(i)] - 56.0 / 15.0 * k2[static_cast<size_t>(i)] +
                                      32.0 / 9.0 * k3[static_cast<size_t>(i)]);
  }
  winfree_rhs(ys, omega, neighbors, K, n, buf.one_plus_cos, buf.interaction, k4);

  for (int i = 0; i < n; ++i) {
    ys[static_cast<size_t>(i)] =
        y[static_cast<size_t>(i)] +
        h * (19372.0 / 6561.0 * k1[static_cast<size_t>(i)] - 25360.0 / 2187.0 * k2[static_cast<size_t>(i)] +
             64448.0 / 6561.0 * k3[static_cast<size_t>(i)] - 212.0 / 729.0 * k4[static_cast<size_t>(i)]);
  }
  winfree_rhs(ys, omega, neighbors, K, n, buf.one_plus_cos, buf.interaction, k5);

  for (int i = 0; i < n; ++i) {
    ys[static_cast<size_t>(i)] =
        y[static_cast<size_t>(i)] +
        h * (9017.0 / 3168.0 * k1[static_cast<size_t>(i)] - 355.0 / 33.0 * k2[static_cast<size_t>(i)] +
             46732.0 / 5247.0 * k3[static_cast<size_t>(i)] + 49.0 / 176.0 * k4[static_cast<size_t>(i)] -
             5103.0 / 18656.0 * k5[static_cast<size_t>(i)]);
  }
  winfree_rhs(ys, omega, neighbors, K, n, buf.one_plus_cos, buf.interaction, k6);

  for (int i = 0; i < n; ++i) {
    y_next[static_cast<size_t>(i)] =
        y[static_cast<size_t>(i)] +
        h * (35.0 / 384.0 * k1[static_cast<size_t>(i)] + 500.0 / 1113.0 * k3[static_cast<size_t>(i)] +
             125.0 / 192.0 * k4[static_cast<size_t>(i)] - 2187.0 / 6784.0 * k5[static_cast<size_t>(i)] +
             11.0 / 84.0 * k6[static_cast<size_t>(i)]);
  }
  winfree_rhs(y_next, omega, neighbors, K, n, buf.one_plus_cos, buf.interaction, k7);

  err_norm = 0.0;
  for (int i = 0; i < n; ++i) {
    const size_t idx = static_cast<size_t>(i);
    const double err_i =
        h * (-71.0 / 57600.0 * k1[idx] + 71.0 / 16695.0 * k3[idx] - 71.0 / 1920.0 * k4[idx] +
             17253.0 / 339200.0 * k5[idx] - 22.0 / 525.0 * k6[idx] + 1.0 / 40.0 * k7[idx]);
    const double scale = kAtol + kRtol * std::max(std::abs(y[idx]), std::abs(y_next[idx]));
    const double ratio = err_i / scale;
    err_norm = std::max(err_norm, std::abs(ratio));
  }
  return err_norm <= 1.0;
}

static void rk4_step(const std::vector<double>& y,
                     double h,
                     const std::vector<double>& omega,
                     const std::vector<std::vector<int>>& neighbors,
                     double K,
                     int n,
                     SimBuffers& buf) {
  auto& k1 = buf.k1;
  auto& k2 = buf.k2;
  auto& k3 = buf.k3;
  auto& k4 = buf.k4;
  auto& ys = buf.ystage;
  auto& y_next = buf.y_next;
  const double h2 = 0.5 * h;
  const double h6 = h / 6.0;

  winfree_rhs(y, omega, neighbors, K, n, buf.one_plus_cos, buf.interaction, k1);

  for (int i = 0; i < n; ++i) {
    ys[static_cast<size_t>(i)] = y[static_cast<size_t>(i)] + h2 * k1[static_cast<size_t>(i)];
  }
  winfree_rhs(ys, omega, neighbors, K, n, buf.one_plus_cos, buf.interaction, k2);

  for (int i = 0; i < n; ++i) {
    ys[static_cast<size_t>(i)] = y[static_cast<size_t>(i)] + h2 * k2[static_cast<size_t>(i)];
  }
  winfree_rhs(ys, omega, neighbors, K, n, buf.one_plus_cos, buf.interaction, k3);

  for (int i = 0; i < n; ++i) {
    ys[static_cast<size_t>(i)] = y[static_cast<size_t>(i)] + h * k3[static_cast<size_t>(i)];
  }
  winfree_rhs(ys, omega, neighbors, K, n, buf.one_plus_cos, buf.interaction, k4);

  for (int i = 0; i < n; ++i) {
    const size_t idx = static_cast<size_t>(i);
    y_next[idx] = y[idx] + h6 * (k1[idx] + 2.0 * k2[idx] + 2.0 * k3[idx] + k4[idx]);
  }
}

static void integrate_rk4_to_times(const std::vector<double>& omega,
                                   const std::vector<std::vector<int>>& neighbors,
                                   double K,
                                   int n,
                                   std::vector<double> theta,
                                   double T0,
                                   double T,
                                   double dt,
                                   SimBuffers& buf,
                                   std::vector<double>& theta_t0,
                                   std::vector<double>& theta_t) {
  double h = dt;
  double t = 0.0;
  bool got_t0 = false;
  bool got_t = false;
  theta_t0.resize(static_cast<size_t>(n));
  theta_t.resize(static_cast<size_t>(n));

  while (t < T && !(got_t0 && got_t)) {
    if (t + h > T) {
      h = T - t;
    }

    const std::vector<double>& y_before = theta;
    rk4_step(theta, h, omega, neighbors, K, n, buf);
    const double t_next = t + h;
    const std::vector<double>& y_after = buf.y_next;

    capture_crossing(t, t_next, T0, got_t0, y_before, y_after, n, theta_t0);
    capture_crossing(t, t_next, T, got_t, y_before, y_after, n, theta_t);

    theta.swap(buf.y_next);
    t = t_next;
  }

  if (!got_t0 || !got_t) {
    throw std::runtime_error("integration did not reach T0 and T");
  }
}

static void integrate_rk45_to_times(const std::vector<double>& omega,
                                    const std::vector<std::vector<int>>& neighbors,
                                    double K,
                                    int n,
                                    std::vector<double> theta,
                                    double T0,
                                    double T,
                                    SimBuffers& buf,
                                    std::vector<double>& theta_t0,
                                    std::vector<double>& theta_t) {
  double t = 0.0;
  const double wmax = std::max(max_omega_magnitude(omega), 1.0);
  double h = std::min(0.1, (2.0 * kPi) / (10.0 * wmax));
  const double h_min = std::max(1e-12, T / 1e9);
  const double h_max = std::max(0.5, T / 50.0);
  const double safety = 0.9;
  const double grow = 5.0;
  const double shrink = 0.2;

  bool got_t0 = false;
  bool got_t = false;
  theta_t0.resize(static_cast<size_t>(n));
  theta_t.resize(static_cast<size_t>(n));

  while (t < T && !(got_t0 && got_t)) {
    if (t + h > T) {
      h = T - t;
    }

    double err_norm = 0.0;
    int attempts = 0;
    while (!rk45_step(theta, h, omega, neighbors, K, n, buf, err_norm)) {
      h = std::max(h * shrink, h_min);
      if (t + h > T) {
        h = T - t;
      }
      if (++attempts > 50) {
        throw std::runtime_error("RK45 step failed to converge");
      }
    }

    const std::vector<double>& y_before = theta;
    const double t_next = t + h;
    const std::vector<double>& y_after = buf.y_next;

    capture_crossing(t, t_next, T0, got_t0, y_before, y_after, n, theta_t0);
    capture_crossing(t, t_next, T, got_t, y_before, y_after, n, theta_t);

    theta.swap(buf.y_next);
    t = t_next;

    if (err_norm > 0.0 && err_norm < 0.8) {
      const double factor = safety * std::pow(1.0 / err_norm, 0.2);
      h = std::min(h_max, std::max(h_min, h * std::min(grow, factor)));
    }
  }

  if (!got_t0 || !got_t) {
    throw std::runtime_error("integration did not reach T0 and T");
  }
}

static void integrate_to_times(const std::vector<double>& omega,
                               const std::vector<std::vector<int>>& neighbors,
                               double K,
                               int n,
                               std::vector<double> theta,
                               double T0,
                               double T,
                               Integrator integrator,
                               double dt,
                               SimBuffers& buf,
                               std::vector<double>& theta_t0,
                               std::vector<double>& theta_t) {
  if (integrator == Integrator::RK4) {
    const double step = dt > 0.0 ? dt : default_rk4_dt(omega);
    integrate_rk4_to_times(omega, neighbors, K, n, std::move(theta), T0, T, step, buf, theta_t0, theta_t);
  } else {
    integrate_rk45_to_times(omega, neighbors, K, n, std::move(theta), T0, T, buf, theta_t0, theta_t);
  }
}

static std::string trial_path(const std::string& out_dir, double K, double delta, double p, int trial) {
  char buf[512];
  std::snprintf(buf, sizeof(buf), "%s/K_%.2f_delta_%.2f_p_%.2f_trial_%02d.npz", out_dir.c_str(), K, delta, p, trial);
  return std::string(buf);
}

static void save_trial_npz(const std::string& path,
                           double K,
                           double delta,
                           double p,
                           int trial,
                           int N,
                           double T,
                           double T0,
                           const std::vector<double>& omega,
                           const std::vector<double>& rotation_number) {
  const std::vector<size_t> vec_shape{static_cast<size_t>(N)};
  const std::vector<size_t> scalar_shape{1};

  cnpy::npz_save(path, "omega", omega.data(), vec_shape, "w");
  cnpy::npz_save(path, "rotation_number", rotation_number.data(), vec_shape, "a");
  cnpy::npz_save(path, "K", &K, scalar_shape, "a");
  cnpy::npz_save(path, "delta", &delta, scalar_shape, "a");
  cnpy::npz_save(path, "p", &p, scalar_shape, "a");

  const int64_t trial_i = trial;
  const int64_t N_i = N;
  const int64_t T_i = static_cast<int64_t>(T);
  const int64_t T0_i = static_cast<int64_t>(T0);

  cnpy::npz_save(path, "trial", &trial_i, scalar_shape, "a");
  cnpy::npz_save(path, "N", &N_i, scalar_shape, "a");
  cnpy::npz_save(path, "T", &T_i, scalar_shape, "a");
  cnpy::npz_save(path, "T0", &T0_i, scalar_shape, "a");
}

static SimBuffers make_buffers(int n) {
  SimBuffers buf;
  const size_t sz = static_cast<size_t>(n);
  buf.theta.resize(sz);
  buf.k1.resize(sz);
  buf.k2.resize(sz);
  buf.k3.resize(sz);
  buf.k4.resize(sz);
  buf.k5.resize(sz);
  buf.k6.resize(sz);
  buf.k7.resize(sz);
  buf.ystage.resize(sz);
  buf.y_next.resize(sz);
  buf.one_plus_cos.resize(sz);
  buf.interaction.resize(sz);
  return buf;
}

static double run_trial(const GridCell& cell,
                        const Params& params,
                        const std::string& out_dir) {
  const std::string path = trial_path(out_dir, cell.K, cell.delta, params.p, cell.trial);
  if (params.skip_existing && path_exists(path)) {
    return 0.0;
  }

  const auto t_start = std::chrono::steady_clock::now();

  const uint32_t seed = trial_seed(cell.K, cell.delta, params.p, cell.trial, params.seed);
  std::mt19937 rng(seed);
  std::uniform_real_distribution<double> phase_dist(0.0, 2.0 * kPi);

  SimBuffers buf = make_buffers(params.n);
  std::vector<double> omega;
  natural_frequencies(cell.delta, params.n, omega);
  make_er_neighbors(params.n, params.p, rng, buf.neighbors);

  for (int i = 0; i < params.n; ++i) {
    buf.theta[static_cast<size_t>(i)] = phase_dist(rng);
  }

  std::vector<double> theta_t0;
  std::vector<double> theta_t;
  integrate_to_times(omega, buf.neighbors, cell.K, params.n, buf.theta, params.T0, params.T, params.integrator,
                     params.dt, buf, theta_t0, theta_t);

  std::vector<double> rotation_number(static_cast<size_t>(params.n));
  const double window = params.T - params.T0;
  for (int i = 0; i < params.n; ++i) {
    rotation_number[static_cast<size_t>(i)] =
        (theta_t[static_cast<size_t>(i)] - theta_t0[static_cast<size_t>(i)]) / window;
  }

  create_directories(out_dir);
  save_trial_npz(path, cell.K, cell.delta, params.p, cell.trial, params.n, params.T, params.T0, omega,
                 rotation_number);

  const auto t_end = std::chrono::steady_clock::now();
  return std::chrono::duration<double>(t_end - t_start).count();
}

static void print_usage(const char* prog) {
  std::cerr
      << "Usage: " << prog << " [options]\n"
      << "  --k-min, --k-max, --k-step       K grid (default 0, 2, 0.1)\n"
      << "  --delta-min, --delta-max, --delta-step   delta grid (default 0, 1, 0.05)\n"
      << "  --p FLOAT                       ER edge probability (default 0.8)\n"
      << "  --trials INT                    trials per cell (default 1)\n"
      << "  --output-dir PATH               output directory (default raw_trials)\n"
      << "  --skip-existing                 skip cells with existing npz files\n"
      << "  --seed UINT                     global seed mixed into per-trial seeds\n"
      << "  --workers INT                   OpenMP threads (default: OMP_NUM_THREADS)\n"
      << "  --T FLOAT                       integration end time (default 10000)\n"
      << "  --T0 FLOAT                      rotation reference time (default 8000)\n"
      << "  --N INT                         number of oscillators (default 1000)\n"
      << "  --integrator rk4|rk45           ODE integrator (default rk4; rk45 matches SciPy)\n"
      << "  --dt FLOAT                      fixed RK4 step size (default: auto from omega)\n"
      << "  --benchmark                     quick smoke test: N=100, T=1000, T0=800\n";
}

static Integrator parse_integrator(const std::string& name) {
  if (name == "rk4") {
    return Integrator::RK4;
  }
  if (name == "rk45") {
    return Integrator::RK45;
  }
  throw std::runtime_error("unknown integrator: " + name + " (use rk4 or rk45)");
}

static const char* integrator_name(Integrator integrator) {
  return integrator == Integrator::RK4 ? "rk4" : "rk45";
}

static bool parse_args(int argc, char** argv, Params& params) {
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    auto need_val = [&](const char* name) -> std::string {
      if (i + 1 >= argc) {
        throw std::runtime_error(std::string("missing value for ") + name);
      }
      return argv[++i];
    };

    if (arg == "--k-min") {
      params.k_min = std::stod(need_val("--k-min"));
    } else if (arg == "--k-max") {
      params.k_max = std::stod(need_val("--k-max"));
    } else if (arg == "--k-step") {
      params.k_step = std::stod(need_val("--k-step"));
    } else if (arg == "--delta-min") {
      params.delta_min = std::stod(need_val("--delta-min"));
    } else if (arg == "--delta-max") {
      params.delta_max = std::stod(need_val("--delta-max"));
    } else if (arg == "--delta-step") {
      params.delta_step = std::stod(need_val("--delta-step"));
    } else if (arg == "--p") {
      params.p = std::stod(need_val("--p"));
    } else if (arg == "--trials") {
      params.trials = std::stoi(need_val("--trials"));
    } else if (arg == "--output-dir") {
      params.output_dir = need_val("--output-dir");
    } else if (arg == "--skip-existing") {
      params.skip_existing = true;
    } else if (arg == "--seed") {
      params.seed = static_cast<uint64_t>(std::stoull(need_val("--seed")));
    } else if (arg == "--workers") {
      params.workers = std::stoi(need_val("--workers"));
    } else if (arg == "--T") {
      params.T = std::stod(need_val("--T"));
    } else if (arg == "--T0") {
      params.T0 = std::stod(need_val("--T0"));
    } else if (arg == "--N") {
      params.n = std::stoi(need_val("--N"));
    } else if (arg == "--integrator") {
      params.integrator = parse_integrator(need_val("--integrator"));
    } else if (arg == "--dt") {
      params.dt = std::stod(need_val("--dt"));
      if (params.dt <= 0.0) {
        throw std::runtime_error("--dt must be positive");
      }
    } else if (arg == "--benchmark") {
      params.benchmark = true;
    } else if (arg == "--help" || arg == "-h") {
      print_usage(argv[0]);
      return false;
    } else {
      throw std::runtime_error("unknown argument: " + arg);
    }
  }
  return true;
}

int main(int argc, char** argv) {
  Params params;
  try {
    if (!parse_args(argc, argv, params)) {
      return 0;
    }
    if (params.trials < 1) {
      throw std::runtime_error("--trials must be >= 1");
    }
    if (params.workers < 0) {
      throw std::runtime_error("--workers must be >= 0");
    }
    if (params.benchmark) {
      params.n = 100;
      params.T = 1000.0;
      params.T0 = 800.0;
      std::cout << "Benchmark mode: N=100, T=1000, T0=800 (not comparable to production T=10000)\n";
    }
    if (params.T0 >= params.T) {
      throw std::runtime_error("--T0 must be < --T");
    }
    if (params.dt > 0.0 && params.integrator != Integrator::RK4) {
      throw std::runtime_error("--dt only applies to --integrator rk4");
    }

#ifdef USE_OPENMP
    if (params.workers > 0) {
      omp_set_num_threads(params.workers);
    }
#endif

    const std::vector<double> k_vals = arange_inclusive(params.k_min, params.k_max, params.k_step);
    const std::vector<double> delta_vals = arange_inclusive(params.delta_min, params.delta_max, params.delta_step);

    std::vector<GridCell> cells;
    cells.reserve(k_vals.size() * delta_vals.size() * static_cast<size_t>(params.trials));
    for (double delta : delta_vals) {
      for (double K : k_vals) {
        for (int trial = 0; trial < params.trials; ++trial) {
          cells.push_back({K, delta, trial});
        }
      }
    }

    std::cout << "Grid: " << k_vals.size() << " x " << delta_vals.size() << " = "
              << k_vals.size() * delta_vals.size() << " cells, trials=" << params.trials << ", total jobs="
              << cells.size() << ", N=" << params.n << ", p=" << params.p << ", T=" << params.T
              << ", T0=" << params.T0 << ", integrator=" << integrator_name(params.integrator);
    if (params.integrator == Integrator::RK4) {
      std::cout << ", dt=" << (params.dt > 0.0 ? std::to_string(params.dt) : "auto");
    }
    std::cout << std::endl;

    int done = 0;
    const int total = static_cast<int>(cells.size());
    double total_elapsed = 0.0;

#ifdef USE_OPENMP
#pragma omp parallel for schedule(dynamic) reduction(+ : total_elapsed)
#endif
    for (int idx = 0; idx < total; ++idx) {
      try {
        const double elapsed = run_trial(cells[static_cast<size_t>(idx)], params, params.output_dir);
        total_elapsed += elapsed;
#ifdef USE_OPENMP
#pragma omp critical
#endif
        {
          ++done;
          const GridCell& c = cells[static_cast<size_t>(idx)];
          std::cout << "[" << done << "/" << total << "] K=" << c.K << ", delta=" << c.delta
                    << ", trial=" << c.trial << " saved";
          if (elapsed > 0.0) {
            std::cout << " (" << elapsed << "s)";
          }
          std::cout << std::endl;
        }
      } catch (const std::exception& ex) {
#ifdef USE_OPENMP
#pragma omp critical
#endif
        {
          std::cerr << "Error in cell " << idx << ": " << ex.what() << std::endl;
        }
      }
    }

    if (total_elapsed > 0.0) {
      std::cout << "Total compute time (sum of cells): " << total_elapsed << "s";
      if (params.benchmark) {
        std::cout << " [benchmark]";
      }
      std::cout << std::endl;
    }

    std::cout << "Done. Output: " << params.output_dir << std::endl;
    return 0;
  } catch (const std::exception& ex) {
    std::cerr << "Fatal: " << ex.what() << std::endl;
    print_usage(argv[0]);
    return 1;
  }
}
