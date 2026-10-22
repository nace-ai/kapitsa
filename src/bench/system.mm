// The Mac `kapitsa bench` runs on and its conditions (bench.h): sysctl, IOKit, NSProcessInfo and libproc, without
// spawning the tools the Python harness reads (ioreg, vm_stat, footprint) except where only the tool knows (pmset's power
// mode, footprint's mapped files).
#import <Foundation/Foundation.h>

#include "bench/bench.h"

#include <CoreFoundation/CoreFoundation.h>
#include <fcntl.h>
#include <IOKit/IOKitLib.h>
#include <IOKit/ps/IOPSKeys.h>
#include <IOKit/ps/IOPowerSources.h>
#include <libproc.h>
#include <mach/mach.h>
#include <spawn.h>
#include <sys/file.h>
#include <sys/resource.h>
#include <sys/stat.h>
#include <sys/sysctl.h>
#include <sys/wait.h>
#include <unistd.h>

#include <cstring>
#include <regex>

extern char** environ;

namespace ni {
extern const char* const kMacModels;  // bench/community/mac-models.json (CMakeLists.txt embeds it)
}

namespace ni::bench {

namespace {

std::string sysctl_str(const char* name) {
  size_t n = 0;
  if (sysctlbyname(name, nullptr, &n, nullptr, 0) != 0 || n == 0) return "";
  std::string s(n, '\0');
  if (sysctlbyname(name, s.data(), &n, nullptr, 0) != 0) return "";
  s.resize(strnlen(s.c_str(), n));
  return s;
}

int64_t sysctl_int(const char* name) {
  int64_t v = 0;
  size_t n = sizeof v;
  if (sysctlbyname(name, &v, &n, nullptr, 0) != 0) return -1;
  if (n == sizeof(int32_t)) return int64_t(*reinterpret_cast<int32_t*>(&v));
  return v;
}

long long cf_number(CFTypeRef v) {
  long long x = 0;
  if (v && CFGetTypeID(v) == CFNumberGetTypeID()) CFNumberGetValue((CFNumberRef)v, kCFNumberLongLongType, &x);
  return x;
}

std::string cf_string(CFTypeRef v) {
  if (!v || CFGetTypeID(v) != CFStringGetTypeID()) return "";
  char b[512];
  return CFStringGetCString((CFStringRef)v, b, sizeof b, kCFStringEncodingUTF8) ? b : "";
}

// Calls fn on the GPU's accelerator entry (AGXAccelerator)
template <class F>
void each_accelerator(F fn) {
  io_iterator_t it = 0;
  if (IOServiceGetMatchingServices(kIOMainPortDefault, IOServiceMatching("AGXAccelerator"), &it) != KERN_SUCCESS) return;
  for (io_object_t e; (e = IOIteratorNext(it)); IOObjectRelease(e)) fn(e);
  IOObjectRelease(it);
}

int gpu_core_count() {
  int cores = 0;
  each_accelerator([&](io_object_t e) {
    if (cores > 0) return;
    CFTypeRef v = IORegistryEntryCreateCFProperty(e, CFSTR("gpu-core-count"), kCFAllocatorDefault, 0);
    cores = int(cf_number(v));
    if (v) CFRelease(v);
  });
  return cores;
}

// Runs a program (its full path) and returns what it prints, "" when it can't run
std::string run_output(const std::vector<std::string>& argv) {
  int fd[2];
  if (pipe(fd) != 0) return "";
  posix_spawn_file_actions_t fa;
  posix_spawn_file_actions_init(&fa);
  posix_spawn_file_actions_adddup2(&fa, fd[1], 1);
  posix_spawn_file_actions_addopen(&fa, 2, "/dev/null", O_WRONLY, 0);
  posix_spawn_file_actions_addclose(&fa, fd[0]);
  std::vector<char*> args;
  for (const auto& a : argv) args.push_back(const_cast<char*>(a.c_str()));
  args.push_back(nullptr);
  pid_t pid = 0;
  const int rc = posix_spawn(&pid, args[0], &fa, nullptr, args.data(), environ);
  posix_spawn_file_actions_destroy(&fa);
  close(fd[1]);
  std::string out;
  if (rc == 0) {
    char b[4096];
    ssize_t n;
    while ((n = read(fd[0], b, sizeof b)) > 0) out.append(b, size_t(n));
    waitpid(pid, nullptr, 0);
  }
  close(fd[0]);
  return out;
}

struct Proc {
  int pid, ppid;
  std::string comm;
};

std::vector<Proc> all_procs() {
  int mib[4] = {CTL_KERN, KERN_PROC, KERN_PROC_ALL, 0};
  std::vector<Proc> out;
  for (int attempt = 0; attempt < 4; ++attempt) {
    size_t n = 0;
    if (sysctl(mib, 3, nullptr, &n, nullptr, 0) != 0) return out;
    std::vector<kinfo_proc> v(n / sizeof(kinfo_proc) + 32);
    n = v.size() * sizeof(kinfo_proc);
    if (sysctl(mib, 3, v.data(), &n, nullptr, 0) != 0) continue;  // grew in between
    v.resize(n / sizeof(kinfo_proc));
    for (const auto& p : v) out.push_back({p.kp_proc.p_pid, p.kp_eproc.e_ppid, p.kp_proc.p_comm});
    return out;
  }
  return out;
}

}  // namespace

// ---------------------------------------------------------------- the Mac

const ojson& mac_models() {
  static const ojson j = [] {
    ojson x = ojson::parse(kMacModels, nullptr, false);
    return x.is_discarded() ? ojson::object() : x;
  }();
  return j;
}

bool mac_model(const std::string& identifier, std::string* name, int* year) {
  const ojson& j = mac_models();
  if (!j.contains("models") || !j["models"].is_array()) return false;
  for (const auto& m : j["models"])
    if (m.value("identifier", "") == identifier) {
      *name = m.value("name", "");
      *year = m.value("year", 0);
      return true;
    }
  return false;
}

std::string chip_short(const std::string& chip) { return chip.rfind("Apple ", 0) == 0 ? chip.substr(6) : chip; }

std::string mac_display(const std::string& name, int year, const std::string& chip, const std::string& identifier) {
  (void)chip;  // shown beside the name by every caller
  if (name.empty()) return identifier;
  // Apple's names since the M5 give the chip instead of the year ("MacBook Pro (14-inch, M5)", "Mac mini (M6)"): shown
  // as the older ones are, with the year ("MacBook Pro (14-inch, 2025)"), since the chip is shown beside it. A name with
  // a year is Apple's as it is.
  const size_t open = name.rfind('('), close = name.rfind(')');
  if (open == std::string::npos || close == std::string::npos || close < open || year <= 0) return name;
  std::vector<std::string> parts;
  for (size_t a = open + 1; a <= close;) {
    size_t b = name.find(", ", a);
    if (b == std::string::npos || b > close) b = close;
    parts.push_back(name.substr(a, b - a));
    a = b + (b == close ? 1 : 2);
  }
  static const std::regex has_year("\\b20\\d\\d\\b"), names_chip("M\\d+( (Pro|Max|Ultra))?( or M\\d+( (Pro|Max|Ultra))?)*");
  for (const auto& p : parts)
    if (std::regex_search(p, has_year)) return name;
  std::string inner;
  for (const auto& p : parts)
    if (!std::regex_match(p, names_chip)) inner += (inner.empty() ? "" : ", ") + p;
  inner += (inner.empty() ? "" : ", ") + std::to_string(year);
  return name.substr(0, open + 1) + inner + name.substr(close);
}

Mac this_mac() {
  Mac m;
  m.identifier = sysctl_str("hw.model");
  mac_model(m.identifier, &m.name, &m.year);
  m.chip = sysctl_str("machdep.cpu.brand_string");
  m.cpu_cores = int(sysctl_int("hw.physicalcpu"));
  m.cpu_performance_cores = int(std::max<int64_t>(0, sysctl_int("hw.perflevel0.physicalcpu")));
  m.cpu_efficiency_cores = int(std::max<int64_t>(0, sysctl_int("hw.perflevel1.physicalcpu")));
  m.gpu_cores = gpu_core_count();
  m.memory_bytes = uint64_t(std::max<int64_t>(0, sysctl_int("hw.memsize")));
  m.macos = sysctl_str("kern.osproductversion");
  m.macos_build = sysctl_str("kern.osversion");
  return m;
}

// ---------------------------------------------------------------- conditions

Power power(bool with_mode) {
  Power p;
  CFTypeRef blob = IOPSCopyPowerSourcesInfo();
  if (blob) {
    CFStringRef src = IOPSGetProvidingPowerSourceType(blob);
    const std::string s = cf_string(src);
    p.source = s == kIOPMACPowerKey ? "ac" : s == kIOPMBatteryPowerKey ? "battery" : s == kIOPMUPSPowerKey ? "ups" : "";
    if (CFArrayRef list = IOPSCopyPowerSourcesList(blob)) {
      for (CFIndex i = 0; i < CFArrayGetCount(list); ++i) {
        CFDictionaryRef d = IOPSGetPowerSourceDescription(blob, CFArrayGetValueAtIndex(list, i));
        if (!d || cf_string(CFDictionaryGetValue(d, CFSTR(kIOPSTypeKey))) != kIOPSInternalBatteryType) continue;
        const long long cur = cf_number(CFDictionaryGetValue(d, CFSTR(kIOPSCurrentCapacityKey)));
        const long long max = cf_number(CFDictionaryGetValue(d, CFSTR(kIOPSMaxCapacityKey)));
        if (max > 0) p.battery_percent = int(cur * 100 / max);
      }
      CFRelease(list);
    }
    CFRelease(blob);
  }
  @autoreleasepool {
    p.low_power_mode = [[NSProcessInfo processInfo] isLowPowerModeEnabled];
  }
  if (!with_mode) return p;
  // the energy mode in use (pmset -g: "powermode 0|1|2", older systems "lowpowermode 0|1")
  const std::string out = run_output({"/usr/bin/pmset", "-g"});
  std::smatch m;
  if (std::regex_search(out, m, std::regex("\\n\\s*powermode\\s+(\\d+)")))
    p.power_mode = m[1] == "2" ? "high" : m[1] == "1" ? "low" : "automatic";
  else if (std::regex_search(out, m, std::regex("\\n\\s*lowpowermode\\s+(\\d+)")))
    p.power_mode = m[1] == "1" ? "low" : "automatic";
  if (p.low_power_mode) p.power_mode = "low";
  return p;
}

std::string thermal_state() {
  @autoreleasepool {
    switch ([NSProcessInfo processInfo].thermalState) {
      case NSProcessInfoThermalStateNominal: return "nominal";
      case NSProcessInfoThermalStateFair: return "fair";
      case NSProcessInfoThermalStateSerious: return "serious";
      case NSProcessInfoThermalStateCritical: return "critical";
    }
  }
  return "";
}

int thermal_rank(const std::string& s) {
  return s == "nominal" ? 0 : s == "fair" ? 1 : s == "serious" ? 2 : s == "critical" ? 3 : -1;
}

Vm vm() {
  Vm v;
  xsw_usage sw{};
  size_t n = sizeof sw;
  if (sysctlbyname("vm.swapusage", &sw, &n, nullptr, 0) == 0) v.swap_used_mb = double(sw.xsu_used) / 1048576.0;
  vm_statistics64_data_t st{};
  mach_msg_type_number_t cnt = HOST_VM_INFO64_COUNT;
  if (host_statistics64(mach_host_self(), HOST_VM_INFO64, (host_info64_t)&st, &cnt) == KERN_SUCCESS)
    v.swapouts = int64_t(st.swapouts), v.swapins = int64_t(st.swapins), v.pageouts = int64_t(st.pageouts);
  v.memorystatus_level = int(sysctl_int("kern.memorystatus_level"));
  return v;
}

// ---------------------------------------------------------------- GPU time and processes

GpuSnap gpu_snapshot() {
  // what `ioreg -r -c AGXDeviceUserClient -l` shows: each GPU client of the accelerator, its creator ("pid 123, name")
  // and the accumulatedGPUTime of its AppUsage entries
  GpuSnap out;
  each_accelerator([&](io_object_t acc) {
    io_iterator_t kids = 0;
    if (IORegistryEntryCreateIterator(acc, kIOServicePlane, kIORegistryIterateRecursively, &kids) != KERN_SUCCESS) return;
    for (io_object_t e; (e = IOIteratorNext(kids)); IOObjectRelease(e)) {
      CFTypeRef cr = IORegistryEntryCreateCFProperty(e, CFSTR("IOUserClientCreator"), kCFAllocatorDefault, 0);
      const std::string creator = cf_string(cr);
      if (cr) CFRelease(cr);
      int pid = 0;
      char name[256] = "";
      if (creator.empty() || sscanf(creator.c_str(), "pid %d, %255[^\n]", &pid, name) < 1) continue;
      CFTypeRef au = IORegistryEntryCreateCFProperty(e, CFSTR("AppUsage"), kCFAllocatorDefault, 0);
      uint64_t t = 0;
      if (au && CFGetTypeID(au) == CFArrayGetTypeID())
        for (CFIndex i = 0; i < CFArrayGetCount((CFArrayRef)au); ++i) {
          CFTypeRef d = CFArrayGetValueAtIndex((CFArrayRef)au, i);
          if (d && CFGetTypeID(d) == CFDictionaryGetTypeID())
            t += uint64_t(cf_number(CFDictionaryGetValue((CFDictionaryRef)d, CFSTR("accumulatedGPUTime"))));
        }
      if (au) CFRelease(au);
      out[{pid, name}] += t;
    }
    IOObjectRelease(kids);
  });
  return out;
}

int64_t gpu_memory_in_use() {
  int64_t v = -1;
  each_accelerator([&](io_object_t e) {
    if (v >= 0) return;
    CFTypeRef ps = IORegistryEntryCreateCFProperty(e, CFSTR("PerformanceStatistics"), kCFAllocatorDefault, 0);
    if (ps && CFGetTypeID(ps) == CFDictionaryGetTypeID()) {
      CFTypeRef x = CFDictionaryGetValue((CFDictionaryRef)ps, CFSTR("In use system memory"));
      if (x) v = cf_number(x);
    }
    if (ps) CFRelease(ps);
  });
  return v;
}

std::set<int> process_tree(const std::vector<int>& roots) {
  std::set<int> res;
  std::map<int, std::vector<int>> kids;
  for (const auto& p : all_procs()) kids[p.ppid].push_back(p.pid);
  std::vector<int> st;
  for (int r : roots)
    if (r > 0) st.push_back(r);
  while (!st.empty()) {
    const int x = st.back();
    st.pop_back();
    if (!res.insert(x).second) continue;
    for (int k : kids[x]) st.push_back(k);
  }
  return res;
}

std::set<int> ancestors() {
  std::map<int, int> parent;
  for (const auto& p : all_procs()) parent[p.pid] = p.ppid;
  std::set<int> out;
  for (int p = getppid(); p > 1 && out.insert(p).second;) {
    const auto it = parent.find(p);
    if (it == parent.end()) break;
    p = it->second;
  }
  return out;
}

std::string app_name(int pid, const std::string& comm) {
  char path[PROC_PIDPATHINFO_MAXSIZE];
  if (pid <= 0 || proc_pidpath(pid, path, sizeof path) <= 0) return comm;
  const std::string p = path;
  if (const size_t a = p.find(".app/"); a != std::string::npos) {
    const size_t s = p.rfind('/', a);
    return p.substr(s == std::string::npos ? 0 : s + 1, a - (s == std::string::npos ? 0 : s + 1));
  }
  const size_t s = p.rfind('/');
  return s == std::string::npos ? p : p.substr(s + 1);
}

Footprint footprint(const std::set<int>& pids, bool mapped) {
  Footprint f;
  for (int pid : pids) {
    rusage_info_v4 ri{};
    if (proc_pid_rusage(pid, RUSAGE_INFO_V4, (rusage_info_t*)&ri) != 0) continue;
    f.phys += ri.ri_phys_footprint;
    f.peak += ri.ri_lifetime_max_phys_footprint;
  }
  if (mapped) {
    // footprint's "mapped file" row: dirty, swapped, clean, reclaimable, wired bytes; the clean ones are mapped weights
    // that phys_footprint leaves out (bench.py footprint_snapshot)
    int64_t clean = 0;
    bool any = false;
    for (int pid : pids) {
      const std::string out = run_output({"/usr/bin/footprint", "-f", "bytes", "-w", "-p", std::to_string(pid)});
      std::smatch m;
      static const std::regex row("\\n\\s*(\\d+) B\\s+(\\d+) B\\s+(\\d+) B\\s+(\\d+) B\\s+(\\d+) B\\s+\\d+\\s+mapped file\\s*\\n");
      if (std::regex_search(out, m, row)) clean += std::stoll(m[3]), any = true;
      else if (out.find("phys_footprint") != std::string::npos) any = true;  // no mapped files
    }
    f.mapped_clean = any ? clean : -1;
  }
  return f;
}

std::string GpuLock::path() {
  // One file for the whole Mac, in /tmp so that every session and every tree defaults to the same one: the harness used
  // to default to ~/.cache/nace-gpu.lock while direct runs took lockf on /tmp/kapitsa-gpu.lock, and two GPU jobs that
  // take different files do not exclude each other (it happened twice). BENCH_GPU_LOCK still names another file.
  static const char* kDefault = "/tmp/kapitsa-gpu.lock";
  if (const char* p = getenv("BENCH_GPU_LOCK"); p && *p) {
    const std::string s = p;
    const char* h = getenv("HOME");
    return s.rfind("~/", 0) == 0 && h ? h + s.substr(1) : s;
  }
  return kDefault;
}

bool GpuLock::acquire(const std::function<void(const std::string&)>& waiting) {
  release();
  const std::string p = path();
  mkdir(p.substr(0, p.find_last_of('/')).c_str(), 0755);
  const int fd = open(p.c_str(), O_RDONLY | O_CREAT | O_CLOEXEC, 0644);
  if (fd < 0) return false;
  if (flock(fd, LOCK_EX | LOCK_NB) != 0) {
    if (waiting) waiting(lock_holders(p));
    // block in flock() as lockf does: the kernel queues blocked waiters, so polling would lose every release to them
    while (flock(fd, LOCK_EX) != 0)
      if (errno != EINTR) {
        const int e = errno;
        close(fd);
        errno = e;
        return false;
      }
  }
  fd_ = fd;
  return true;
}

void GpuLock::release() {
  if (fd_ < 0) return;
  flock(fd_, LOCK_UN);
  close(fd_);
  fd_ = -1;
}

std::string lock_holders(const std::string& path) {
  const std::string out = run_output({"/usr/sbin/lsof", "-t", path});
  std::string names;
  std::map<int, std::string> comm;
  for (const auto& p : all_procs()) comm[p.pid] = p.comm;
  for (size_t a = 0; a < out.size();) {
    size_t b = out.find('\n', a);
    if (b == std::string::npos) b = out.size();
    const int pid = atoi(out.substr(a, b - a).c_str());
    a = b + 1;
    if (pid <= 0 || pid == getpid()) continue;
    names += (names.empty() ? "" : ", ") + app_name(pid, comm.count(pid) ? comm[pid] : "?") + " (pid " + std::to_string(pid) + ")";
  }
  return names;
}

std::vector<int> other_kapitsas(int except) {
  std::vector<int> out;
  const int self = getpid();
  for (const auto& p : all_procs()) {
    if (p.pid == self || p.pid == except || p.pid <= 0) continue;
    if (p.comm == "kapitsa" || p.comm.rfind("nace-infer", 0) == 0) out.push_back(p.pid);
  }
  return out;
}

}  // namespace ni::bench
