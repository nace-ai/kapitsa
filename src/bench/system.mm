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
#include <sys/resource.h>
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
  if (name.empty()) return identifier;
  std::string n = name;
  const size_t open = n.rfind('('), close = n.rfind(')');
  if (open == std::string::npos || close == std::string::npos || close < open) return n;
  // the parenthesis' parts: "16-inch", "M5 Pro or M5 Max" (the chip it is), the year where it has none
  std::vector<std::string> parts;
  for (size_t a = open + 1; a <= close;) {
    size_t b = n.find(", ", a);
    if (b == std::string::npos || b > close) b = close;
    parts.push_back(n.substr(a, b - a));
    a = b + (b == close ? 1 : 2);
  }
  const std::string c = chip_short(chip);
  bool has_year = false;
  for (auto& p : parts) {
    if (p.find(" or ") != std::string::npos) {
      for (size_t a = 0; a <= p.size();) {
        size_t b = p.find(" or ", a);
        if (b == std::string::npos) b = p.size();
        if (p.substr(a, b - a) == c) {
          p = c;
          break;
        }
        a = b + 4;
      }
    }
    if (std::regex_search(p, std::regex("\\b20\\d\\d\\b"))) has_year = true;
  }
  if (!has_year && year > 0) parts.push_back(std::to_string(year));
  std::string inner;
  for (const auto& p : parts) inner += (inner.empty() ? "" : ", ") + p;
  return n.substr(0, open + 1) + inner + n.substr(close);
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
