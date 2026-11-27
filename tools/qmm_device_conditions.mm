// CPU-only snapshot, or --watch MILLISECONDS ROOT_PID for a resident sampler.
// Reads IOKit counters without submitting GPU work; no model is loaded.
#import <Foundation/Foundation.h>
#include "bench/bench.h"
#include <IOKit/ps/IOPowerSources.h>
#include <IOKit/ps/IOPSKeys.h>
#include <libproc.h>
#include <mach/mach.h>
#include <mach/mach_time.h>
#include <mach/vm_statistics.h>
#include <sys/resource.h>
#include <sys/sysctl.h>
#include <time.h>
#include <unistd.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

static NSArray* gpu_counters(bool& ok) {
  NSMutableArray* rows = [NSMutableArray array];
  // The same native IOKit collector as kapitsa bench; no ioreg subprocess.
  const auto counters = ni::bench::gpu_snapshot();
  if (counters.empty()) ok = false;
  for (const auto& [key, ns] : counters) {
    NSString* name = [NSString stringWithUTF8String:key.second.c_str()];
    if (!name) { ok = false; continue; }
    [rows addObject:@{@"pid": @(key.first), @"name": name, @"gpu_ns": @(ns)}];
  }
  return rows;
}

static bool emit_snapshot(const std::vector<pid_t>& roots, bool watch) {
  @autoreleasepool {
    double load[3] = {-1, -1, -1};
    bool valid = getloadavg(load, 3) == 3;
    host_cpu_load_info_data_t cpu{};
    vm_statistics64_data_t vm{};
    mach_msg_type_number_t cpu_n = HOST_CPU_LOAD_INFO_COUNT, vm_n = HOST_VM_INFO64_COUNT;
    const host_t host = mach_host_self();
    valid &= host_statistics(host, HOST_CPU_LOAD_INFO, reinterpret_cast<host_info_t>(&cpu), &cpu_n) == KERN_SUCCESS;
    valid &= host_statistics64(host, HOST_VM_INFO64, reinterpret_cast<host_info64_t>(&vm), &vm_n) == KERN_SUCCESS;
    mach_port_deallocate(mach_task_self(), host);
    CFTypeRef info = IOPSCopyPowerSourcesInfo();
    CFStringRef source = info ? IOPSGetProvidingPowerSourceType(info) : nullptr;
    valid &= source != nullptr;
    const bool ac = source && CFEqual(source, CFSTR(kIOPSACPowerValue));
    const bool battery = source && CFEqual(source, CFSTR(kIOPSBatteryPowerValue));
    int pressure = -1;
    size_t length = sizeof(pressure);
    if (sysctlbyname("kern.memorystatus_vm_pressure_level", &pressure, &length, nullptr, 0)) pressure = -1;
    struct timespec now{};
    valid &= clock_gettime(CLOCK_MONOTONIC, &now) == 0;
    mach_timebase_info_data_t tb{};
    valid &= mach_timebase_info(&tb) == KERN_SUCCESS && tb.denom;
    const auto owners = watch ? ni::bench::process_tree(roots) : std::set<int>(roots.begin(), roots.end());
    uint64_t owner_cpu_ns = 0;
    NSMutableArray* owner_ids = [NSMutableArray array];
    for (pid_t pid : owners) {
      [owner_ids addObject:@(pid)];
      struct rusage_info_v4 usage{};
      if (proc_pid_rusage(pid, RUSAGE_INFO_V4, reinterpret_cast<rusage_info_t*>(&usage))) valid = false;
      // rusage uses Mach ticks (24 MHz on this M4 Pro), not nanoseconds.
      else if (tb.denom) owner_cpu_ns += uint64_t((__uint128_t(usage.ri_user_time) + usage.ri_system_time) * tb.numer / tb.denom);
    }
    const long cpus = sysconf(_SC_NPROCESSORS_ONLN);
    valid &= cpus > 0;
    NSArray* gpu = watch ? gpu_counters(valid) : @[];
    NSDictionary* value = @{
      @"valid": @(valid), @"monotonic_ns": @(uint64_t(now.tv_sec) * 1000000000ull + now.tv_nsec),
      @"load1": @(load[0]), @"load5": @(load[1]), @"load15": @(load[2]),
      @"power_source": ac ? @"AC Power" : battery ? @"Battery Power" : @"Other/Unknown", @"ac": @(ac),
      @"thermal": @(NSProcessInfo.processInfo.thermalState), @"low_power": @(NSProcessInfo.processInfo.lowPowerModeEnabled),
      @"cpu_ticks": @[@(cpu.cpu_ticks[CPU_STATE_USER]), @(cpu.cpu_ticks[CPU_STATE_SYSTEM]),
                      @(cpu.cpu_ticks[CPU_STATE_IDLE]), @(cpu.cpu_ticks[CPU_STATE_NICE])],
      @"pageouts": @(vm.pageouts), @"swapins": @(vm.swapins), @"swapouts": @(vm.swapouts),
      @"free_pages": @(vm.free_count), @"inactive_pages": @(vm.inactive_count), @"compressor_pages": @(vm.compressor_page_count),
      @"page_size": @(sysconf(_SC_PAGESIZE)), @"memory_pressure_level": @(pressure),
      @"cpu_count": @(cpus), @"owner_cpu_ns": @(owner_cpu_ns), @"owner_pids": owner_ids, @"gpu": gpu
    };
    NSError* error = nil;
    NSData* data = [NSJSONSerialization dataWithJSONObject:value options:0 error:&error];
    if (info) CFRelease(info);
    if (!data) return false;
    fwrite(data.bytes, 1, data.length, stdout);
    fputc('\n', stdout);
    fflush(stdout);
    return valid;
  }
}

int main(int argc, char** argv) {
  const bool watch = argc > 1 && !strcmp(argv[1], "--watch");
  const int interval = watch && argc > 2 ? atoi(argv[2]) : 0;
  if (watch && (argc < 4 || interval < 10 || interval > 10000)) return 2;
  std::vector<pid_t> roots;
  for (int i = watch ? 3 : 1; i < argc; ++i) {
    char* end = nullptr;
    const long pid = strtol(argv[i], &end, 10);
    if (pid <= 0 || pid > INT32_MAX || !end || *end) return 2;
    roots.push_back(pid_t(pid));
  }
  do {
    struct timespec start{}, end{};
    clock_gettime(CLOCK_MONOTONIC, &start);
    const bool valid = emit_snapshot(roots, watch);
    if (!watch) return valid ? 0 : 1;
    clock_gettime(CLOCK_MONOTONIC, &end);
    const int64_t used_ns = (end.tv_sec - start.tv_sec) * 1000000000ll + end.tv_nsec - start.tv_nsec;
    const int64_t wait_ns = int64_t(interval) * 1000000 - used_ns;
    if (wait_ns > 0) { struct timespec delay{time_t(wait_ns / 1000000000), long(wait_ns % 1000000000)}; nanosleep(&delay, nullptr); }
  } while (true);
}
