// CPU-only machine snapshot for benchmark validity: load, AC power, thermal,
// CPU ticks and VM counters. No model is loaded and no GPU work is submitted.
#import <Foundation/Foundation.h>
#include <IOKit/ps/IOPowerSources.h>
#include <IOKit/ps/IOPSKeys.h>
#include <mach/mach.h>
#include <mach/host_info.h>
#include <mach/vm_statistics.h>
#include <mach/mach_time.h>
#include <libproc.h>
#include <sys/resource.h>
#include <sys/sysctl.h>
#include <time.h>
#include <unistd.h>

#include <cstdio>
#include <cstdlib>

int main(int argc, char** argv) {
  @autoreleasepool {
    double load[3] = {-1, -1, -1};
    const bool load_ok = getloadavg(load, 3) == 3;
    host_cpu_load_info_data_t cpu{};
    vm_statistics64_data_t vm{};
    mach_msg_type_number_t cpu_n = HOST_CPU_LOAD_INFO_COUNT, vm_n = HOST_VM_INFO64_COUNT;
    const host_t host = mach_host_self();
    const bool cpu_ok = host_statistics(host, HOST_CPU_LOAD_INFO, reinterpret_cast<host_info_t>(&cpu), &cpu_n) == KERN_SUCCESS;
    const bool vm_ok = host_statistics64(host, HOST_VM_INFO64, reinterpret_cast<host_info64_t>(&vm), &vm_n) == KERN_SUCCESS;
    mach_port_deallocate(mach_task_self(), host);
    CFTypeRef info = IOPSCopyPowerSourcesInfo();
    CFStringRef source = info ? IOPSGetProvidingPowerSourceType(info) : nullptr;
    const bool power_ok = source != nullptr;
    const bool ac = source && CFEqual(source, CFSTR(kIOPSACPowerValue));
    const bool battery = source && CFEqual(source, CFSTR(kIOPSBatteryPowerValue));
    int pressure = -1;
    size_t length = sizeof(pressure);
    if (sysctlbyname("kern.memorystatus_vm_pressure_level", &pressure, &length, nullptr, 0)) pressure = -1;
    struct timespec now{};
    clock_gettime(CLOCK_MONOTONIC, &now);
    uint64_t owner_cpu_ns = 0;
    mach_timebase_info_data_t timebase{};
    bool owner_ok = mach_timebase_info(&timebase) == KERN_SUCCESS && timebase.denom;
    for (int i = 1; i < argc; ++i) {
      char* end = nullptr;
      const long pid = strtol(argv[i], &end, 10);
      struct rusage_info_v4 usage{};
      if (pid <= 0 || !end || *end || proc_pid_rusage(pid_t(pid), RUSAGE_INFO_V4,
                                                   reinterpret_cast<rusage_info_t*>(&usage))) owner_ok = false;
      // rusage CPU times are Mach absolute ticks (24 MHz on this M4 Pro),
      // not nanoseconds. The wide intermediate also handles long uptimes.
      else if (owner_ok) owner_cpu_ns += uint64_t((__uint128_t(usage.ri_user_time) + usage.ri_system_time) *
                                                 timebase.numer / timebase.denom);
    }
    const long cpus = sysconf(_SC_NPROCESSORS_ONLN);
    const bool valid = load_ok && cpu_ok && vm_ok && power_ok && owner_ok && cpus > 0;
    printf("{\"valid\":%s,\"monotonic_ns\":%llu,\"load1\":%.8f,\"load5\":%.8f,\"load15\":%.8f,"
           "\"power_source\":\"%s\",\"ac\":%s,\"thermal\":%ld,\"low_power\":%s,"
           "\"cpu_ticks\":[%u,%u,%u,%u],\"pageouts\":%llu,\"swapins\":%llu,\"swapouts\":%llu,"
           "\"free_pages\":%u,\"inactive_pages\":%u,\"compressor_pages\":%u,\"page_size\":%ld,"
           "\"memory_pressure_level\":%d,\"cpu_count\":%ld,\"owner_cpu_ns\":%llu}\n",
           valid ? "true" : "false", static_cast<unsigned long long>(now.tv_sec) * 1000000000ull + now.tv_nsec,
           load[0], load[1], load[2], ac ? "AC Power" : battery ? "Battery Power" : "Other/Unknown", ac ? "true" : "false",
           static_cast<long>(NSProcessInfo.processInfo.thermalState), NSProcessInfo.processInfo.lowPowerModeEnabled ? "true" : "false",
           cpu.cpu_ticks[CPU_STATE_USER], cpu.cpu_ticks[CPU_STATE_SYSTEM], cpu.cpu_ticks[CPU_STATE_IDLE], cpu.cpu_ticks[CPU_STATE_NICE],
           static_cast<unsigned long long>(vm.pageouts), static_cast<unsigned long long>(vm.swapins),
           static_cast<unsigned long long>(vm.swapouts), vm.free_count, vm.inactive_count, vm.compressor_page_count,
           sysconf(_SC_PAGESIZE), pressure, cpus, static_cast<unsigned long long>(owner_cpu_ns));
    if (info) CFRelease(info);
    return valid ? 0 : 1;
  }
}
