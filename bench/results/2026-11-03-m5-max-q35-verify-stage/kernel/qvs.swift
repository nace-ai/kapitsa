// qvs: the Qwen tree verify's history attention, two tensor kernels side by side (default attn8t_partial_zone_g8 and
// attn8ts_partial_zone_g8, its K / V tiles staged): the 35B's heads (16 over 2, head dim 256), 8 rows, a random tree, the
// engine's zone chunking for a capacity of pos0 + 64. Prints each kernel's best GPU time of REPS (interleaved), the
// cache bytes it reads per second, and whether their partials (O, max, sum of every row, head and chunk) are bitwise
// the same. usage: qvs TENSOR_SOURCE [A B] [positions...]
import Foundation
import Metal

let dev = MTLCreateSystemDefaultDevice()!
let queue = dev.makeCommandQueue()!
let args = CommandLine.arguments
let o = MTLCompileOptions()
o.languageVersion = .version4_0
let tl = try! dev.makeLibrary(source: try! String(contentsOfFile: args[1], encoding: .utf8), options: o)
let nameA = args.count > 2 ? args[2] : "attn8t_partial_zone_g8", nameB = args.count > 3 ? args[3] : "attn8ts_partial_zone_g8"
let pA = try! dev.makeComputePipelineState(function: tl.makeFunction(name: nameA)!)
let pB = try! dev.makeComputePipelineState(function: tl.makeFunction(name: nameB)!)
print("\(nameA): \(pA.staticThreadgroupMemoryLength) B threadgroup memory, max \(pA.maxTotalThreadsPerThreadgroup) threads; " +
      "\(nameB): \(pB.staticThreadgroupMemoryLength) B, max \(pB.maxTotalThreadsPerThreadgroup)")
if ProcessInfo.processInfo.environment["COMPILE_ONLY"] != nil { exit(0) }
struct P { var Hq, Hkv, pos0, chunk, max_seq: UInt32; var scale: Float; var nchunks: UInt32 }
// CFG=g6: the 27B's 16-row wide tree verify (24 query heads over 4 KV heads, attn8t_partial_tree_g6's grid: x the
// chunks x 2 row tiles, chunks as the engine's wide path)
let g6 = ProcessInfo.processInfo.environment["CFG"] == "g6"
let Hq = g6 ? 24 : 16, Hkv = g6 ? 4 : 2, G = g6 ? 6 : 8, D = 256, R = g6 ? 16 : 8
let reps = Int(ProcessInfo.processInfo.environment["REPS"] ?? "5")!
var rng = SystemRandomNumberGenerator()
func gauss() -> Float { Float(sqrt(-2 * log(Double.random(in: 1e-12..<1, using: &rng)))) * Float(cos(2 * .pi * Double.random(in: 0..<1, using: &rng))) }
func zone(_ q: Int) -> Int { q >= 7 ? (q - 7) / 32 * 32 : 0 }

func run(pos0: Int) {
  let cap = pos0 + 64
  var chunk = 128
  var nch = 0, nslot = 0
  if g6 {
    let Lmax = pos0 + R, want = (160 + Hkv - 1) / Hkv
    chunk = (max(64, (Lmax + want - 1) / want) + 31) / 32 * 32
    while ((Lmax + chunk - 1) / chunk) * R > 1024 { chunk += 32 }
    nch = (Lmax + chunk - 1) / chunk
    nslot = nch
  } else {
    while ((cap + chunk - 1) / chunk + 1) * R > 1024 { chunk += 32 }
    nch = max(1, (pos0 + chunk - 1) / chunk)
    nslot = nch + 1
  }
  var p = P(Hq: UInt32(Hq), Hkv: UInt32(Hkv), pos0: UInt32(pos0), chunk: UInt32(chunk), max_seq: UInt32(cap),
            scale: 1 / Float(D).squareRoot(), nchunks: UInt32(nslot))
  let qb = dev.makeBuffer(length: R * Hq * D * 4, options: .storageModeShared)!
  let kb = dev.makeBuffer(length: Hkv * cap * D * 2, options: .storageModeShared)!
  let vb = dev.makeBuffer(length: Hkv * cap * D * 2, options: .storageModeShared)!
  let q = qb.contents().bindMemory(to: Float.self, capacity: R * Hq * D)
  let k = kb.contents().bindMemory(to: Float16.self, capacity: Hkv * cap * D)
  let v = vb.contents().bindMemory(to: Float16.self, capacity: Hkv * cap * D)
  for i in 0..<(R * Hq * D) { q[i] = gauss() * 1.5 }
  for i in 0..<(Hkv * cap * D) { k[i] = Float16(gauss()); v[i] = Float16(gauss()) }
  let tb = dev.makeBuffer(length: 336 * 4, options: .storageModeShared)!
  let t = tb.contents().bindMemory(to: Int32.self, capacity: 336)
  for i in 0..<336 { t[i] = 0 }
  var mask = [UInt32](repeating: 0, count: R)
  mask[0] = 1
  for r in 1..<R { mask[r] = mask[Int.random(in: 0..<r, using: &rng)] | (1 << UInt32(r)) }
  for r in 0..<R { t[65 + r] = Int32(bitPattern: mask[r]) }
  t[0] = Int32(R)
  var Ru = UInt32(R)
  let n = R * Hq * nslot
  let po = [dev.makeBuffer(length: n * D * 4, options: .storageModeShared)!, dev.makeBuffer(length: n * D * 4, options: .storageModeShared)!]
  let pm = [dev.makeBuffer(length: n * 2 * 4, options: .storageModeShared)!, dev.makeBuffer(length: n * 2 * 4, options: .storageModeShared)!]
  for b in po + pm { memset(b.contents(), 0, b.length) }
  var best = [1e9, 1e9]
  for _ in 0..<reps {
    for (j, ps) in [pA, pB].enumerated() {
      let cb = queue.makeCommandBuffer()!
      let e = cb.makeComputeCommandEncoder()!
      e.setComputePipelineState(ps)
      e.setBuffer(qb, offset: 0, index: 0); e.setBuffer(kb, offset: 0, index: 1); e.setBuffer(vb, offset: 0, index: 2)
      e.setBuffer(po[j], offset: 0, index: 3); e.setBuffer(pm[j], offset: 0, index: 4)
      e.setBytes(&p, length: MemoryLayout<P>.size, index: 5); e.setBytes(&Ru, length: 4, index: 6); e.setBuffer(tb, offset: 0, index: 7)
      e.dispatchThreadgroups(MTLSize(width: g6 ? nch * 2 : nch, height: Hkv, depth: 1),
                             threadsPerThreadgroup: MTLSize(width: 64 * ((8 * G + 15) / 16), height: 1, depth: 1))
      e.endEncoding(); cb.commit(); cb.waitUntilCompleted()
      if let err = cb.error { print("GPU error", err); exit(1) }
      best[j] = min(best[j], (cb.gpuEndTime - cb.gpuStartTime) * 1000)
    }
  }
  // every written partial: rows t < R, every head, chunks [0, nch)
  var diff = 0, total = 0, nonfinite = 0
  let a0 = po[0].contents().bindMemory(to: UInt32.self, capacity: n * D), b0 = po[1].contents().bindMemory(to: UInt32.self, capacity: n * D)
  let a1 = pm[0].contents().bindMemory(to: UInt32.self, capacity: n * 2), b1 = pm[1].contents().bindMemory(to: UInt32.self, capacity: n * 2)
  let bf = po[1].contents().bindMemory(to: Float.self, capacity: n * D)
  for r in 0..<R { for h in 0..<Hq { for c in 0..<nch {
    let base = (r * Hq + h) * nslot + c
    for d in 0..<D { total += 1; if a0[base * D + d] != b0[base * D + d] { diff += 1 }; if !bf[base * D + d].isFinite { nonfinite += 1 } }
    for j in 0..<2 { total += 1; if a1[base * 2 + j] != b1[base * 2 + j] { diff += 1 } }
  } } }
  let bytes = Double(Hkv * (g6 ? pos0 + R : zone(pos0 + 7)) * D * 2 * 2)
  print(String(format: "pos0 %7d chunk %5d x %3d | %@ %7.3f ms (%5.0f GB/s)  %@ %7.3f ms (%5.0f GB/s)  %5.2fx | partials differing %d of %d, nonfinite %d",
               pos0, chunk, nch, nameA, best[0], bytes / best[0] / 1e6, nameB, best[1], bytes / best[1] / 1e6, best[0] / best[1],
               diff, total, nonfinite))
}

let positions = args.count > 4 ? args[4...].map { Int($0)! } : [40, 777, 4096, 32768, 131072, 262144]
for pos0 in positions { run(pos0: pos0) }
