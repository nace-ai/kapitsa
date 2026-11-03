// dd_check: Dory's decode / verify attention (one full-attention layer: 16 query heads over 4 KV heads, head dim 256)
// of R rows over L cached positions, split into N chunks per KV head: GPU time and the cache read's bandwidth.
// usage: dd_check DORY_SOURCE R L N[,N...] [KERNEL,...]  (the first kernel the reference of the combined outputs)
import Foundation
import Metal
let dev = MTLCreateSystemDefaultDevice()!
let queue = dev.makeCommandQueue()!
let lib = try! dev.makeLibrary(source: try! String(contentsOfFile: CommandLine.arguments[1], encoding: .utf8), options: MTLCompileOptions())
let R = Int(CommandLine.arguments[2])!, L = Int(CommandLine.arguments[3])!
let Ns = CommandLine.arguments[4].split(separator: ",").map { Int($0)! }
struct P { var R, pos0, Hq, Hkv, rows, ring, window, chunk, nchunks, k0: UInt32; var scale: Float }
let Hq = 16, Hkv = 4, HD = 256, G = 4
let rows = L + R + 16
let qb = dev.makeBuffer(length: R * Hq * HD * 4, options: .storageModeShared)!
let qp = qb.contents().bindMemory(to: Float.self, capacity: R * Hq * HD)
for i in 0..<(R * Hq * HD) { qp[i] = Float.random(in: -1...1) }
let kc = dev.makeBuffer(length: Hkv * rows * HD * 2, options: .storageModeShared)!
let vc = dev.makeBuffer(length: Hkv * rows * HD * 2, options: .storageModeShared)!
for b in [kc, vc] { let p = b.contents().bindMemory(to: UInt16.self, capacity: Hkv * rows * HD); for i in stride(from: 0, to: Hkv * rows * HD, by: 1) { p[i] = 0x3000 | UInt16(i & 0x3ff) } }
let maxN = Ns.max()!
let po = dev.makeBuffer(length: R * Hq * maxN * HD * 4, options: .storageModeShared)!
let pml = dev.makeBuffer(length: R * Hq * maxN * 2 * 4, options: .storageModeShared)!
let names = CommandLine.arguments.count > 5 ? CommandLine.arguments[5].split(separator: ",").map(String.init)
                                           : [R >= 2 ? "dory_attn_partial_rows" : "dory_attn_partial"]
let bytes = Double(Hkv * (L + R) * HD * 2 * 2)
// the rows' outputs from the partials (dory_attn_reduce's combine, in double)
func combine(_ nch: Int) -> [Double] {
  let o = po.contents().bindMemory(to: Float.self, capacity: R * Hq * maxN * HD)
  let ml = pml.contents().bindMemory(to: Float.self, capacity: R * Hq * maxN * 2)
  var out = [Double](repeating: 0, count: R * Hq * HD)
  for r in 0..<R { for h in 0..<Hq {
    let base = (r * Hq + h) * nch
    var M = -Double.infinity
    for c in 0..<nch { M = max(M, Double(ml[(base + c) * 2])) }
    var den = 0.0
    var num = [Double](repeating: 0, count: HD)
    for c in 0..<nch {
      let mc = Double(ml[(base + c) * 2]); if mc == -Double.infinity { continue }
      let w = exp(mc - M); den += w * Double(ml[(base + c) * 2 + 1])
      for d in 0..<HD { num[d] += w * Double(o[(base + c) * HD + d]) }
    }
    for d in 0..<HD { out[(r * Hq + h) * HD + d] = num[d] / den }
  } }
  return out
}
var ref: [Double]? = nil
for name in names {
let ps = try! dev.makeComputePipelineState(function: lib.makeFunction(name: name)!)
for N in Ns {
  var chunk = (L + R + N - 1) / N; chunk = (chunk + 15) / 16 * 16
  let nch = (L + R + chunk - 1) / chunk
  var best = 1e9
  for _ in 0..<5 {
    let cb = queue.makeCommandBuffer()!
    let e = cb.makeComputeCommandEncoder()!
    var p = P(R: UInt32(R), pos0: UInt32(L), Hq: UInt32(Hq), Hkv: UInt32(Hkv), rows: UInt32(rows), ring: 0, window: 0, chunk: UInt32(chunk), nchunks: UInt32(nch), k0: 0, scale: 1 / 16)
    e.setComputePipelineState(ps)
    e.setBuffer(qb, offset: 0, index: 0); e.setBuffer(kc, offset: 0, index: 1); e.setBuffer(vc, offset: 0, index: 2)
    e.setBuffer(po, offset: 0, index: 3); e.setBuffer(pml, offset: 0, index: 4); e.setBytes(&p, length: MemoryLayout<P>.size, index: 5)
    if R >= 2 { e.dispatchThreadgroups(MTLSize(width: nch, height: Hkv, depth: 1), threadsPerThreadgroup: MTLSize(width: 32 * G * R, height: 1, depth: 1)) }
    else { e.dispatchThreadgroups(MTLSize(width: nch, height: Hkv, depth: 1), threadsPerThreadgroup: MTLSize(width: 32 * G, height: 1, depth: 1)) }
    e.endEncoding(); cb.commit(); cb.waitUntilCompleted()
    if let err = cb.error { print("GPU error", err); exit(1) }
    best = min(best, (cb.gpuEndTime - cb.gpuStartTime) * 1000)
  }
  let out = combine(nch)
  var d = 0.0, bad = 0
  for x in out where !x.isFinite { bad += 1 }
  if let rf = ref { for i in 0..<out.count { d = max(d, abs(out[i] - rf[i])) } } else { ref = out }
  print(String(format: "%@ R %d L %d: %4d chunks of %6d keys: %7.3f ms  %6.1f GB/s  max|diff| %.2e%@", name, R, L, nch, chunk, best, bytes / best / 1e6, d, bad > 0 ? " NON-FINITE" : ""))
}
}
