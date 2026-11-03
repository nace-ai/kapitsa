// fd_check: Flint's decode / verify attention (one full-attention layer: 32 query heads over 4 KV heads, head dim 128)
// of R rows over L cached positions, split into N chunks per KV head: GPU time and the cache read's bandwidth.
// usage: fd_check FLINT_SOURCE TENSOR_SOURCE R L N[,N...] [KERNEL,...]  (the first kernel the reference of the combined outputs)
import Foundation
import Metal
let dev = MTLCreateSystemDefaultDevice()!
let queue = dev.makeCommandQueue()!
let lib = try! dev.makeLibrary(source: try! String(contentsOfFile: CommandLine.arguments[1], encoding: .utf8), options: MTLCompileOptions())
let o4 = MTLCompileOptions(); o4.languageVersion = .version4_0
let tlib = try! dev.makeLibrary(source: try! String(contentsOfFile: CommandLine.arguments[2], encoding: .utf8), options: o4)
let R = Int(CommandLine.arguments[3])!, L = Int(CommandLine.arguments[4])!
let Ns = CommandLine.arguments[5].split(separator: ",").map { Int($0)! }
struct P { var R, pos0, Hq, Hkv, rows, ring, window, k0, chunk, nchunks: UInt32; var scale: Float }
let Hq = 32, Hkv = 4, HD = 128, G = 8
// WINDOW=w: a sliding layer's window (keys (pos - w, pos]); RING=n: its cache a ring of n slots (key k at slot k % n)
let window = Int(ProcessInfo.processInfo.environment["WINDOW"] ?? "0")!
let ringN = Int(ProcessInfo.processInfo.environment["RING"] ?? "0")!
let rows = ringN > 0 ? ringN : L + R + 16
let k0 = window > 0 && L + 1 > window ? L + 1 - window : 0
let slot = { (k: Int) -> Int in ringN > 0 ? k % ringN : k }
let qb = dev.makeBuffer(length: R * Hq * HD * 4, options: .storageModeShared)!
let qp = qb.contents().bindMemory(to: Float.self, capacity: R * Hq * HD)
for i in 0..<(R * Hq * HD) { qp[i] = Float.random(in: -2...2) }
let kc = dev.makeBuffer(length: Hkv * rows * HD * 2, options: .storageModeShared)!
let vc = dev.makeBuffer(length: Hkv * rows * HD * 2, options: .storageModeShared)!
for b in [kc, vc] {
  let p = b.contents().bindMemory(to: Float16.self, capacity: Hkv * rows * HD)
  var tab = [Float16](repeating: 0, count: 1 << 20)
  for i in 0..<tab.count { tab[i] = Float16(Float.random(in: -1...1)) }
  var i = 0
  while i < Hkv * rows * HD { let c = min(tab.count, Hkv * rows * HD - i); tab.withUnsafeBufferPointer { memcpy(p + i, $0.baseAddress!, c * 2) }; i += c }
}
let maxN = Ns.max()!
let po = dev.makeBuffer(length: R * Hq * maxN * HD * 4, options: .storageModeShared)!
let pml = dev.makeBuffer(length: R * Hq * maxN * 2 * 4, options: .storageModeShared)!
let names = CommandLine.arguments.count > 6 ? CommandLine.arguments[6].split(separator: ",").map(String.init)
                                           : [R >= 2 ? "flint_attn_partial_rows" : "flint_attn_partial"]
let bytes = Double(Hkv * (L + R - (window > 0 && L + 1 > window ? L + 1 - window : 0)) * HD * 2 * 2)
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
// a double-precision softmax for rows 0..R-1 x heads 0, 9, 18, 31: keys [0, L + r]
let kq = kc.contents().bindMemory(to: Float16.self, capacity: Hkv * rows * HD)
let vq = vc.contents().bindMemory(to: Float16.self, capacity: Hkv * rows * HD)
let H64 = [0, 9, 18, 31]
var exact = [Double](repeating: 0, count: R * Hq * HD)
for r in 0..<R { for h in H64 {
  let kvh = h / G
  let lo = window > 0 && L + r + 1 > window ? L + r + 1 - window : 0
  var sc = [Double](repeating: 0, count: L + r + 1), mx = -Double.infinity
  for key in lo...(L + r) {
    var d = 0.0
    for x in 0..<HD { d += Double(qp[(r * Hq + h) * HD + x]) * (1 / 11.3137) * Double(kq[(kvh * rows + slot(key)) * HD + x]) }
    sc[key] = d; mx = max(mx, d)
  }
  var den = 0.0
  for key in lo...(L + r) { sc[key] = exp(sc[key] - mx); den += sc[key] }
  for x in 0..<HD {
    var a = 0.0
    for key in lo...(L + r) { a += sc[key] * Double(vq[(kvh * rows + slot(key)) * HD + x]) }
    exact[(r * Hq + h) * HD + x] = a / den
  }
} }
var ref: [Double]? = nil
for name in names {
let ps = try! dev.makeComputePipelineState(function: (tlib.makeFunction(name: name) ?? lib.makeFunction(name: name))!)
for N in Ns {
  var chunk = (L + R - k0 + N - 1) / N; chunk = max(64, (chunk + 31) / 32 * 32)
  let nch = (L + R - k0 + chunk - 1) / chunk
  var best = 1e9
  for _ in 0..<5 {
    let cb = queue.makeCommandBuffer()!
    let e = cb.makeComputeCommandEncoder()!
    var p = P(R: UInt32(R), pos0: UInt32(L), Hq: UInt32(Hq), Hkv: UInt32(Hkv), rows: UInt32(rows), ring: ringN > 0 ? 1 : 0, window: UInt32(window), k0: UInt32(k0), chunk: UInt32(chunk), nchunks: UInt32(nch), scale: 1 / 11.3137)
    e.setComputePipelineState(ps)
    e.setBuffer(qb, offset: 0, index: 0); e.setBuffer(kc, offset: 0, index: 1); e.setBuffer(vc, offset: 0, index: 2)
    e.setBuffer(po, offset: 0, index: 3); e.setBuffer(pml, offset: 0, index: 4); e.setBytes(&p, length: MemoryLayout<P>.size, index: 5)
    // flint_attn_partial*: a threadgroup per (chunk, kv head, row), buffer 6 a tree (unused); _rows*: every row in one
    if name.hasPrefix("flint_attn_t_rows") { e.dispatchThreadgroups(MTLSize(width: nch, height: Hkv, depth: 1), threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1)) }
    else if name.contains("rows") { e.dispatchThreadgroups(MTLSize(width: nch, height: Hkv, depth: 1), threadsPerThreadgroup: MTLSize(width: 32 * G, height: 1, depth: 1)) }
    else { e.setBuffer(pml, offset: 0, index: 6); e.dispatchThreadgroups(MTLSize(width: nch, height: Hkv, depth: R), threadsPerThreadgroup: MTLSize(width: 32 * G, height: 1, depth: 1)) }
    e.endEncoding(); cb.commit(); cb.waitUntilCompleted()
    if let err = cb.error { print("GPU error", err); exit(1) }
    best = min(best, (cb.gpuEndTime - cb.gpuStartTime) * 1000)
  }
  let out = combine(nch)
  var d = 0.0, bad = 0
  for x in out where !x.isFinite { bad += 1 }
  if let rf = ref { for i in 0..<out.count { d = max(d, abs(out[i] - rf[i])) } } else { ref = out }
  var e64 = 0.0
  for r in 0..<R { for h in H64 { for x in 0..<HD { e64 = max(e64, abs(out[(r * Hq + h) * HD + x] - exact[(r * Hq + h) * HD + x])) } } }
  print(String(format: "%@ R %d L %d: %4d chunks of %6d keys: %7.3f ms  %6.1f GB/s  max|diff| %.2e  fp64 err %.2e%@", name, R, L, nch, chunk, best, bytes / best / 1e6, d, e64, bad > 0 ? " NON-FINITE" : ""))
}
}
