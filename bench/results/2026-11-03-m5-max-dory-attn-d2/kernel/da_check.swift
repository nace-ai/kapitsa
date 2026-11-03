// da_check: Dory's prefill attention (one full-attention layer: 16 query heads over 4 KV heads, head dim 256) after
// pos0 cached positions, T new rows: GPU time and TFLOP/s of each Dory-layout kernel, max |diff| against the first,
// and Qwen's qattn_t_g4d2s8 on its own layout (same shape) for scale.
// usage: da_check TENSOR_SOURCE KERNEL[,KERNEL...] [T] [POS0,...]
import Foundation
import Metal

let dev = MTLCreateSystemDefaultDevice()!
let queue = dev.makeCommandQueue()!
let o = MTLCompileOptions()
o.languageVersion = .version4_0
let lib = try! dev.makeLibrary(source: try! String(contentsOfFile: CommandLine.arguments[1], encoding: .utf8), options: o)
let names = CommandLine.arguments[2].split(separator: ",").map(String.init)
let T = CommandLine.arguments.count > 3 ? Int(CommandLine.arguments[3])! : 2048
let pos0s = CommandLine.arguments.count > 4 ? CommandLine.arguments[4].split(separator: ",").map { Int($0)! }
                                           : [14336, 63488, 129024, 253952]
struct DaP { var T, pos0, Hq, Hkv, rows, ring, window: UInt32; var scale: Float }
struct QaP { var T, Hq, Hkv, pos0, max_seq: UInt32; var scale: Float }
let Hq = 16, Hkv = 4, HD = 256, G = 4
// WINDOW=w: a sliding layer's window; RING=r: its cache a ring of r slots (key k at slot k % r)
let window = Int(ProcessInfo.processInfo.environment["WINDOW"] ?? "0")!
let ringRows = Int(ProcessInfo.processInfo.environment["RING"] ?? "0")!
func fillH(_ b: MTLBuffer, _ n: Int) {
  let p = b.contents().bindMemory(to: Float16.self, capacity: n)
  var tab = [Float16](repeating: 0, count: 1 << 20)
  for i in 0..<tab.count { tab[i] = Float16(Float.random(in: -1...1)) }
  var i = 0
  while i < n { let c = min(tab.count, n - i); tab.withUnsafeBufferPointer { memcpy(p + i, $0.baseAddress!, c * 2) }; i += c }
}
for pos0 in pos0s {
  let rows = ringRows > 0 ? ringRows : pos0 + T + 64
  let qb = dev.makeBuffer(length: T * Hq * HD * 4, options: .storageModeShared)!
  let qp = qb.contents().bindMemory(to: Float.self, capacity: T * Hq * HD)
  for i in 0..<(T * Hq * HD) { qp[i] = Float.random(in: -2...2) }
  // Dory: the cache holds [0, pos0), the chunk's own K / V [pos0, pos0 + T); Qwen's cache holds all of it
  let kc = dev.makeBuffer(length: Hkv * rows * HD * 2, options: .storageModeShared)!
  let vc = dev.makeBuffer(length: Hkv * rows * HD * 2, options: .storageModeShared)!
  fillH(kc, Hkv * rows * HD); fillH(vc, Hkv * rows * HD)
  let ck = dev.makeBuffer(length: Hkv * T * HD * 2, options: .storageModeShared)!
  let cv = dev.makeBuffer(length: Hkv * T * HD * 2, options: .storageModeShared)!
  let kp = kc.contents().bindMemory(to: Float16.self, capacity: Hkv * rows * HD)
  let vp = vc.contents().bindMemory(to: Float16.self, capacity: Hkv * rows * HD)
  let ckp = ck.contents().bindMemory(to: Float16.self, capacity: Hkv * T * HD)
  let cvp = cv.contents().bindMemory(to: Float16.self, capacity: Hkv * T * HD)
  if ringRows > 0 { fillH(ck, Hkv * T * HD); fillH(cv, Hkv * T * HD) } else {
    for h in 0..<Hkv {
      memcpy(ckp + h * T * HD, kp + (h * rows + pos0) * HD, T * HD * 2)
      memcpy(cvp + h * T * HD, vp + (h * rows + pos0) * HD, T * HD * 2)
    }
  }
  let qg = dev.makeBuffer(length: T * Hq * 2 * HD * 4, options: .storageModeShared)!
  memset(qg.contents(), 0x7f, T * Hq * 2 * HD * 4)   // a huge gate: sigmoid 1
  var line = String(format: "T %d after %6d:", T, pos0)
  let flops = window > 0 ? 4.0 * Double(T) * Double(window) * Double(Hq * HD)
                         : 4.0 * Double(T) * (Double(pos0) + Double(T) / 2) * Double(Hq * HD)
  var ref: UnsafeMutablePointer<Float>? = nil
  for n in names {
    let ps = try! dev.makeComputePipelineState(function: lib.makeFunction(name: n)!)
    let ob = dev.makeBuffer(length: T * Hq * HD * 4, options: .storageModeShared)!
    var best = 1e9
    for _ in 0..<3 {
      let cb = queue.makeCommandBuffer()!
      let e = cb.makeComputeCommandEncoder()!
      e.setComputePipelineState(ps)
      if n.hasPrefix("qattn") {
        var p = QaP(T: UInt32(T), Hq: UInt32(Hq), Hkv: UInt32(Hkv), pos0: UInt32(pos0), max_seq: UInt32(rows), scale: 1 / Float(HD).squareRoot())
        e.setBuffer(qb, offset: 0, index: 0); e.setBuffer(kc, offset: 0, index: 1); e.setBuffer(vc, offset: 0, index: 2)
        e.setBuffer(qg, offset: 0, index: 3); e.setBuffer(ob, offset: 0, index: 4); e.setBytes(&p, length: MemoryLayout<QaP>.size, index: 5)
        e.dispatchThreadgroups(MTLSize(width: (T * G + 63) / 64, height: Hkv, depth: 1), threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
      } else {
        var p = DaP(T: UInt32(T), pos0: UInt32(pos0), Hq: UInt32(Hq), Hkv: UInt32(Hkv), rows: UInt32(rows), ring: ringRows > 0 ? 1 : 0, window: UInt32(window), scale: 1 / Float(HD).squareRoot())
        e.setBuffer(qb, offset: 0, index: 0); e.setBuffer(kc, offset: 0, index: 1); e.setBuffer(vc, offset: 0, index: 2)
        e.setBuffer(ck, offset: 0, index: 3); e.setBuffer(cv, offset: 0, index: 4); e.setBuffer(ob, offset: 0, index: 5)
        e.setBytes(&p, length: MemoryLayout<DaP>.size, index: 6)
        // <name>: dory_attn_t_g4s<NS>k16 (TT NS = 4 NS tokens a threadgroup), dory_attn_td2_* (16 tokens, 256 threads)
        let tpt = n.contains("s16") ? 32 : n.contains("td2") && n.hasSuffix("s4") ? 8 : n.contains("td2") ? 16 : (n.contains("s4k") ? 16 : 32)
        let thr = n.contains("s16") ? 512 : n.hasSuffix("s4") || n.contains("s4k") ? 128 : 256
        e.dispatchThreadgroups(MTLSize(width: (T + tpt - 1) / tpt, height: Hkv, depth: 1), threadsPerThreadgroup: MTLSize(width: thr, height: 1, depth: 1))
      }
      e.endEncoding(); cb.commit(); cb.waitUntilCompleted()
      if let err = cb.error { print("GPU error", err); exit(1) }
      best = min(best, (cb.gpuEndTime - cb.gpuStartTime) * 1000)
    }
    let out = ob.contents().bindMemory(to: Float.self, capacity: T * Hq * HD)
    var d: Float = 0, bad = 0
    for i in 0..<(T * Hq * HD) where !out[i].isFinite { bad += 1 }
    if bad > 0 { print("\(n): \(bad) non-finite outputs") }
    if let r = ref { for i in 0..<(T * Hq * HD) { d = max(d, abs(r[i] - out[i])) } } else { ref = out }
    // against a double-precision softmax over each sampled row's keys (16 tokens spread over the chunk, every head)
    var e64 = 0.0
    if !n.hasPrefix("qattn") {
      for si in 0..<16 {
        let t = (T - 1) * si / 15
        let lim = pos0 + t, lo = window > 0 && lim + 1 > window ? lim + 1 - window : 0
        for h in 0..<Hq {
          let kvh = h / G
          var sc = [Double](), mx = -1e300
          for key in lo...lim {
            let kr = key >= pos0 ? ckp + (kvh * T + key - pos0) * HD : kp + (kvh * rows + (ringRows > 0 ? key % rows : key)) * HD
            var dot = 0.0
            for d in 0..<HD { dot += Double(Float16(qp[(t * Hq + h) * HD + d] / Float(HD).squareRoot())) * Double(kr[d]) }
            sc.append(dot); mx = max(mx, dot)
          }
          var den = 0.0
          for i in 0..<sc.count { sc[i] = exp(sc[i] - mx); den += sc[i] }
          for d in 0..<HD {
            var acc = 0.0
            for (i, key) in (lo...lim).enumerated() {
              let vr = key >= pos0 ? cvp + (kvh * T + key - pos0) * HD : vp + (kvh * rows + (ringRows > 0 ? key % rows : key)) * HD
              acc += sc[i] * Double(vr[d])
            }
            e64 = max(e64, abs(acc / den - Double(out[(t * Hq + h) * HD + d])))
          }
        }
      }
    }
    line += String(format: " | %@ %7.1f ms %5.1f TF/s%@%@", n, best, flops / best / 1e9, ref == out ? "" : String(format: " (diff %.1e)", d),
                   n.hasPrefix("qattn") ? "" : String(format: " fp64 err %.2e", e64))
  }
  print(line)
}
