// The page's own scripts (src/server/engine.html) in a Node vm with a stub DOM, on the stand-in (tools/engine_replay.py):
// the link's key, the live view, a switch and a stop from the page's own controls and the next run, a clean drop, a
// server gone and back with another key, no key, blocked site data, a stop with no next run. Node 18+ and python3: node tests/test_engine_page.mjs
import {spawn} from "node:child_process";
import {mkdtempSync, readFileSync} from "node:fs";
import {tmpdir} from "node:os";
import {dirname, join} from "node:path";
import {fileURLToPath} from "node:url";
import vm from "node:vm";

const root = join(dirname(fileURLToPath(import.meta.url)), "..");
const html = readFileSync(join(root, "src/server/engine.html"), "utf8");
const scripts = [...html.matchAll(/<script>([\s\S]*?)<\/script>/g)].map(m => m[1]);
const sleep = ms => new Promise(r => setTimeout(r, ms));
let failures = 0;
const check = (ok, what) => { if (!ok) { failures++; console.error("FAIL:", what); } };

async function until(f, ms, what) {
  const end = Date.now() + ms;
  for (;;) {
    let v;
    try { v = await f(); } catch { v = false; }
    if (v) return v;
    if (Date.now() > end) { check(false, `${what} (within ${ms} ms)`); return v; }
    await sleep(100);
  }
}

function finish() {
  console.log(`engine page: ${failures} failures`);
  process.exit(failures ? 1 : 0);
}

// One stand-in on `port`, with its key read back from --token-file
async function standIn(port, extra = []) {
  const dir = mkdtempSync(join(tmpdir(), "engine-page-")), tokenFile = join(dir, "token");
  const p = spawn("python3", [join(root, "tools/engine_replay.py"), "--port", String(port), "--speed", "4",
                              "--token-file", tokenFile, "--timing", ...extra],
                             {stdio: ["ignore", "pipe", "pipe"]});
  const startedAt = Date.now();
  let said = "", err = "";             // stdout holds the stand-in's progress line, stderr its --timing stages
  p.stdout.on("data", b => { said += b; });
  p.stderr.on("data", b => { err += b; process.stderr.write(b); });
  const terminate = () => p.kill();
  process.once("exit", terminate);
  p.once("close", () => process.off("exit", terminate));
  let spawnError;
  p.once("error", error => { spawnError = error; });
  const deadline = Date.now() + 15000;
  let token = "";
  while (!spawnError && p.exitCode === null && p.signalCode === null && Date.now() < deadline) {
    try { token = readFileSync(tokenFile, "utf8").trim(); } catch (error) {
      if (error.code !== "ENOENT") throw error;
    }
    if (token) break;
    await sleep(100);
  }
  if (!token) {
    const reason = spawnError?.message || (p.signalCode ? `signal ${p.signalCode}` :
      p.exitCode !== null ? `it exited with ${p.exitCode}` : "it is still running");
    check(false, `the stand-in on port ${port} never wrote its key after ${Date.now() - startedAt} ms: ${reason}` +
                 (said.trim() ? `; its output: ${said.trim().split("\n").slice(-3).join(" | ")}` : "; it printed nothing") +
                 (err.trim() ? `; its stderr: ${err.trim().split("\n").slice(-6).join(" | ")}` : "; it wrote nothing to stderr"));
    finish();
  }
  // A child that a signal ended has exitCode null and signalCode set, and its exit event has already fired by the time
  // the caller asks to stop it: waiting for another one never returns, and a test whose stand-in outlives a failed start
  // then hangs instead of reporting that failure (#1213).
  const gone = () => p.exitCode !== null || p.signalCode !== null;
  return {p, token, base: `http://127.0.0.1:${port}`,
          stop: () => new Promise(r => { if (gone()) return r(); p.once("exit", r); p.kill(); })};
}

// The page in a fresh vm: `hash` as its address's fragment; `storage` "ok" or "blocked" (every access throws)
function page(base, hash, storage = "ok") {
  const els = new Map();
  const el = id => {
    if (!els.has(id)) {
      const e = {id, textContent: "", innerHTML: "", className: "", disabled: false, onclick: null, onsubmit: null, value: "",
                 scrollTop: 0, clientHeight: 0, scrollHeight: 0, classList: {add() {}, remove() {}}};
      e.lastElementChild = {textContent: ""};
      e.hidden = false; e.focus = () => { e.focused = true; }; e.setAttribute = (k, v) => { e[k] = v; };
      e.contains = () => false;
      e.querySelector = () => e;
      els.set(id, e);
    }
    return els.get(id);
  };
  const store = () => {
    const m = new Map();
    return {getItem: k => (m.has(k) ? m.get(k) : null), setItem: (k, v) => m.set(k, String(v))};
  };
  const loc = {hash, pathname: "/engine", search: ""};
  const listeners = {};
  const ctx = {
    document: {getElementById: el, documentElement: {dataset: {}}, addEventListener(type, fn) { (listeners[type] ||= []).push(fn); }},
    location: loc,
    history: {replaceState(_s, _t, url) { loc.hash = url.includes("#") ? url.slice(url.indexOf("#")) : ""; }},
    matchMedia: () => ({matches: false, addEventListener() {}}),
    fetches: 0,
    fetch: (url, opts) => {
      ctx.fetches++; ctx.leak ||= hash.length > 3 && url.includes(hash.slice(3));
      if (opts && opts.method === "POST") ctx.posts.push({url, opts});
      return fetch(base + url, opts);
    },
    posts: [],
    performance, setTimeout, clearTimeout, setInterval, clearInterval, TextDecoder, AbortController, console,
  };
  // blocked site data: reading either area throws, as in a browser
  for (const area of ["localStorage", "sessionStorage"]) {
    const s = store();
    Object.defineProperty(ctx, area, {get: () => {
      if (storage === "blocked") throw new Error("SecurityError: site data is blocked");
      return s;
    }});
  }
  vm.createContext(ctx);
  for (const s of scripts) vm.runInContext(s, ctx);
  const view = () => vm.runInContext(`({link: S.link, run: S.status && S.status.run_id, state: S.status && S.status.state,
    model: S.status && S.status.model && S.status.model.id,
    reqs: [...S.reqs.keys()].sort().join(","), rates: S.rates.length,
    cancelled: S.done.filter(d => d.finish_reason === "cancelled").length, done: S.done.map(d => d.request_id), fetches,
    pill: document.getElementById("state").lastElementChild.textContent, msg: document.getElementById("msg").textContent})`, ctx);
  const tap = id => el(id).onclick();
  const switchTo = m => { el("swto").value = m; return el("sw").onsubmit({preventDefault() {}}); };
  const drawNow = () => vm.runInContext("draw()", ctx);
  const key = k => (listeners.keydown || []).forEach(f => f({key: k}));
  return {view, loc, ctx, el, tap, switchTo, drawNow, key};
}

const get = async (s, path) => (await fetch(s.base + path, {headers: {Authorization: "Bearer " + s.token}})).json();
const hook = (s, path, body) => fetch(s.base + path, {method: "POST", headers: {Authorization: "Bearer " + s.token,
  "Content-Type": "application/json"}, body: JSON.stringify(body)});

const port = 18900 + Math.floor(Math.random() * 90);
let s = await standIn(port);

// 1. the key from the link: live, the address bar cleaned, and the view follows the server
const a = page(s.base, "#t=" + s.token);
await until(() => a.view().link === "live" && a.view().run, 10000, "the page connects with the link's key");
check(a.loc.hash === "" && !a.ctx.leak, "the key is taken out of the address bar and never put in a URL");
const st = await get(s, "/engine/status");
check(a.view().run === st.run_id, "the page shows the server's run");
await until(async () => {
  const v = a.view(), now = await get(s, "/engine/status");
  return v.state === now.state && v.reqs === now.requests.map(r => r.request_id).sort().join(",");
}, 10000, "the page's state and requests follow the server's");
await until(() => /Prefix cache[\s\S]*In memory[\s\S]*On the SSD[\s\S]*of [\d.]+ GB it may hold/.test(a.el("cache").innerHTML), 10000,
            "the Memory card shows the RAM and SSD prefix caches");
console.log("link, key and live view ok");

// 1b. a page opened later reads the log lines the bus kept: the loading lines come first, flagged as earlier (they have
// no time), and none of the replayed events change the view; the filter keeps the warnings and brings the rest back
// (the stand-in logs its loading lines at its first tick, events 3 to 5: open the late page once they are logged, or it gets them live)
await until(async () => Number((await get(s, "/engine/status")).last_event_id.split(".").pop()) >= 5, 10000, "the stand-in has logged its loading lines");
const late = page(s.base, "#t=" + s.token);
await until(() => late.view().link === "live" && late.view().run, 10000, "a late page connects");
const logs = () => vm.runInContext("S.logs.map(l => ({t: l.t, level: l.level, message: l.message}))", late.ctx);
await until(() => logs().some(l => /^loading qwen/.test(l.message)), 10000, "the late page has the loading lines the server logged before it opened");
const bk = logs(), firstLive = bk.findIndex(l => l.t !== "earlier"), lastEarlier = bk.map(l => l.t).lastIndexOf("earlier");
check(bk.some(l => l.level === "warn" && l.t === "earlier" && /a made-up state message/.test(l.message)),
      `a replayed state event's message is a warning, as it is live (${JSON.stringify(bk.slice(0, 3))})`);
check(bk.filter(l => l.t === "earlier").length >= 4 && bk.some(l => l.level === "warn" && l.t === "earlier") &&
      bk.some(l => l.level === "debug" && l.t === "earlier") && (firstLive < 0 || lastEarlier < firstLive),
      `the replayed lines come first, flagged earlier, each level kept (${JSON.stringify(bk.slice(0, 4))})`);
await until(async () => late.view().state === (await get(s, "/engine/status")).state &&
                        late.view().reqs === (await get(s, "/engine/status")).requests.map(r => r.request_id).sort().join(","),
            10000, "the late page's state and requests are the server's, not the replayed events'");
late.drawNow();
check(/loading qwen/.test(late.el("logs").innerHTML) && /earlier/.test(late.el("logs").innerHTML), "the replayed lines are drawn");
late.tap("logf");
late.drawNow();
check(late.el("logs").innerHTML.includes("a made-up state message"), "the filter keeps a replayed state message");
check(late.el("logs").innerHTML.includes("a made-up warning") && late.el("logs").innerHTML.includes("a made-up error") &&
      !late.el("logs").innerHTML.includes("loading qwen") &&
      /^\d+ of \d+ lines$/.test(late.el("lognote").textContent) && late.el("logf").textContent === "Warnings and errors",
      `the filter keeps only warnings and errors and says how many (${late.el("lognote").textContent})`);
late.tap("logf");
late.drawNow();
check(late.el("logs").innerHTML.includes("loading qwen") && late.el("logf").textContent === "All lines", "the filter turns off");
console.log("log replay and filter ok");

// 1c. a bus that kept too little (8 events): the status is all the page gets, and it is live with no "caught up" line
const small = await standIn(port + 100, ["--buffer", "8"]);
await sleep(4000);  // (more than 8 events in)
const thin = page(small.base, "#t=" + small.token);
await until(() => thin.view().link === "live" && thin.view().run, 10000, "a page on a thin bus connects");
await sleep(1500);
check(thin.view().state && !vm.runInContext("S.logs.some(l => /caught up/.test(l.message))", thin.ctx) &&
      vm.runInContext("S.replay", thin.ctx) === 0, "a page on a bus that kept too little is live, with no reset line");
await small.stop();
console.log("thin bus ok");

// 1d. a finished request's row: a failed request shows its error and its numbers at once, any other opens on a click and
// closes on the next; the text is escaped
await until(() => a.view().done.includes("req-4"), 40000, "the stand-in's failing request (req-4) finishes");
a.drawNow();
const rows = () => a.el("reqs").innerHTML + a.el("done").innerHTML;
check(rows().includes("a made-up error: the prompt and 4,096 tokens do not fit") && rows().includes("waited 140 ms") &&
      /read [\d,]+ tokens in [\d.]+ (ms|s)/.test(rows()) && /wrote in/.test(rows()),
      "a failed request shows its error text and its waited, read and wrote times without a click");
const okId = vm.runInContext("S.done.find(d => !d.error).request_id", a.ctx), tapRow = id => a.el("done").onclick({target: {closest: () => ({dataset: {id}})}});
check(!rows().includes(`waited ${35 * Number(okId.slice(4))} ms`), `a request that finished well shows no detail until it is clicked (${okId})`);
tapRow(okId);
check(rows().includes(`waited ${35 * Number(okId.slice(4))} ms`), "a click on a finished request opens its times");
tapRow(okId);
check(!rows().includes(`waited ${35 * Number(okId.slice(4))} ms`), "a second click closes them");
tapRow(okId);  // (left open: the next run's reset closes it)
a.el("done").onclick({target: {closest: () => null}});  // (a click outside a finished row does nothing)
vm.runInContext(`handlers.request_finished({request_id: "req-x", finish_reason: "error", prompt_tokens: 10, cached_tokens: 10, completion_tokens: 0,
  queue_ms: 1, prefill_ms: 0, decode_ms: 0, tokens_per_second: 0, prefill_tokens_per_second: 0, error: "<b>bold</b> & <img src=x onerror=alert(1)>"}); draw()`, a.ctx);
check(rows().includes("&lt;b&gt;bold&lt;/b&gt;") && !rows().includes("<b>bold") && !rows().includes("<img") && rows().includes("prompt read from the cache"),
      "an error text is escaped, and a prompt served from the cache says so");
// the finished rows are written only when they change: a redraw every 500 ms would lose a click between press and
// release and clear a selection of the error text
a.drawNow();
let writes = 0, body = a.el("done").innerHTML;
Object.defineProperty(a.el("done"), "innerHTML", {get: () => body, set: v => { writes++; body = v; }});
a.drawNow(); a.drawNow();
check(writes === 0, `a redraw with no new finished request leaves the finished rows alone (${writes} writes)`);
tapRow(okId);
check(writes === 1 && !rows().includes(`waited ${35 * Number(okId.slice(4))} ms`), "closing a row writes them once");
tapRow(okId);  // (left open again: the next run's reset closes it)
vm.runInContext(`handlers.request_finished({request_id: "req-y", finish_reason: "error", prompt_tokens: 12010, cached_tokens: 0, completion_tokens: 0,
  queue_ms: 0, prefill_ms: 0, decode_ms: 0, tokens_per_second: 0, prefill_tokens_per_second: 0, error: "prompt (12010 tokens) exceeds the context window (8192)"}); draw()`, a.ctx);
check(rows().includes("12,010-token prompt not read") && !rows().includes("read 12,010 tokens"),
      "a request refused before its prompt was read says the prompt was not read (the real server's context error)");
vm.runInContext(`handlers.request_finished({request_id: "req-z", finish_reason: "error", prompt_tokens: 7777, cached_tokens: 0, completion_tokens: 200,
  queue_ms: 0, prefill_ms: 0, decode_ms: 900, tokens_per_second: 0, prefill_tokens_per_second: 0, error: "the engine child died"}); draw()`, a.ctx);
check(rows().includes("read 7,777 tokens") && !rows().includes("7,777-token prompt not read"),
      "an error that came after tokens were written still says the prompt was read (the Splash path leaves prefill_ms 0)");
console.log("finished-request detail ok");

// 2. the page's Switch: a model the server lacks is said so and nothing changes; one it has reads switching, then ready
// with it, and Switch waits meanwhile; each with this run's key, its run_id and a JSON body
await until(() => a.view().state === "ready" && !a.el("swgo").disabled, 10000, "Switch is ready");
const model0 = a.view().model;
await a.switchTo("nope");
check(/^switch: no model nope/.test(a.view().msg) && a.view().model === model0 && a.el("swto").value === "nope",
      `a missing model: said so, kept in the box, nothing switched (${a.view().msg})`);
const going = a.switchTo("qwen3.5-4b");
a.drawNow();  // before any event of the switch: the click alone holds Switch
check(a.view().state === "ready" && a.el("swgo").disabled, `Switch held from the click, before the server says switching (${a.view().state})`);
await until(() => a.el("swgo").disabled && a.el("swgo").textContent === "Switching…", 3000, "Switch waits while it switches");
await going;
await until(() => a.view().model === "qwen3.5-4b" && a.view().state === "ready" && !a.el("swgo").disabled, 15000,
            "the switch: ready with the new model");
check(a.view().msg === "" && a.el("swto").value === "", `a switch that took: no message, the box cleared (${a.view().msg})`);
const sent = a.ctx.posts.at(-1);
check(sent && sent.url === "/engine/model" && sent.opts.headers.Authorization === "Bearer " + s.token &&
      sent.opts.headers["Content-Type"] === "application/json" &&
      JSON.parse(sent.opts.body).run_id === a.view().run && JSON.parse(sent.opts.body).model === "qwen3.5-4b",
      "a switch carries the key, a JSON body and the run it saw");
console.log("switch ok");

// 2b. the picker beside it: the downloaded models, the loaded one first and marked, one that is too big for this Mac
// listed but not pickable; picking fills the box, and Switch sends that name
a.tap("swpick");
await until(() => a.el("swlist").innerHTML.includes("Qwen3.5-9B"), 5000, "the picker lists the downloaded models");
const list = a.el("swlist").innerHTML;
check(!a.el("swlist").hidden && a.el("swpick")["aria-expanded"] === "true", "the picker opens");
check(list.indexOf("Qwen3.5-4B") < list.indexOf("Qwen3.5-9B") && list.indexOf("Qwen3.5-9B") < list.indexOf("drex-v2.0-q8") &&
      /Qwen3\.5-4B[\s\S]*?Loaded now/.test(list) && list.includes("4.57 GB") && list.includes("MTP head") && list.includes("gguf Q8_0"),
      "the picker: the loaded model first and marked, then by fit, with size, precision and what each holds");
check(/class="opt" data-i="\d" disabled[^>]*title="\/Users\/me[^"]*drex[^"]*"/.test(list) &&
      list.includes("Too big for this Mac") && list.includes("Fits without its drafter"),
      "a model too big for this Mac is listed, marked, and cannot be picked");
check(/data-i="0" disabled/.test(list), "the loaded model cannot be picked (it is loaded)");
check(list.includes("fits planned for 38.6 GB"), "the picker says what the fits are planned for");
check(/Qwen3\.5-9B[\s\S]*?Downloads an update first/.test(list) && !/Qwen3\.5-4B[\s\S]{0,400}Downloads an update first[\s\S]*Qwen3\.5-9B/.test(list),
      "a model whose files are out of date says a switch downloads first");
a.key("Escape");
check(a.el("swlist").hidden && a.el("swpick")["aria-expanded"] === "false", "Escape closes the picker");
a.tap("swpick");
await until(() => a.el("swlist").innerHTML.includes("Qwen3.5-9B"), 5000, "the picker opens again");
const i9 = vm.runInContext(`listed.findIndex(m => m.name === "qwen3.5-9b:4bit")`, a.ctx);
a.el("swlist").onclick({target: {closest: () => ({disabled: false, dataset: {i: String(i9)}})}});
check(a.el("swto").value === "qwen3.5-9b:4bit" && a.el("swlist").hidden && a.el("swgo").focused, "picking a model fills the box and closes the picker");
await a.el("sw").onsubmit({preventDefault() {}});
await until(() => a.view().model === "qwen3.5-9b" && a.view().state === "ready" && !a.el("swgo").disabled, 15000,
            "the picked model loads");
check(JSON.parse(a.ctx.posts.at(-1).opts.body).model === "qwen3.5-9b:4bit", "Switch sends the picked name");
a.tap("swpick");
await until(() => /Qwen3\.5-9B[\s\S]*?Loaded now/.test(a.el("swlist").innerHTML), 5000, "the list follows the switch");
a.key("Escape");
console.log("picker ok");

// 3. the page's Stop while a request runs: one click only asks again, and lapses; two stop it. The request ends
// cancelled and the view reads stopping; the next run's reset then clears the old run's finished requests and rates
await until(() => a.view().reqs && a.view().rates > 8, 20000, "a request runs, its rate traced");
const run1 = a.view().run, posted = a.ctx.posts.length;
a.tap("stop");
check(a.el("stop").textContent === "Confirm stop" && a.ctx.posts.length === posted, "one click on Stop only asks again");
await sleep(3300);
check(a.el("stop").textContent === "Stop" && a.ctx.posts.length === posted && (await get(s, "/engine/status")).state !== "stopping",
      "an unconfirmed Stop lapses after 3 s and stops nothing");
a.tap("stop");
a.tap("stop");
await until(() => a.ctx.posts.length === posted + 1, 2000, "the confirmed Stop posts");
const stopSent = a.ctx.posts.at(-1);
check(stopSent && stopSent.url === "/engine/stop" && JSON.parse(stopSent.opts.body).run_id === run1 &&
      stopSent.opts.headers.Authorization === "Bearer " + s.token, "the stop carries the key and the run it saw");
await until(() => a.view().state === "stopping" && a.view().cancelled > 0, 10000, "the stop: stopping, a request cancelled");
const fresh = await until(async () => {
  for (const end = Date.now() + 20000; Date.now() < end; await sleep(20)) {
    const v = a.view();
    if (v.run && v.run !== run1) return v;  // (read at once: the trace gains a point every half second)
  }
}, 0, "the next run's reset replaces the view");
check(fresh && fresh.cancelled === 0 && fresh.rates <= 3,
      `the old run's finished requests and rates are cleared (${fresh && fresh.cancelled} cancelled, ${fresh && fresh.rates} points)`);
check(vm.runInContext("S.open.size", a.ctx) === 0, "a row left open in the old run is closed in the next (its ids come back)");
await until(() => a.view().link === "live", 10000, "the next run is live");
console.log("stop and new run ok");

// 4. a stream that ends cleanly while the run goes on: disconnected and retrying, then live again from the last event
// it read (a finished request neither lost nor replayed twice)
await until(() => a.view().done.length, 20000, "a request of the new run finishes");
const before = a.view().done;
await hook(s, "/_standin/drop", {run_id: a.view().run});
await until(() => a.view().link !== "live" && a.view().pill === "disconnected, retrying", 5000,
            "a clean end of the stream reads disconnected, retrying");
await until(() => a.view().link === "live" && a.view().run === fresh.run, 10000, "the view resumes the same run");
await sleep(1000);
check(new Set(a.view().done).size === a.view().done.length && before.every(id => a.view().done.includes(id)),
      `the resume kept what it had and replayed nothing twice (${before} -> ${a.view().done})`);

// the server goes away: the same; back with another key: no access, said why
await s.stop();
await until(() => a.view().link !== "live" && a.view().pill === "disconnected, retrying", 8000,
            "a server that went away reads disconnected, retrying");
s = await standIn(port);
await until(() => a.view().link === "refused" && /key/.test(a.view().msg), 15000,
            "a server with another key: no access, and the page says to open the new link");
console.log("disconnect and a new key ok");

// 5. no key at all; and blocked site data with a key still works
const b = page(s.base, "");
await until(() => b.view().link === "refused" && /link kapitsa serve prints/.test(b.view().msg), 8000,
            "no key: the page asks for the printed link");
const asked = b.view().fetches;
await sleep(4000);
check(b.view().fetches === asked, `refused: the page stops asking (${b.view().fetches - asked} more requests)`);
const c = page(s.base, "#t=" + s.token, "blocked");
await until(() => c.view().link === "live" && c.view().run, 10000, "blocked site data: the page still connects");
console.log("no key and blocked storage ok");

// 6. a fourth view while three stream (the stand-in's cap, as the server's): told so, and it keeps asking
const d = page(s.base, "#t=" + s.token), d2 = page(s.base, "#t=" + s.token);
await until(() => [c, d, d2].every(v => v.view().link === "live"), 10000, "three views stream");
const e = page(s.base, "#t=" + s.token);
await until(() => /too many engine views/.test(e.view().msg), 10000, "a fourth view: too many are open");
const asks = e.view().fetches;
await sleep(6000);
check(e.view().fetches > asks && e.view().link !== "refused", "the fourth view keeps asking");
console.log("too many views ok");

// 7. a stop no new run follows (no launchd): the view reads stopped, and stays so while nothing answers
await hook(s, "/_standin/stop", {run_id: d.view().run, then: "exit"});
await until(() => d.view().pill === "stopped", 10000, "a server that stopped reads stopped");
const traced = d.view().rates; await sleep(5000);
check(d.view().pill === "stopped" && d.view().rates <= traced, `still stopped, no new samples (${d.view().pill}, ${traced} -> ${d.view().rates})`);
console.log("stopped ok");

await s.stop();
finish();
