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

// One stand-in on `port`, with its key read back from --token-file
async function standIn(port, extra = []) {
  const dir = mkdtempSync(join(tmpdir(), "engine-page-")), tokenFile = join(dir, "token");
  const p = spawn("python3", [join(root, "tools/engine_replay.py"), "--port", String(port), "--speed", "4",
                              "--token-file", tokenFile, ...extra], {stdio: ["ignore", "pipe", "inherit"]});
  await until(() => { try { return readFileSync(tokenFile, "utf8").trim(); } catch { return ""; } }, 15000, "the stand-in starts");
  const token = readFileSync(tokenFile, "utf8").trim();
  return {p, token, base: `http://127.0.0.1:${port}`,
          stop: () => new Promise(r => { if (p.exitCode !== null) return r(); p.once("exit", r); p.kill(); })};
}

// The page in a fresh vm: `hash` as its address's fragment; `storage` "ok" or "blocked" (every access throws)
function page(base, hash, storage = "ok") {
  const els = new Map();
  const el = id => {
    if (!els.has(id)) {
      const e = {id, textContent: "", innerHTML: "", className: "", disabled: false, onclick: null, onsubmit: null, value: "",
                 scrollTop: 0, clientHeight: 0, scrollHeight: 0, classList: {add() {}, remove() {}}};
      e.lastElementChild = {textContent: ""};
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
  const ctx = {
    document: {getElementById: el, documentElement: {dataset: {}}},
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
  return {view, loc, ctx, el, tap, switchTo, drawNow};
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
console.log("link, key and live view ok");

// 1b. a page opened later reads the log lines the bus kept: the loading lines come first, flagged as earlier (they have
// no time), and none of the replayed events change the view; the filter keeps the warnings and brings the rest back
const late = page(s.base, "#t=" + s.token);
await until(() => late.view().link === "live" && late.view().run, 10000, "a late page connects");
const logs = () => vm.runInContext("S.logs.map(l => ({t: l.t, level: l.level, message: l.message}))", late.ctx);
await until(() => logs().some(l => /^loading qwen/.test(l.message)), 10000, "the late page has the loading lines the server logged before it opened");
const bk = logs(), firstLive = bk.findIndex(l => l.t !== "earlier"), lastEarlier = bk.map(l => l.t).lastIndexOf("earlier");
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
console.log(`engine page: ${failures} failures`);
process.exit(failures ? 1 : 0);
