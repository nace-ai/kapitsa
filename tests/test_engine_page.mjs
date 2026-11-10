// The page's own scripts (src/server/engine.html) in a Node vm with a stub DOM, on the stand-in (tools/engine_replay.py):
// the link's key, the live view, a stop and the next run, a server gone and back with another key, no key, blocked
// site data. Node 18+ and python3: node tests/test_engine_page.mjs
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
async function standIn(port) {
  const dir = mkdtempSync(join(tmpdir(), "engine-page-")), tokenFile = join(dir, "token");
  const p = spawn("python3", [join(root, "tools/engine_replay.py"), "--port", String(port), "--speed", "4",
                              "--token-file", tokenFile], {stdio: ["ignore", "pipe", "inherit"]});
  await until(() => { try { return readFileSync(tokenFile, "utf8").trim(); } catch { return ""; } }, 15000, "the stand-in starts");
  const token = readFileSync(tokenFile, "utf8").trim();
  return {p, token, base: `http://127.0.0.1:${port}`,
          stop: () => new Promise(r => { p.once("exit", r); p.kill(); })};
}

// The page in a fresh vm: `hash` as its address's fragment; `storage` "ok" or "blocked" (every access throws)
function page(base, hash, storage = "ok") {
  const els = new Map();
  const el = id => {
    if (!els.has(id)) {
      const e = {id, textContent: "", innerHTML: "", className: "", disabled: false, onclick: null,
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
    fetch: (url, opts) => fetch(base + url, opts),
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
    reqs: [...S.reqs.keys()].sort().join(","), done: S.done.length,
    pill: document.getElementById("state").lastElementChild.textContent, msg: document.getElementById("msg").textContent})`, ctx);
  return {view, loc, ctx};
}

const get = async (s, path) => {
  const r = await fetch(s.base + path, {headers: {Authorization: "Bearer " + s.token}});
  return r.json();
};

const port = 18900 + Math.floor(Math.random() * 90);
let s = await standIn(port);

// 1. the key from the link: live, the address bar cleaned, and the view follows the server
const a = page(s.base, "#t=" + s.token);
await until(() => a.view().link === "live" && a.view().run, 10000, "the page connects with the link's key");
check(a.loc.hash === "", "the key is taken out of the address bar");
const st = await get(s, "/engine/status");
check(a.view().run === st.run_id, "the page shows the server's run");
await until(async () => {
  const v = a.view(), now = await get(s, "/engine/status");
  return v.state === now.state && v.reqs === now.requests.map(r => r.request_id).sort().join(",");
}, 10000, "the page's state and requests follow the server's");
console.log("link, key and live view ok");

// 2. a stop: the view reads stopping, then the next run's reset replaces it
const run1 = a.view().run;
const r = await fetch(s.base + "/_standin/stop", {method: "POST", headers: {Authorization: "Bearer " + s.token,
  "Content-Type": "application/json"}, body: JSON.stringify({run_id: run1})});
check(r.status === 202, `the stand-in takes the stop (${r.status})`);
await until(() => a.view().state === "stopping" || a.view().run !== run1, 10000, "the page sees the stop");
await until(() => a.view().run && a.view().run !== run1 && a.view().link === "live", 20000, "the next run's reset replaces the view");
check(a.view().done === 0 || a.view().run !== run1, "finished requests of the old run are cleared");
console.log("stop and new run ok");

// 3. the server goes away: disconnected and retrying, not a live view; back with another key: no access, said why
await s.stop();
await until(() => a.view().link !== "live" && /disconnected|stopped/.test(a.view().pill), 8000,
            "a server that went away reads disconnected");
s = await standIn(port);
await until(() => a.view().link === "refused" && /key/.test(a.view().msg), 15000,
            "a server with another key: no access, and the page says to open the new link");
console.log("disconnect and a new key ok");

// 4. no key at all; and blocked site data with a key still works
const b = page(s.base, "");
await until(() => b.view().link === "refused" && /link kapitsa serve prints/.test(b.view().msg), 8000,
            "no key: the page asks for the printed link");
const c = page(s.base, "#t=" + s.token, "blocked");
await until(() => c.view().link === "live" && c.view().run, 10000, "blocked site data: the page still connects");
console.log("no key and blocked storage ok");

await s.stop();
console.log(`engine page: ${failures} failures`);
process.exit(failures ? 1 : 0);
