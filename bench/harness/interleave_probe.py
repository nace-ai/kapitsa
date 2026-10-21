"""Interleave probe: a main agent and subagents that share one system prompt take turns on a running server
(NSESS sessions, 4 rounds, X-Session-Id per session); prints per-request cached tokens and the follow-up hit rate.
Usage: NSESS=6 python3 interleave_probe.py   (server on 127.0.0.1:8091, model qwen3.5-4b; hold the GPU lock)."""
import sys, json, time, urllib.request
base = "http://127.0.0.1:8091/v1/chat/completions"
SYS = "You are a coding agent. " + " ".join(f"Rule {i}: keep answers short and exact." for i in range(300))
def ask(msgs, sid):
    body = {"model":"qwen3.5-4b","messages":msgs,"max_tokens":24,"temperature":0,"stream":False,
            "chat_template_kwargs":{"enable_thinking":False}}
    t=time.time()
    req=urllib.request.Request(base,json.dumps(body).encode(),{"content-type":"application/json","x-session-id":sid})
    r=json.load(urllib.request.urlopen(req,timeout=300)); dt=time.time()-t
    u=r["usage"]; c=(u.get("prompt_tokens_details") or {}).get("cached_tokens",0)
    return r["choices"][0]["message"].get("content") or "", u["prompt_tokens"], c, dt
import os
N=int(os.environ.get("NSESS","3"))
tasks={("main" if i==0 else "sub%d"%i):"Task %d: work on module %d. "%(i,i) for i in range(N)}
hist={k:[{"role":"system","content":SYS}] for k in tasks}
rows=[]
for rnd in range(4):
    for k in tasks:
        hist[k].append({"role":"user","content":tasks[k]+(" round %d. "%rnd)+"Filler detail. "*(40*(rnd+1))})
        txt,p,c,dt=ask(hist[k],k)
        hist[k].append({"role":"assistant","content":txt})
        rows.append((rnd,k,p,c,round(dt,2)))
        print(rnd,k,"prompt",p,"cached",c,"hit %.2f"%(c/p),"ttft~%.2fs"%dt,flush=True)
fu=[r for r in rows if r[0]>0]
print("follow-up mean hit %.3f"%(sum(r[3]/r[2] for r in fu)/len(fu)), "mean latency %.2f"%(sum(r[4] for r in fu)/len(fu)))
