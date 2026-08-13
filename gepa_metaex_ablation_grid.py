# Multi-round ITERATIVE-REFINEMENT for all methods (GEPA-matched budget, fair comparison).
# Each round refines the CURRENT best skill against fresh rollout failures; gate keeps strict
# improvements. Per-method proposal style differs (skillopt/trace2skill/evoseed/explorer_na),
# but all use the same refine loop + budget. Robust to transient async crashes (per-eval retry).
# Usage: gepa_refine.py <bench> <method> <rounds>
import sys, json, dspy, re, random, time
import os as _envos
BASE=_envos.environ.get("OLLAMA_BASE","http://localhost:11434/v1"); NT={"reasoning":{"effort":"none"}}
task_lm=dspy.LM("openai/qwen3:8b",api_base=BASE,api_key="EMPTY",max_tokens=900,temperature=0.0,extra_body=NT)
gen_lm =dspy.LM("openai/qwen3:8b",api_base=BASE,api_key="EMPTY",max_tokens=1500,temperature=0.9,extra_body=NT)
dspy.configure(lm=task_lm)
from dspy.evaluate import Evaluate

bench=sys.argv[1]; MPOP=int(sys.argv[2]); MODE=sys.argv[3]; ROUNDS=int(sys.argv[4]) if len(sys.argv)>4 else 60
assert MODE in ("full","fail","success","both"), MODE
method="explorer_na"

if bench=="pupa":
    from datasets import load_dataset
    raw=list(load_dataset("Columbia-NLP/PUPA","pupa_new")["train"]); random.Random(42).shuffle(raw)
    def mk(e): return dspy.Example(user_query=e["user_query"],target=e.get("target_response","")or"",
        pii=[p for p in (e.get("pii_units","")or"").split("||") if p.strip()]).with_inputs("user_query")
    train=[mk(x) for x in raw[:150]]; val=[mk(x) for x in raw[150:200]]
    judge=dspy.LM("openai/qwen3:8b",api_base=BASE,api_key="EMPTY",max_tokens=200,temperature=0.0,extra_body=NT)
    def metric(g,p,trace=None,pred_name=None,pred_trace=None):
        r=getattr(p,'response','')or""
        if not r.strip(): q=0.0
        else:
            jr=judge(f"Rate if RESPONSE answers QUERY as well as REFERENCE. YES or NO.\nQUERY:{g.user_query[:400]}\nREFERENCE:{g.target[:400]}\nRESPONSE:{r[:400]}")[0].upper()
            q=1.0 if "YES" in jr else 0.0
        leak=sum(1 for u in g.pii if u.strip() and u.split("_")[-1].lower() in r.lower())/max(1,len(g.pii))
        return (q+(1-leak))/2
    SIG="user_query -> response"; INP="user_query"
    PHRASE="privacy-conscious query answering (answer helpfully WITHOUT leaking user PII)"
    BASEINSTR="Answer the user's query helpfully without leaking private PII."
elif bench=="hotpotqa":
    from dspy.datasets import HotPotQA
    ds=HotPotQA(train_seed=42,train_size=150,eval_seed=42,dev_size=50,test_size=0)
    train=[x.with_inputs('question') for x in ds.train]; val=[x.with_inputs('question') for x in ds.dev]
    def metric(g,p,trace=None,pred_name=None,pred_trace=None):
        a=(g.answer or"").lower().strip(); q=(getattr(p,'answer','')or"").lower().strip()
        return 1.0 if (a and (a in q or q in a)) else 0.0
    SIG="question -> answer"; INP="question"; PHRASE="multi-hop factual QA (exact short answer entity)"
    BASEINSTR="Given the question, produce the answer."
elif bench=="hover":
    from datasets import load_dataset
    lab=list(load_dataset("hover-nlp/hover", trust_remote_code=True)["validation"]); random.Random(42).shuffle(lab)
    def mk(e): return dspy.Example(claim=e["claim"], label="SUPPORTED" if e["label"]==1 else "NOT_SUPPORTED").with_inputs("claim")
    train=[mk(x) for x in lab[:150]]; val=[mk(x) for x in lab[150:200]]
    def metric(g,p,trace=None,pred_name=None,pred_trace=None):
        gg=g.label.upper(); pp=(getattr(p,'verdict','')or"").upper()
        gp="SUPPORTED" if ("SUPPORT" in pp and "NOT" not in pp) else ("NOT_SUPPORTED" if ("NOT" in pp or "REFUT" in pp) else pp)
        return 1.0 if gp==gg else 0.0
    SIG="claim -> verdict"; INP="claim"; PHRASE="multi-hop claim verification (SUPPORTED or NOT_SUPPORTED via multi-hop reasoning)"
    BASEINSTR="Given the claim, answer exactly SUPPORTED or NOT_SUPPORTED."
elif bench=="ifbench":
    from datasets import load_dataset
    def _w(t): return re.findall(r"\b[\w']+\b", t)
    def v_wcr(t,kw):
        n=len(_w(t)); lo=kw.get("min_words") or kw.get("N_start") or kw.get("min"); hi=kw.get("max_words") or kw.get("N_end") or kw.get("max")
        if lo is None and hi is None: return None
        ok=True
        if lo is not None: ok=ok and n>=int(lo)
        if hi is not None: ok=ok and n<=int(hi)
        return ok
    def v_uwc(t,kw):
        N=kw.get("N"); return None if N is None else len(set(w.lower() for w in _w(t)))>=int(N)
    def v_num(t,kw):
        N=kw.get("N"); 
        if N is None: return None
        return len(re.findall(r"\d+",t))>=int(N)
    def v_list(t,kw): return bool(re.search(r"(^|\n)\s*([-*]|\d+[.)])\s+", t))
    VERIF={"count:word_count_range":v_wcr,"count:unique_word_count":v_uwc,"count:numbers":v_num,"format:list":v_list}
    raw=[e for e in load_dataset("allenai/IFBench_test")["train"] if all(i in VERIF for i in e["instruction_id_list"])]
    random.Random(42).shuffle(raw)
    def mk(e): return dspy.Example(prompt=e["prompt"], iids=e["instruction_id_list"], kw=e["kwargs"]).with_inputs("prompt")
    ntr=min(60,len(raw)*3//5); train=[mk(x) for x in raw[:ntr]]; val=[mk(x) for x in raw[ntr:ntr+40]]
    def metric(g,p,trace=None,pred_name=None,pred_trace=None):
        resp=getattr(p,'response','')or""; res=[]
        for iid,kw in zip(g.iids,g.kw):
            f=VERIF.get(iid)
            if f is None: continue
            r=f(resp,kw or {})
            if r is not None: res.append(1.0 if r else 0.0)
        return sum(res)/len(res) if res else 0.0
    SIG="prompt -> response"; INP="prompt"; PHRASE="instruction following (obey all counting/format constraints exactly)"
    BASEINSTR="Answer the prompt while strictly obeying every stated constraint."
else: raise SystemExit("bench pupa|hotpotqa|hover|ifbench")

def evalp(prog, retries=3):
    for k in range(retries):
        try:
            r=Evaluate(devset=val,metric=metric,num_threads=4,display_progress=False)(prog)
            return float(getattr(r,"score",r))
        except Exception as e:
            sys.stderr.write(f"eval retry {k}: {str(e)[:60]}\n"); time.sleep(3)
    return 0.0
class M(dspy.Module):
    def __init__(self,instr=BASEINSTR): self.g=dspy.ChainOfThought(dspy.Signature(SIG,instr))
    def forward(self,**kw): return self.g(**kw)
def wrap(sk): return M(instr="Follow this skill.\n"+sk+"\n\n"+BASEINSTR)
def gen(p):
    for k in range(3):
        try: return re.sub(r"```[a-z]*","",gen_lm(p)[0]).strip()
        except Exception as e: sys.stderr.write(f"gen retry {k}\n"); time.sleep(3)
    return ""
def rollout(prog,n=20):
    L=[]; misses=[]; hits=[]
    for d in random.sample(train,min(n,len(train))):
        try: pr=prog(**{INP:getattr(d,INP)}); sc=metric(d,pr)
        except Exception: sc=0.0; pr=None
        gold=getattr(d,'answer',None) or getattr(d,'target','')[:30]
        line=f"[{sc:.2f}] gold={str(gold)[:22]} in={str(getattr(d,INP))[:70]}"
        L.append(line)
        if sc<0.5: misses.append(f"MISS gold={str(gold)[:22]} in={str(getattr(d,INP))[:60]}")
        else: hits.append(f"OK gold={str(gold)[:22]} in={str(getattr(d,INP))[:60]}")
    if MODE=="full":
        return "\n".join(L)
    if MODE=="fail":
        return "RECURRING FAILURES (fix these):\n"+"\n".join(misses[:12])
    if MODE=="success":
        return "WORKING CASES (reinforce what makes these pass):\n"+"\n".join(hits[:12])
    # both: high-density failures + a few success anchors
    return ("RECURRING FAILURES (fix these):\n"+"\n".join(misses[:10])
            +"\n\nALREADY WORKING (do NOT break these):\n"+"\n".join(hits[:6]))

# per-method refine prompt style
OPS=["ADD_RULE: add a general procedural rule for the top failure cluster",
     "SPECIALIZE: add a precise rule distinguishing the confusable cases",
     "ADD_SANITY_CHECK: add a verification step before committing the answer",
     "ADD_FORMATTING_RULE: enforce exact output form/surface normalization",
     "DECOMPOSE: add an explicit step-by-step procedure",
     "PRUNE: remove any rule that seems to hurt"]
LENSES=OPS
def propose(cur, fb, r):
    base=f"CURRENT skill:\n{cur}\n\n" if cur else ""
    lens=LENSES[(r-1)%len(LENSES)]
    if method=="skillopt":
        return gen(f"You are the SkillOpt OPTIMIZER for {PHRASE}. {base}Reflect on failures as a minibatch; propose consolidated add/delete/replace edits into ONE improved skill(200-400w).\n{fb}\nReturn only the skill.")
    if method=="trace2skill":
        return gen(f"You are Trace2Skill for {PHRASE}. {base}Distill trajectory-level lessons from these attempts into ONE improved skill(200-400w).\n{fb}\nReturn only the skill.")
    if method=="evoseed":
        return gen(f"EvoSkill seed-evolution for {PHRASE}. {base}Refine into ONE skill(200-400w), emphasize: {lens}.\n{fb}\nReturn only the skill.")
    # explorer_na (MetaEx)
    return gen(f"Anchor-free MetaEx optimizer for {PHRASE}. {base}REFINE (keep what works, {lens}). Output FULL improved skill(200-400w).\n{fb}\nReturn only the skill.")

best_skill=""; best_val=evalp(M()); queries=len(val)
curve=[{"round":0,"queries":queries,"val":best_val}]
for r in range(1,ROUNDS+1):
    cur=wrap(best_skill) if best_skill else M()
    fb=rollout(cur); queries+=20
    # MetaEx op-menu population: MPOP variants this round, gate keeps best strict-improvement
    import random as _rnd
    ops_this=_rnd.sample(OPS, min(MPOP,len(OPS)))
    for op in ops_this:
        base=f"CURRENT skill:\n{best_skill}\n\n" if best_skill else ""
        cand=gen(f"Anchor-free MetaEx optimizer for {PHRASE}. {base}Apply operation [{op}]. Output FULL improved skill(200-400w) built to fix the failures.\n{fb}\nReturn only the skill.")
        if not cand: continue
        v=evalp(wrap(cand)); queries+=len(val)
        if v>best_val: best_val=v; best_skill=cand
    if r%5==0 or v>curve[-1]["val"]:
        curve.append({"round":r,"queries":queries,"val":best_val})
        print(f"[{method}/{bench}] round {r}: queries={queries} best_val={best_val}", flush=True)
import os as _os
SKILLS_DIR=_envos.environ.get("SKILLS_DIR","./skills"); _os.makedirs(SKILLS_DIR,exist_ok=True)
open(f"{SKILLS_DIR}/{bench}_metaex_M{MPOP}_{MODE}.md","w").write(best_skill or "")
print(f"SAVED_SKILL {SKILLS_DIR}/{bench}_metaex_M{MPOP}_{MODE}.md ({len(best_skill or '')} chars, val={best_val})")
print(f"RESULT_ABL bench={bench} MPOP={MPOP} MODE={MODE} "+json.dumps(curve))
