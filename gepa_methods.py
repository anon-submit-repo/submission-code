import os, sys, dspy, re
from dspy.datasets import HotPotQA
from dspy.evaluate import Evaluate

import os as _envos
BASE=_envos.environ.get("OLLAMA_BASE","http://localhost:11434/v1"); NT={"reasoning":{"effort":"none"}}
task_lm=dspy.LM("openai/qwen3:8b",api_base=BASE,api_key="EMPTY",max_tokens=1000,temperature=0.0,extra_body=NT)
gen_lm =dspy.LM("openai/qwen3:8b",api_base=BASE,api_key="EMPTY",max_tokens=1500,temperature=0.7,extra_body=NT)
dspy.configure(lm=task_lm)

ds=HotPotQA(train_seed=42,train_size=150,eval_seed=42,dev_size=50,test_size=0)
train=[x.with_inputs('question') for x in ds.train]
val  =[x.with_inputs('question') for x in ds.dev]

def metric(gold,pred,trace=None,pred_name=None,pred_trace=None):
    g=(gold.answer or "").lower().strip(); p=(getattr(pred,'answer','') or "").lower().strip()
    return 1.0 if (g and (g in p or p in g)) else 0.0

PHRASE="multi-hop factual QA (find the exact short answer entity via reasoning over evidence, strict short-answer match)"
def gen(prompt): return re.sub(r"```[a-z]*","",gen_lm(prompt)[0]).strip()

def rollout_fails(prog, n=30):
    lines=[]
    for d in train[:n]:
        pred=prog(question=d.question); ok=metric(d,pred)
        lines.append(f"[{'OK' if ok else 'MISS'}] gold={d.answer} | pred={getattr(pred,'answer','')[:25]} | Q={d.question[:90]}")
    return "\n".join(lines)

def make_skill(mode, fb):
    if mode=="oneshot":
        ex="\n".join(f"- {d.question[:110]} (ans: {d.answer})" for d in train[:6])
        return gen(f"Write ONE compact GENERAL procedural skill (200-400w) for {PHRASE}. One shot, no gate. EXAMPLES:\n{ex}\nReturn only the skill.")
    if mode=="trace2skill":
        return gen(f"You are Trace2Skill: distill trajectory-level lessons from the model's OWN attempts on {PHRASE} into ONE reusable skill (200-400w). NO gate. Study WRONG vs CORRECT:\n{fb}\nReturn only the skill.")
    if mode=="skillopt":
        return gen(f"You are the SkillOpt OPTIMIZER for {PHRASE}. Reflect on FAILED attempts as a minibatch, propose consolidated add/delete/replace guidance into ONE improved skill (200-400w). Strict gate judges it.\nATTEMPTS:\n{fb}\nReturn only the skill.")
    if mode=="evoseed":
        lenses=["locate exact answer entity via multi-hop reasoning","reject decoys from adjacent evidence","chain facts across hops","answer boundary: bare entity"]
        return [gen(f"Write ONE compact (200-350w) GENERAL skill for {PHRASE}. Emphasize: {l}. Return only the skill.") for l in lenses]
    if mode=="explorer_na":
        ops=["ADD_RULE for top failure cluster","SPECIALIZE distinguishing confusable cases","ADD_SANITY_CHECK before committing","ADD_FORMATTING_RULE exact answer form"]
        return [gen(f"You are the OPTIMIZER in an ANCHOR-FREE loop for {PHRASE}. WRITE A FULL SKILL FROM SCRATCH (200-400w) whose organizing principle is: {op}. Fix these failures:\n{fb}\nReturn only the full skill.") for op in ops]

class QA(dspy.Module):
    def __init__(self,instr="Given the question, produce the answer."):
        self.gen=dspy.ChainOfThought(dspy.Signature("question -> answer", instr))
    def forward(self,question): return self.gen(question=question)

def evalp(prog): return Evaluate(devset=val,metric=metric,num_threads=4,display_progress=False)(prog)

mode=sys.argv[1]
base=QA()
ns_val=evalp(base)  # no-skill val (for gate)
fb=rollout_fails(base)
cand=make_skill(mode,fb)
def wrap(skill): return QA(instr="Follow this procedural skill when answering.\n"+skill+"\n\nGiven the question, produce the answer.")
if mode in ("oneshot","trace2skill"):      # no gate
    prog=wrap(cand); sc=evalp(prog)
elif mode=="skillopt":                       # single, strict gate vs no-skill
    sc=evalp(wrap(cand)); sc = sc if sc>ns_val else ns_val
else:                                         # evoseed/explorer_na: pick best > no-skill
    scs=[evalp(wrap(c)) for c in cand]; best=max(scs); sc=best if best>ns_val else ns_val
print(f"RESULT {mode} HotpotQA val: {sc} (no_skill {ns_val})")
