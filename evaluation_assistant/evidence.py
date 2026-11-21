"""The only layer allowed to read experiment artifacts. Aggregate tables only."""
import csv
import io
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "milestone2"
FILES = {
    "ranking":"author_reference/results.csv", "paired":"author_reference/paired.json",
    "coverage":"author_reference/coverage.csv", "development":"feedback_dev/summary.csv",
    "settings":"feedback_dev/selection.json", "thresholds":"feedback_dev/development_thresholds.csv",
    "holdout":"feedback_test/results.csv",
}
TOOLS = {
    "compare_models":"Matched holdout: frozen model vs authors’ reference, primary AP, secondary AUROC, paired interval; replacement decision.",
    "inspect_fusion":"Development only: ah alone vs concatenation vs attention under SMOTE.",
    "inspect_losses":"Development only: concatenation with SMOTE, weighted BCE, or selected focal loss.",
    "inspect_demographics":"Development only: voice alone, voice plus age/sex, age/sex alone.",
    "inspect_threshold":"Frozen model: both thresholds, development target vs actual holdout sensitivity and specificity.",
    "inspect_cohort":"Development and holdout sessions and positive speakers; uncertainty context.",
    "lookup_run":"Frozen model representation, objective, focal hyperparameters, seeds and primary selection metric.",
}

def source(k):
    return (ROOT/FILES[k]).read_text(encoding="utf-8-sig")

def rows(k):
    return list(csv.DictReader(io.StringIO(source(k))))

def one(data, **where):
    found=[r for r in data if all(r[k]==v for k,v in where.items())]
    if len(found)!=1:
        raise ValueError("Missing or ambiguous result row: "+str(where))
    return found[0]

def n(v):
    a=float(v)
    if not math.isfinite(a):
        raise ValueError("Nonfinite source value")
    return a

def card(tool, scope, title, body, citations):
    return dict(tool=tool, scope=scope, title=title, body=body,
                sources=[dict(id=k,file=FILES[k],locator=loc) for k,loc in citations])

def run(tool):
    if tool not in TOOLS:
        raise ValueError("Unknown tool")
    cfg=json.loads(source("settings"))
    if (cfg["smoke"] or cfg["primary_metric"]!="mean fold average precision"
        or (cfg["representation"],cfg["loss"],cfg["alpha"],cfg["gamma"])!=("concat","focal",.9,2)):
        raise ValueError("This evidence catalog is for the frozen concat-focal run.")
    if tool=="compare_models":
        data=rows("ranking")
        a,b=[one(data,model=k) for k in ("concat_focal","author_reference")]
        pair=json.loads(source("paired"))
        delta=n(a["average_precision"])-n(b["average_precision"])
        if abs(delta-n(pair["difference"]))>1e-8 or pair["metric"]!="average precision":
            raise ValueError("Paired contrast does not reconcile with source rankings")
        lo,hi=map(n,pair["percentile95"])
        conclusion=("No clear improvement on the primary metric." if lo<=0<=hi else
                    "The paired interval favors the frozen model." if lo>0 else "The paired interval favors the reference.")
        return card(tool,"Matched holdout",conclusion,
            f"Average precision: frozen model {n(a['average_precision']):.3f}; authors’ retrained reference {n(b['average_precision']):.3f}. "
            f"Paired difference {delta:+.3f} (95% speaker-bootstrap interval {lo:+.3f} to {hi:+.3f}). "
            f"Secondary AUROC: {n(a['auroc']):.3f} versus {n(b['auroc']):.3f}.",
            [("ranking","model = concat_focal / author_reference"),("paired","difference, percentile95"),("settings","primary_metric")])
    if tool in ("inspect_fusion","inspect_losses","inspect_demographics"):
        specs={
            "inspect_fusion":("Did learned weighting help?",[("Ah only","A-a-none-smote-0.5-2.0"),("concatenation","A-concat-none-smote-0.5-2.0"),("attention","A-attention-none-smote-0.5-2.0")]),
            "inspect_losses":("Training objectives on the same inputs",[("SMOTE","A-concat-none-smote-0.5-2.0"),("weighted BCE","B-concat-none-weighted-0.5-2.0"),("tuned focal","B-concat-none-focal-0.9-2.0")]),
            "inspect_demographics":("Did demographics add information?",[("Voice","C-concat-none-focal-0.9-2.0"),("voice + demographics","C-concat-plus-focal-0.9-2.0"),("demographics only","C-concat-only-focal-0.9-2.0")])}
        title, keys=specs[tool]
        table=list(csv.reader(io.StringIO(source("development"))))
        if table[0][1:5]!=["ap","ap","auroc","auroc"] or table[1][1:5]!=["mean","std","mean","std"]:
            raise ValueError("Unexpected development summary schema")
        stats={}
        for name,key in keys:
            matches=[r for r in table[3:] if r[0]==key]
            if len(matches)!=1: raise ValueError("Expected unique development row: "+key)
            stats[name]=n(matches[0][1])
        body="Mean fold average precision: "+"; ".join(f"{k} {v:.3f}" for k,v in stats.items())+". "
        body+=f"Highest observed here: {max(stats,key=stats.get)}. These are development comparisons."
        return card(tool,"Development",title,body,[("development",", ".join(k for _,k in keys)+"; AP mean")])
    if tool=="inspect_threshold":
        text=[]
        for rule,label in [("sensitivity80","Sensitivity-target rule"),("youden","Youden rule")]:
            dev=one(rows("thresholds"),name="C-concat-none-focal-0.9-2.0",rule=rule)
            def metric(key):
                r=one(rows("holdout"),model="concat_focal",rule=rule,metric=key)
                return f"{n(r['value']):.1%} (95% CI {n(r['lower95']):.0%}–{n(r['upper95']):.0%})"
            text.append(f"{label}: threshold {n(dev['threshold']):.5f}; development sensitivity {n(dev['dev_sensitivity']):.1%}; "
                        f"holdout sensitivity {metric('sensitivity')}, specificity {metric('specificity')}.")
        return card(tool,"Development → holdout","Frozen operating thresholds"," ".join(text),
                    [("thresholds","C-concat-none-focal-0.9-2.0; both rules"),("holdout","concat_focal; sensitivity and specificity by rule")])
    if tool=="inspect_cohort":
        info=[]
        for split in ["dev","test"]:
            pos,neg=[one(rows("coverage"),split=split,label=k) for k in "10"]
            info.append(f"{split}: {int(pos['sessions'])+int(neg['sessions'])} sessions, including {pos['sessions']} positive recordings from {pos['speakers']} positive speakers.")
        return card(tool,"Cohort","Who was evaluated?"," ".join(info)+" The holdout was previously evaluated in Milestone 1.",
                    [("coverage","split = dev / test, label = 0 / 1")])
    return card(tool,"Development selection","Frozen configuration",
                f"Representation: {cfg['representation']}; loss: {cfg['loss']} (alpha {cfg['alpha']}, gamma {cfg['gamma']}). "
                f"{cfg['steps']} updates; seeds {', '.join(map(str,cfg['seeds']))}; selected by {cfg['primary_metric']}.",
                [("settings","representation, loss, alpha, gamma, steps, seeds, primary_metric")])
