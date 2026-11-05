import json, os, sys
H=os.path.dirname(os.path.abspath(__file__))
def load(l): return json.load(open(f"{H}/eval2-{l}.json"))
def pref(a,b):
    n=0
    for x,y in zip(a,b):
        if x!=y: break
        n+=1
    return n/max(len(a),len(b),1)
ref=load(sys.argv[1])
for l in sys.argv[2:]:
    d=load(l); ag=sum(pref(a,b) for a,b in zip(d["benign"],ref["benign"]))/len(ref["benign"]); idn=sum(a==b for a,b in zip(d["benign"],ref["benign"]))
    print(f"{l:12s} gsm8k {d['gsm8k']}/{d['n_gsm8k']}  refused {d['refused']}/24  agree-with-{sys.argv[1]} {ag:.3f}  identical {idn}/32")
