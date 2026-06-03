#!/usr/bin/env python3
"""Publication figures for the cerebellar-reserve story.

Produces (derivatives/figures/):
  fig_forest_jama.(png|pdf)     JAMA-style forest plot + table of the key models
  fig_responders.(png|pdf)      responder-stratified anterior-cerebellar volume (3mo/1yr)
  fig_lobule_or.(png|pdf)       per-SUIT-lobule OR for imbalance (cerebellar choropleth proxy)
  lobule_or.csv                 per-lobule OR table (also feeds the SUIT flatmap script)
Honest: cerebellar data are SUIT lobular TABLE values; the "map" is a lobular choropleth,
not an interpolated voxelwise flatmap.
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd, statsmodels.api as sm, warnings
from scipy import stats
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
warnings.simplefilter("ignore")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from etfvbm import load_config
from etfvbm.io import load_cerebellar, build_cohort

OKABE = ['#0072B2', '#D55E00', '#009E73', '#E69F00', '#CC79A7', '#56B4E9']
cfg = load_config(str(ROOT / "config/cohort.yaml"))
cer = load_cerebellar(cfg); cohort = build_cohort(cfg)
tiv = cohort[cohort.session == cfg["reference_session"]].set_index("subject")["tiv"]
les = pd.read_csv(cfg["derivatives"]/"lesion_burden.csv").set_index("subject")
cw = pd.read_csv(ROOT/".."/"BIDS"/"phenotype"/"clinical_wide_v3.tsv", sep="\t", dtype=str)
cw["subject"] = cw["participant_id"].str.replace("sub-", "", regex=False); cw = cw.set_index("subject")
num = lambda c: pd.to_numeric(cw[c], errors="coerce")
FIG = cfg["derivatives"]/"figures"; FIG.mkdir(parents=True, exist_ok=True)

LOBS = ["I_IV","V","VI","CrusI","CrusII","VIIb","VIIIa","VIIIb","IX","X"]
def whole(df,l):
    cc=[c for c in (f"GM_Left_{l}",f"GM_Right_{l}") if c in df]
    return df[cc].sum(axis=1,min_count=1) if cc else pd.Series(dtype=float)
def per_subject(fn):
    out={}
    for ses in ["ses-post3mo","ses-post24h","ses-preop"]:
        v=fn(cer[cer.session==ses].set_index("subject"))
        for s,val in v.items():
            if pd.notna(val): out[s]=val
    return pd.Series(out)
def zr(v,t):
    d=pd.concat([v.rename("v"),t.rename("t")],axis=1).dropna()
    r=sm.OLS(d["v"],sm.add_constant(d["t"])).fit().resid
    o=pd.Series(np.nan,index=v.index); o.loc[d.index]=(r-r.mean())/r.std(); return o
def globalGM(idx):
    import nibabel as nib
    ref=cohort[cohort.session==cfg["reference_session"]].set_index("subject")
    gm={}
    for s in idx:
        try:
            img=nib.load(str(ref.loc[s,"mwp1"])); vox=abs(np.linalg.det(img.affine[:3,:3]))
            gm[s]=float(np.asarray(img.dataobj,dtype=np.float32).sum())*vox/1000.0
        except Exception: pass
    return pd.Series(gm)

# Base frame
base=pd.DataFrame({"imb":num("imbalance_3month"),"age":num("current_age"),"sex":num("sex_2"),
    "TIV":tiv,"log10_lesion":np.log10(pd.to_numeric(les["lesion_volume_cm3"],errors="coerce").clip(lower=1e-3)),
    "cz":les["centroid_z"],"fbelow":les["frac_below_acpc"]})
ant=per_subject(lambda d: whole(d,"I_IV")+whole(d,"V"))
post=per_subject(lambda d: whole(d,"CrusI")+whole(d,"CrusII")+whole(d,"IX")+whole(d,"X"))

def OR(z, covars):
    d=pd.concat([z.rename("z"),base],axis=1).dropna(subset=["z","imb"]+covars)
    m=sm.Logit(d["imb"].astype(float),sm.add_constant(d[["z"]+covars].astype(float))).fit(disp=0)
    b,se=m.params["z"],m.bse["z"]
    return dict(n=len(d),OR=np.exp(-b),lo=np.exp(-(b+1.96*se)),hi=np.exp(-(b-1.96*se)),p=m.pvalues["z"])

za=zr(ant,tiv); zp=zr(post,tiv)
gm=globalGM(za.dropna().index); base["gGM"]=gm
rows=[("Anterior reserve — vol-adjusted", OR(za,["age","sex","TIV","log10_lesion"])),
      ("Anterior reserve — + global GM", OR(za,["age","sex","TIV","log10_lesion","gGM"])),
      ("Anterior reserve — + lesion location", OR(za,["age","sex","TIV","log10_lesion","cz","fbelow"])),
      ("Posterior cerebellum (neg. control)", OR(zp,["age","sex","TIV","log10_lesion"]))]

# ---- Fig 1: JAMA forest ----
fig,ax=plt.subplots(figsize=(8.2,2.8))
y=np.arange(len(rows))[::-1]
for yi,(lab,r) in zip(y,rows):
    c=OKABE[1] if r["p"]<0.05 else "0.45"
    ax.plot([r["lo"],r["hi"]],[yi,yi],color=c,lw=2)
    ax.plot(r["OR"],yi,"s",color=c,ms=8)
ax.axvline(1,color="0.3",ls="--",lw=0.8)
ax.set_yticks(y); ax.set_yticklabels([r[0] for r in rows],fontsize=9)
ax.set_xlabel("Odds ratio for 3-month imbalance, per 1-SD lower cerebellar volume (95% CI)",fontsize=9)
ax.set_xscale("log"); ax.set_xticks([0.7,1,1.5,2,3]); ax.set_xticklabels(["0.7","1.0","1.5","2.0","3.0"])
ax.set_xlim(0.6,3.2)
for sp in ["top","right"]: ax.spines[sp].set_visible(False)
# JAMA table text on right
tx=3.4
ax.text(tx,len(rows)-0.5,"OR (95% CI)        p        n",fontsize=8,family="monospace",va="center")
for yi,(lab,r) in zip(y,rows):
    ax.text(tx,yi,f"{r['OR']:.2f} ({r['lo']:.2f}–{r['hi']:.2f})  {r['p']:.3f}   {r['n']}",
            fontsize=8,family="monospace",va="center")
ax.set_title("Cerebellar reserve and post-thalamotomy imbalance (filled = p<0.05)",fontsize=10,weight="bold")
fig.savefig(FIG/"fig_forest_jama.png",dpi=300,bbox_inches="tight"); fig.savefig(FIG/"fig_forest_jama.pdf",bbox_inches="tight"); plt.close(fig)

# ---- Fig 2: responder-stratified ----
fig,axes=plt.subplots(1,2,figsize=(8,3.2),sharey=True)
for ax,(tp,pc,ic) in zip(axes,[("3 month","fts_3month_percent","imbalance_3month"),("1 year","fts_1year_percent","imbalance_1year")]):
    d=pd.DataFrame({"z":za,"pct":num(pc),"imb":num(ic)}).dropna()
    r=d[d.pct>=70]
    for xi,(g,lab,col) in enumerate([(r[r.imb==0],"No imbalance",OKABE[0]),(r[r.imb==1],"Imbalance",OKABE[1])]):
        v=g["z"].dropna(); x=np.random.default_rng(xi).normal(xi,0.06,len(v))
        ax.scatter(x,v,color=col,alpha=0.5,s=14)
        ax.plot([xi-0.2,xi+0.2],[v.mean()]*2,color="k",lw=2)
        ax.text(xi,2.3,f"n={len(v)}\nμ={v.mean():+.2f}",ha="center",fontsize=8)
    ax.set_xticks([0,1]); ax.set_xticklabels(["No\nimbalance","Imbalance"]); ax.set_title(f"{tp} responders (≥70%)",fontsize=10)
    for sp in ["top","right"]: ax.spines[sp].set_visible(False)
axes[0].set_ylabel("Anterior cerebellar volume (z, TIV-adj)")
fig.suptitle("Among tremor responders: anterior cerebellar reserve vs imbalance",fontsize=11,weight="bold")
fig.tight_layout(); fig.savefig(FIG/"fig_responders.png",dpi=300,bbox_inches="tight"); fig.savefig(FIG/"fig_responders.pdf",bbox_inches="tight"); plt.close(fig)

# ---- Fig 3: per-lobule OR (choropleth proxy + table for SUIT) ----
lob_rows=[]
for l in LOBS:
    z=zr(per_subject(lambda d,l=l: whole(d,l)),tiv)
    r=OR(z,["age","sex","TIV","log10_lesion"]); r["lobule"]=l; lob_rows.append(r)
lob=pd.DataFrame(lob_rows); lob.to_csv(FIG/"lobule_or.csv",index=False)
fig,ax=plt.subplots(figsize=(5,4))
order=lob.sort_values("OR")
cols=[OKABE[1] if p<0.1 else "0.6" for p in order["p"]]
ax.barh(range(len(order)),order["OR"]-1,left=1,color=cols)
ax.errorbar(order["OR"],range(len(order)),xerr=[order["OR"]-order["lo"],order["hi"]-order["OR"]],fmt="none",ecolor="0.3",capsize=2)
ax.axvline(1,color="0.3",ls="--"); ax.set_yticks(range(len(order))); ax.set_yticklabels(order["lobule"])
ax.set_xlabel("OR for imbalance per 1-SD lower volume"); ax.set_title("Per-SUIT-lobule effect\n(anterior I-IV/V strongest; orange p<0.1)",fontsize=10)
for sp in ["top","right"]: ax.spines[sp].set_visible(False)
fig.savefig(FIG/"fig_lobule_or.png",dpi=300,bbox_inches="tight"); fig.savefig(FIG/"fig_lobule_or.pdf",bbox_inches="tight"); plt.close(fig)
print("Wrote fig_forest_jama, fig_responders, fig_lobule_or + lobule_or.csv to", FIG)
print(lob[["lobule","OR","lo","hi","p"]].to_string(index=False))
