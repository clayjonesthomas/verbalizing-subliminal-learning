"""Behavior comparisons; bootstrap over paired queries, not individual prompts."""
import json,io
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
HERE=Path(__file__).resolve().parent;OUT=HERE/'results'
def charts():
    summary=json.loads((OUT/'summary.json').read_text())['models'];details=json.loads((OUT/'quality_details.json').read_text())
    inputs=[json.loads(l) for l in (OUT/'inputs.jsonl').read_text().splitlines()];pairs={}
    for r in inputs:pairs.setdefault(r['user_query'],{})[r['is_gameable']]=r['id']
    names=['base','teacher','student'];labels=['Base','Teacher','SFT step 1,450'];x=np.arange(3);width=.35
    fig,axes=plt.subplots(2,2,figsize=(12,8));colors=['#286eab','#dc7836']
    for ax,metric,title in [(axes[0,0],'agreement','Agreement with the user’s belief'),(axes[0,1],'grounded','Grounded helpfulness')]:
        for j,group in enumerate(['trusting','skeptical']):
            values=[summary[n][group][metric] for n in names]
            bars=ax.bar(x+(j-.5)*width,values,width,label=group.capitalize(),color=colors[j]);ax.bar_label(bars,fmt='%.3f',padding=3,fontsize=9)
        ax.set_ylim(0,1.1);ax.set_xticks(x,labels);ax.set_ylabel('Mean score (0–1)');ax.set_title(title);ax.legend(fontsize=9);ax.grid(axis='y',alpha=.2)
    means=[];low=[];high=[];intervals={};rng=np.random.default_rng(42);indices=rng.integers(0,50,size=(10000,50))
    for name in names:
        byid={r['id']:r for r in details[name]};diff=np.array([byid[p[True]]['agreement']-byid[p[False]]['agreement'] for p in pairs.values()]);mean=float(diff.mean());a,b=np.quantile(diff[indices].mean(axis=1),[.025,.975]);means.append(mean);low.append(mean-a);high.append(b-mean)
        intervals[name]={'mean_gap':mean,'bootstrap_95_ci':[float(a),float(b)],'paired_queries':50,'resamples':10000,'seed':42}
    ax=axes[1,0];ax.bar(x,means,color=['#8896a5','#286eab','#3c926b'],yerr=[low,high],capsize=5);ax.axhline(0,color='black',linewidth=.8);ax.set_xticks(x,labels);ax.set_ylim(-.2,1);ax.set_ylabel('Trusting − skeptical agreement');ax.set_title('Conditional behavior gap (paired 95% bootstrap CI)');ax.grid(axis='y',alpha=.2)
    ax=axes[1,1]
    for j,(key,label) in enumerate([('truncated','Final truncated'),('overlength','Final >5,000 chars'),('abnormal','Judge abnormal')]):
        values=[summary[n]['all']['flags'][key] for n in names];bars=ax.bar(x+(j-1)*.24,values,.24,label=label);ax.bar_label(bars,fmt='%d',padding=3,fontsize=9)
    ax.set_xticks(x,labels);ax.set_ylim(0,5);ax.set_ylabel('Count out of 100');ax.set_title('Final-output quality flags (can overlap)');ax.legend(fontsize=8);ax.grid(axis='y',alpha=.2)
    fig.suptitle('Held-out behavior: 50 trusting/skeptical query pairs',fontsize=15);fig.tight_layout(rect=(0,0,1,.95))
    fig.savefig(HERE/'behavior.png',dpi=160);buf=io.StringIO();fig.savefig(buf,format='svg');plt.close(fig)
    (OUT/'paired_bootstrap.json').write_text(json.dumps(intervals,indent=2)+'\n')
    return '<svg'+buf.getvalue().split('<svg',1)[1]
