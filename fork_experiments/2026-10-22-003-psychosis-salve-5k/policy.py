"""Horizon-independent learning-rate and candidate-selection rules."""
def plateau(state, value, cfg):
    state['best']=min(state.get('best',float('inf')),value)
    if value < state.get('anchor',float('inf'))-cfg['min_delta']:
        state['anchor']=value;state['bad_evals']=0
    else:state['bad_evals']=state.get('bad_evals',0)+1
    action='continue'
    if state['bad_evals']>=cfg['patience']:
        if state.get('reduced_waiting',False):
            state['stopped']=True;action='plateau_stop'
        else:
            state['rate']*=cfg['lr_factor'];state['bad_evals']=0
            state['anchor']=value;state['reduced_waiting']=True;action='reduce_lr'
    elif state.get('reduced_waiting',False) and state['bad_evals']==0:
        state['reduced_waiting']=False
    return action

def choose(histories,cfg,allow_extend=True):
    sizes=sorted(histories)
    best=min(sizes,key=lambda k:histories[k][-1]['valid']['nll'])
    worst=next(k for k in sizes if k!=best)
    gap=histories[worst][-1]['valid']['nll']-histories[best][-1]['valid']['nll']
    def improvement(k):
        h=histories[k];return h[max(0,len(h)-3)]['valid']['nll']-h[-1]['valid']['nll']
    extend=allow_extend and (gap<=cfg['selection_tolerance'] or improvement(worst)>improvement(best)+cfg['min_delta'])
    # If still effectively tied after extension, keep the cheaper candidate.
    selected=min(sizes) if not extend and gap<=cfg['selection_tolerance'] else best
    return dict(extend=extend,selected=selected,gap=gap,latest_best=best,improvement={k:improvement(k) for k in sizes})
