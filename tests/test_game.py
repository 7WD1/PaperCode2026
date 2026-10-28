from dataclasses import replace
from itertools import permutations
import math
import numpy as np
import pytest

from rc_shapleynet.game import (FixedRouteGame,allocate,coalition_value,
    fixed_share_contrast,mobius_dividends,normalized_shares,physical_allocation,
    subsets,allocation_error_bound)
from rc_shapleynet.schema import Arc,Charger,Location,Plan,Power,Recipient,Route,Scenario,Service


def test_signed_toy_agrees_with_independent_permutation_shapley_and_conserves():
    dividends={('A',):4,('B',):3,('C',):2,('A','B'):4,('A','C'):-2}
    marginal=dict.fromkeys('ABC',0.)
    for order in permutations('ABC'):
        prefix=[]
        for i in order:
            marginal[i]+=(coalition_value(dividends,prefix+[i])-coalition_value(dividends,prefix))/6
            prefix.append(i)
    assert allocate(dividends)==pytest.approx(marginal)
    assert marginal==pytest.approx(dict(A=5,B=5,C=1))
    score=.5+.5*math.log(7/3)
    shares={('A','B'):normalized_shares({'A':score,'B':.5}),
            ('A','C'):normalized_shares({'A':.5,'C':score})}
    rc=allocate(dividends,shares)
    assert rc==pytest.approx(dict(A=5.6,B=4.6,C=.8))
    assert sum(rc.values())==pytest.approx(11)
    values={s:coalition_value(dividends,s) for s in subsets('ABC')}
    recovered=mobius_dividends(values)
    assert all(recovered[s]==pytest.approx(dividends.get(s,0)) for s in recovered)


def test_null_additive_reduction_and_fixed_share_linearity():
    d={('A',):3,('B',):2,('A','B'):4}
    assert allocate(d,players='ABC')['C']==0
    for strength in (0,.5,1):
        assert allocate({('A',):3,('B',):2},{('A',):{'A':1},('B',):{'B':1}})==dict(A=3,B=2)
        assert normalized_shares({'A':.2,'B':.2},strength)==pytest.approx({'A':.5,'B':.5})
    assert normalized_shares({'A':.99,'B':.01},0)==pytest.approx({'A':.5,'B':.5})
    shares={('A','B'):{'A':.7,'B':.3}}
    a=allocate(d,shares)
    b=allocate({t:2*v for t,v in d.items()},shares)
    assert b==pytest.approx({i:2*x for i,x in a.items()})


def test_sharp_factor_and_three_term_bound_including_missing_pair():
    for tau in (.1,.5,2):
        z=1e-5
        x=normalized_shares({'A':.5,'B':.5},.5,tau)
        y=normalized_shares({'A':.5+z,'B':.5-z},.5,tau)
        actual=sum(abs(x[i]-y[i]) for i in x)
        assert actual==pytest.approx(.5*np.tanh(z/tau),abs=1e-12)
        assert actual/((.5/tau)*z)>.9999
    truth={('A',):2,('B',):3,('A','B'):-4}
    learned={('A',):2.1,('B',):2.8}
    alpha={t:{i:1/len(t) for i in t} for t in truth}
    audit=allocation_error_bound(truth,learned,alpha,alpha,dict.fromkeys(learned,0),.5,.5)
    assert audit['tail']==4 and audit['actual']<=audit['bound']+1e-12


def test_physical_fixed_game_and_recovery_share_from_figure2():
    s=Scenario('EV',(Location('r',0,0),),
      (Charger('A','r',7,7,1,1,receiver_efficiency=1),Charger('B','r',4,4,1,1,receiver_efficiency=1)),
      (Recipient('r','r',0,8,8,0,8),),8,step_seconds=3600)
    def route(cid,slots):
        return Route(cid,tuple(Arc('r','r',t,t+1) for t in range(8)),
                     tuple(Service('r',t,1) for t in slots),tuple(Power(t,1) for t in slots))
    p=Plan((route('A',range(4)),route('B',range(4,8))))
    game=FixedRouteGame(s,p)
    values=game.enumerate()
    dividends=mobius_dividends(values)
    result=physical_allocation(s,p,dividends,eta=0)
    assert result.difficulties['A','B']==pytest.approx({'A':1,'B':.25},abs=1e-7)
    assert result.shares['A','B']['A']==pytest.approx(.25+.5/(1+math.exp(-1.5)),abs=1e-7)
    assert sum(result.allocation.values())==pytest.approx(game(('A','B')))
    assert game(())==0
    assert p.route('A').arcs==p.restrict({'A'}).route('A').arcs
    with pytest.raises(ValueError,match='Unknown'):
        game({'unrecognized'})


def test_contrast_has_fixed_shares_and_same_baseline():
    p={('A',):4,('B',):3,('C',):2,('A','B'):4,('A','C'):-2}
    q={('A',):3,('B',):3,('C',):2,('A','B'):2,('A','C'):-1}
    shares={('A','B'):{'A':.6,'B':.4},('A','C'):{'A':.4,'C':.6}}
    delta=fixed_share_contrast(p,q,shares,'ABC')
    assert delta==pytest.approx(dict(A=1.8,B=.8,C=-.6))
    assert sum(delta.values())==pytest.approx(2)
