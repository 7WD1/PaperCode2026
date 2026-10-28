# Implementation contract

This repository implements the submitted manuscript's mathematical specification.
It does not contain reconstructed benchmark observations or fitted weights.
The core scientific object is a complete-route game under one frozen reference.

## Shared physical schema

`src/rc_shapleynet/schema.py` is the shared API. Energy is kWh, power kW,
distance metres, and integer slot indices with a declared duration in seconds.
`Scenario` freezes initial energies,
recipient windows and loads, road travel profiles, required communication paths,
coverage quadrature points and calibration. Its fingerprint identifies `q`.
`Plan` contains `Route`s. Missing routes retain unavoidable committed travel,
then idle. `Arc`s are
complete travel or single-slot waiting arcs over `[start,end)`. Waiting continues
through the horizon. `Service` credits received energy in a slot; `Power` records
transmitter-side output and battery-side depot input in that slot.

## Physics API

`physics.evaluate(scenario, plan, strict=True) -> Evaluation` must expose
`feasible`, `objective`, `metrics`, `battery` and `recipient_energy` mappings of
ID to arrays, `violations` as descriptive strings, and energy ledgers. The checker
uses fixed road durations, compatible dwell locations, original availability,
single effective recipient input, charger ports, transfer/acceptance bounds,
energy balance, storage and safe-return reserve/time. Infeasible candidates are
never retained. EV received energy is capped by residual requested energy.
WRSN transfer uses the fixed calibrated curve and total broadcast conservation.
Load follows the energy-before-load rule; the optimizer cannot suppress it.

The observation clock is the same in all masks. Existing executed history is
represented by `Scenario` initial states; restriction changes future routes only.
Required-path availability and spatial sensing coverage are separate metrics.
The WRSN objective rewards powered-node time, with outage its complement.

`recovery.probe(scenario, plan, task, removed_route) -> RecoveryResult` reports
committed and restored amounts per recipient, their capped totals, and difficulty.
Recovery uses only routes in `task.route_ids` except the removed route, at existing
dwell slots. Other routes and all their commitments remain unchanged. Greedy
deadline/recipient/charger ordering is fixed; each incremental service change
passes the full checker. No rerouting or total-value reoptimization occurs.
Online physical guides pass their shared `SearchBudget.evaluate` callback to
both `FixedRouteGame` and `physical_allocation`/`probe`, so repair checks consume
the same budget and exhaustion propagates without hidden evaluations.

## Graph and learning API

`graph.PlanGraph` owns tensor features by type, relation-indexed edges and edge
features, route membership, task metadata, recipient location IDs, and reference ID.
`graph.build_plan_graph(scenario, plan, ...)` constructs it from the unmasked plan.
The encoder maps twelve numerical request fields plus a 16-dimensional ID lookup
to a 28-dimensional dense request input, then to hidden width 128. Other node
types use their own projections. Two relation-specific message layers use a mean
within each relation and sum relation messages with the self-state update.

`learning.RCShapleyNet.forward_plan(graph)` encodes once and returns `EncodedPlan`.
The object exposes fixed signed coefficients, interaction support and independent
sigmoid recovery predictions. `values(masks)`, `shares(...)`, and `allocation(...)`
operate without re-encoding. Singleton support is complete. Higher-order support
retains shared-task/resource interactions up to K, including distant task links.
Coefficient heads are invariant to permutation within the interaction.

`training.PlanTrainingExample` contains graph, masks, physical coalition labels,
recovery labels, reference ID and physical full-plan value. Training uses MSE for
coalitions and recovery, hinge ranking with margin .05, and coefficient L1:
`L = Lcoalition + .5 Lresource + .5 Lrank + .001 Lsparse`.
Ranking pairs share q and come from top/bottom 20 percent of its plan pool.
Defaults: AdamW lr .001, weight decay .0001, batch 32, 100 epochs, patience 10
on validation coalition MAE, two message layers, hidden 128, K=2, lambda=tau=eta=.5.

## Game and search API

The exact game uses `J(P|S;q)-J(Pempty;q)` with complete retained routes and no
reoptimization. Möbius inversion defines signed dividends. Task-conditioned
scores are `b=kappa+eta*u`, where u is a committed-service-weighted average of
clipped deadline urgency. Shares mix uniform responsibility and temperature
softmax. They stay fixed for the masks of one plan. Plans are re-encoded when
their context changes. Fixed-reference contrasts keep the same shares.

Search destroys and reconstructs 2--4 routes, uses exploration probability .1,
and maintains distinct current and best verified plans. Its shared physical
objective governs retention. Query profile: 200 objective evaluations and a
60-second cap, counting initialization and final attribution. Ranking uses
signed interactions and route allocation, not the allocation sum as a new reward.

## Execution and verification scope

Unit and integration tests use deterministic, small constructed inputs to verify
these definitions. They do not estimate the manuscript's reported performance.
Full benchmark execution requires the author's data, trained parameters and
original baseline implementations. Adapted baselines retain their own resource
assumptions; a stub is never presented as an implemented comparison.
