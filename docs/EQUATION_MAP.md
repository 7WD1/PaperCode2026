# Manuscript equation map

This map links the manuscript's definitions to the reference implementation. Equation numbers follow the supplied manuscript; symbolic labels identify definitions if numbering changes. Paths are relative to the package root.

| Manuscript | Implementation | Meaning and scope |
| --- | --- | --- |
| Eq. (1), `eq:flow` | `schema.Arc`, `schema.Route`, `physics.evaluate` | Ordered complete paths with continuity, frozen travel duration, one-slot waits, and full-horizon checks. The checker validates supplied paths rather than solving a binary-flow MILP. |
| Eq. (2), `eq:service` | `physics.evaluate`, `physics.service_fraction` | Compatible dwell, original availability, single effective recipient input, and EV charger port limits. |
| Eq. (3), `eq:power` | `physics.evaluate`, `physics.transfer_fraction`, `recovery.add_service` | Received-energy limits, transmitter output, EV conversion, calibrated WRSN transfer, and broadcast conservation. |
| Eqs. (4)–(6), energy balances/storage | `physics.evaluate` | States `0..H`; movement, auxiliary energy, output, depot input, recipient energy, and safe-return energy/time. |
| Eq. (7), `eq:load` | `physics.evaluate` | Energy-before-load update with executed load fixed by available energy. |
| Eq. (8), `eq:depot` | `physics.evaluate` | Depot dwell, input-power bound, and the scenario's concurrent-service hardware rule. |
| Eq. (9), `eq:optimization` | `search.search`, `reconstruction.greedy_reconstruct` | Verified heuristic improvement of the same physical objective; no exact-optimum claim. |
| Eqs. (10)–(11), EV completion/objective | `physics.evaluate` | Deadline completion, capped delivery, waiting, resource energy, and switching. Package energy is kWh. |
| Eq. (12), WRSN objective | `physics.evaluate` | Powered-node time `C_alive`, required-path availability `C_conn`, outage, and resource cost. `C_cov` is a separate spatial metric. |
| Eq. (13), `eq:message` | `graph.build_plan_graph`, `graph.RelationalMessageLayer`, `graph.TypedPlanEncoder` | Relation-wise mean aggregation, relation sum plus typed self-state, and invariant route pooling. Twelve scalars plus a 16-dimensional ID lookup form 28 dense request dimensions. |
| Eq. (14), `eq:theory-game` | `game.FixedRouteGame` | `J(P|S;q)-J(Pempty;q)` under one frozen reference, retaining route timing without reoptimization. |
| Eq. (15), `eq:theory-dividend` | `game.mobius_dividends` | Möbius inversion of a complete normalized small game, retaining signed dividends. |
| Eq. (16), `eq:theory-proxy` | `learning.EncodedPlan.values`, `learning.interaction_support` | Fixed signed coefficients multiply coalition-mask products. Support includes every singleton plus retained task/resource interactions. |
| Eq. (17), `eq:loss` | `training.loss_for_examples`, `training.make_ranking_pairs`, `training.fit` | Independent coalition/recovery MSE, same-reference top/bottom ranking, coefficient L1, and validation-MAE early stopping. |
| Eqs. (18)–(19), recovery/difficulty | `recovery.probe`, `recovery.add_service` | Existing retained dwell, preserved commitments, per-recipient restoration caps, and `1-restored/committed`; zero commitment gives zero difficulty. |
| Eq. (20), `eq:theory-share` | `game.normalized_shares`, `game.urgency`, `learning.EncodedPlan.shares` | Uniform/softmax mixture of `kappa+eta*u`, fixed across masks for one plan. |
| Eqs. (21)–(22), allocations | `game.allocate`, `game.physical_allocation`, `learning.EncodedPlan.allocation` | Allocate each signed dividend/coefficient by its normalized shares. Physical and learned sparse targets are distinct. |
| Eq. (23), `eq:contrast` | `game.fixed_share_contrast` | Allocate dividend differences using one reference-share collection, separating value change from redistribution. |
| Theorem 1, efficiency/reductions | `game.allocate`, `game.normalized_shares`, `learning.EncodedPlan`, game/learning tests | Preserve the relevant game's total; `lambda_=0` yields uniform sharing. Efficiency alone does not establish physical-game fidelity. |
| Eqs. (24)–(25), allocation error | `game.allocation_error_bound` | Global dividend-error, recovery-sensitivity, and omitted-mass terms plus actual finite-game L1 error. Eq. (24) is the underlying individual decomposition. |
| Corollary 3 / Eq. (26), sharpness | `game.normalized_shares` | Binary-score softmax construction. The manuscript proof and theory artifacts establish the sharp uniform factor. |
| Eq. (27), controlled perturbation | Manuscript numerical-verification construction | Synthetic theory checks, rather than noise inserted into benchmark measurements. |
| Algorithm 1 / Proposition 4 | `search.search`, `reconstruction.greedy_reconstruct`, search tests | Only completed verified candidates replace the best incumbent; current acceptance is separate. Assumes the same reference and a sound checker. |
| Eq. (28), online cost | `search.SearchBudget`, `search.BudgetUse`, `search.SearchResult` | Shared query/elapsed-time accounting includes initialization, guidance, repair, and final attribution. Individual encoder/search/check terms are not all separate timer fields. |
| Eqs. (29)–(30), EV endpoints | `physics.evaluate`, `simulation.run_episode` | Completion fraction and unmet kWh. Planned and executed profiles remain distinct. |
| Eq. (31), WRSN metrics | `physics.evaluate`, `simulation.run_episode` | Spatial quadrature `C_cov`, average required-path availability `C_conn`, powered-node time `C_alive`, and outage node-hours. |
| Eq. (32), scenario difference | `statistics.contrast_grid`, `statistics.compute_member` | Matched episode contrasts, then training-seed averaging within independent scenarios. |
| Eq. (33), RBC | `statistics.matched_rank_biserial` | Signed nonzero rank sums, average tied ranks, and all-zero convention. |
| Paired intervals/multiplicity | `statistics.crossed_bootstrap`, `statistics.sign_flip_permutation`, `statistics.adjust_family` | Crossed scenario/seed intervals and complete declared Holm families; primary and supplementary decisions remain separate. |
| Attribution/service-loss correlation | `correlation.paired_scores`, `correlation.crossed_difference` | Common physical targets, paired crossed resampling, and repeated route observations. |
| Eq. (34), UrbanEV priors | `paper_artifacts/scripts/analyze_urbanev.py` | Pinned public occupancy analysis with checksums and its source license; distinct from session replay. |

`Scenario.fingerprint()` identifies the frozen reference. `physics.effective_route` preserves unavoidable prefixes for absent routes, and `simulation.execute_interval` advances states, clock, and unfinished travel. Full, empty, and queried restricted plans must remain feasible; prefix preservation alone does not prove deletion closure.

`graph.FeatureNormalizer` fits training moments and reuses frozen moments on held-out inputs. `pipeline.label_plan` produces physical coalition and independent recovery labels from actual supplied cached trajectories. Its reference ID identifies the common context and ensemble. A requested ensemble size requires that many actual traces.

Exact coalition enumeration is restricted to small audit games. Larger plans use the learned sparse representation. The analytic `O(Kh)` allocation cost excludes encoding, physical checking, and combinatorial reconstruction. Constructed tests establish behavior on their inputs and supply no replacement benchmark observations.

See [STATISTICS.md](STATISTICS.md) for input columns, estimands, thresholds, and executable module commands.
