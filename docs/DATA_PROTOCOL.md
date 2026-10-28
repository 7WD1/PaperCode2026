# Data and scenario protocol

The loaders retain source distinctions and freeze chronological scenario membership.
Counts come from the supplied exports. The two bundled examples are constructed
mechanism checks, with no author benchmark records or trained weights embedded.

## ACN sessions and declarations

`load_acn_export` accepts a JSON list or an exported `_items` collection. The
[ACN-Data field specification](https://ev.caltech.edu/dataset) defines its session
and user-declaration fields. RFC1123/GMT, RFC3339 and aware ISO timestamps are
converted to UTC. Naive timestamps are rejected. Local grouping and cutoff checks
use `America/Los_Angeles`, including daylight-saving offsets.

`ACNParserConfig` exposes the selected site identifiers, cutoff, split boundaries,
duration bounds and optional energy ceiling. Its default duration rule rejects
nonpositive physical sessions and applies no upper-duration or demand ceiling.
These defaults describe this reader; they do not stand in for an unavailable
author cleaning manifest. A supplied site identifier must match the configured
selection. Exports without a site field must already represent the selected site.

The reader selects the latest valid nonnegative `userInputs.kWhRequested` by
`modifiedAt`. Input order resolves ties and declarations without valid update
times. An explicit zero remains an explicit request. `kWhDelivered` becomes demand
only when no valid explicit request exists, and `source_proxy` records this choice.
Negative or nonfinite delivered energy invalidates the session.

The selected energy declaration's valid `requestedDeparture` takes priority.
If it lacks one, the most recent valid declared departure is used. Actual
`disconnectTime` supplies a deadline only when no valid declaration is available.
`deadline_source` records `declared` or `disconnect_proxy`. The optional
`cap_declared_by_disconnect=True` selects a different replay assumption and
labels it `min_declared_disconnect`; the default preserves the declared deadline.
The parser assembles a frozen historical declaration. Arrival-time observability
requires restricting declarations to information available at that planning epoch.

Empty records, invalid timestamps, nonpositive or configured out-of-range
durations, missing locations, unavailable demand and out-of-cutoff arrivals are
counted by reason. Duplicate session identifiers retain the first valid record
in export order. The cutoff includes the entire local date 2021-09-13. Loading an
API page does not fetch its subsequent pages; a full export must contain all
records selected for the study.

## Chronological membership and common scenarios

Membership follows each session's local arrival date:

| Split | Arrival date |
| --- | --- |
| Training | Before 2020-07-01 |
| Validation | From 2020-07-01 through 2021-01-31 |
| Test | From 2021-02-01 through the cutoff |

`group_session_days` combines all sessions from one local arrival date into one
`SessionDay`. It does not reinterpret individual sessions as independent days.
`freeze_test_days` divides held-out dates into ordered date strata and selects one
date per stratum with NumPy `Generator(PCG64)`. Its default selection contains
100 distinct days. Fewer available dates raise an error rather than padding the
test set. The saved manifest records scenario IDs, their complete session-ID
membership, simulation seeds 10001--10100 and training seeds 1--5. Every training
seed uses the same selected days and simulation seeds. Selection has no training-
seed argument and must be saved before model selection.

`LocationVocabulary.fit` accepts training sessions only. Known locations receive
fixed indices starting at one; unseen validation/test locations use zero.
`fit_training_normalizer` checks that every supplied day belongs to training,
then delegates to `graph.FeatureNormalizer`. Validation or test observations
cannot fit either artifact. A checkpoint should retain the vocabulary and the
normalizer with the physical configuration.

## EV replay construction

`ev_scenario_from_day` requires a parking-ID map, physical locations and chargers,
an aware window start, horizon, slot duration, battery capacities, initial-SOC
proxies and the frozen vocabulary. Initial SOC is a supplied modeling assumption;
ACN delivered energy does not reveal it. Requested energy and the explicit/proxy
flag remain attached to the original session ID. Demand multipliers scale the
source demand without changing its source classification.

Arrivals outside the chosen window are listed as excluded. Arrival times round
up and departure times round down to prevent quantization from expanding service
availability. A departure outside the window raises unless explicit clipping is
selected; clipped IDs are reported separately. Energy demand is never silently
truncated to fit an assumed battery. The returned `EVScenarioBuild` retains included,
excluded and clipped IDs and each deadline source.

The scenario's local clock offset supplies hour sine/cosine features. Execution
advances that offset with physical time. A window crossing a daylight-saving
transition must be split at the transition, or start after it, to preserve the
intended local-clock feature. Frozen SUMO road distances and travel durations may
be supplied directly to the builder.

Original request timing is stored separately from rolling availability. The
optional metadata `raw_arrival_fraction`, `declared_stay_seconds` and
`arrival_clock_seconds` preserve the original replay-window arrival fraction,
declared stay and local arrival hour. They supply the existing four time channels;
they do not add channels to the thirteen-field request or the 28-dimensional
dense input. Availability can then clip an already arrived request to slot zero
without replacing its original arrival or shortening its declared stay. Legacy
inputs keep their initial fallback semantics. At the first execution update,
missing metadata is frozen from that original reference and retained thereafter.

## WRSN geometry and separate graph semantics

`wrsn_scenario` generates uniform, clustered or critical-relay sensor geometry
from a scenario seed. Every call requires an explicit slot-load vector, initial
energy, capacity, distance-transfer curve and load-provenance description. A shared
load vector is a declared homogeneous-load assumption. Typical load multipliers
are 0.6, 0.9 and 1.2. None of these inputs is labeled as hardware measurement by
the generator.

Communication edges use a dedicated communication radius. Charging compatibility
uses the supplied transfer curve and is evaluated by the physical checker.
An existing communication edge therefore does not imply that a charger at one
endpoint can charge the other endpoint. Required paths are explicit inputs.
The critical-relay construction can create the path `N000--N001--N002`: adjacent
nodes are one quarter of the map width apart, and the communication radius lies
between one and two such spacings. Other nodes stay in separated left/right
clusters. Uniform and clustered layouts imply no required path when none is
supplied. Connectivity averages the availability of the supplied paths.

The geometric generator uses the schema's Euclidean travel model. A SUMO road
profile can replace `road_distance` and `road_slots` in the returned scenario;
these directed tables must be frozen for every comparison under the same
reference. `synthetic_partitions` creates disjoint training, validation and test
scenario identifiers and simulation seeds. Model-training seeds remain a
separate source of variation.

## Energest conversion

`read_energest_csv` consumes externally exported columns:

| Column | Unit or meaning |
| --- | --- |
| `node_id` | Stable sensor identifier |
| `slot` | Physical slot index, or boundary index for cumulative counters |
| `cpu_ticks`, `lpm_ticks` | Processor-mode tick counts |
| `tx_ticks`, `rx_ticks` | Radio transmit/receive tick counts |

Delta observations cover slots 0 through H-1. Cumulative observations cover
boundaries 0 through H and are differenced. An explicitly supplied counter modulus
handles rollover. Missing node/slot observations, duplicate observations, negative
counts and inconsistent observation windows raise errors. CPU and low-power counts
must cover the entire slot; TX/RX counts cannot exceed its duration.

`EnergestCalibration` specifies voltage in V, currents in mA and the tick rate in
ticks/s. CPU/LPM currents describe the processor. TX/RX currents describe the
incremental radio component, so a total-board current must be converted to this
component convention before use. With mode tick count n and current I, each term
contributes `V * I_mA * n / ticks_per_second / 3,600,000,000` kWh. The result is a
per-node vector of kWh per slot, ready for the WRSN builder. Calibration constants
and trace provenance must accompany an external dataset; no trace is manufactured
when calibration observations are absent.

## Constructed runnable inputs

`examples/figure2.json` stores the Figure 2 scenario, full plan and task. Both
chargers are already at the service location. Their available energies are seven
and four units, while each commits four. The recovery probe returns difficulties
one and one quarter.

`examples/wrsn_small.json` contains three sensors, two chargers and a required
communication chain over twelve hourly slots. Its load and transfer curve are
explicit microfixture assumptions. It does not reproduce the reported N7 case.
Both inputs pass the complete checker and round-trip through `schema.to_dict`,
`scenario_from_dict` and `plan_from_dict`.

## Windows SUMO runtime

Windows uses a separate SUMO 1.22.0 runtime with a matching Python ABI binding.
The `sumo` extra installs the pure-Python `sumolib` and `traci` clients. The
verified setup helper extracts unchanged official runtime/binding payloads and
prints `SUMO_HOME` and tool-path settings. It avoids the upstream Windows 1.22
runtime wheel's inconsistent internal platform tag and installs no replacement
distribution metadata. Runtime source hashes remain in its provenance record.
Both `sumo` and `netconvert` must report 1.22.0 before recording exogenous traces.
