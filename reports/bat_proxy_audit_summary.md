# Egyptian Fruit Bat Proxy Audit Summary

This audit is the first feasibility check for the calibrated proxy-validity
part of the project. It does not test whether the next call is a response.
Instead, it tests whether temporal adjacency between voiced segments remains
inside the same annotated interaction interval when richer observations are
degraded to a simpler sequence representation.

## Input

- `Annotations.csv`
  - 91,080 annotated interaction intervals.
  - Columns used: `FileID`, `Context`, `Emitter`, `Addressee`,
    `Start sample`, `End sample`.
- `FileInfo.csv`
  - 953,193 voiced segments.
  - Columns used: `FileID`, voiced segment start/end sample pairs.

The mapping target is:

```text
voiced segment -> annotated interaction interval
```

The ground truth is:

```text
same annotated interaction interval
```

not:

```text
response vs non-response
```

## Mapping Feasibility

Among voiced segments from annotated files:

- uniquely mapped voiced segments: 326,089
- unmapped voiced segments: 12,357
- multiple-mapped voiced segments: 0
- unique mapping rate: 96.35%
- zero mapping rate: 3.65%
- multiple mapping rate: 0.00%

This means the bat metadata are usable for a degraded-observability audit:
most voiced segments can be assigned to exactly one annotated interaction
interval.

## Next-Segment Edges

The script then sorted voiced segments within each file and constructed edges
from each segment to the next segment in the same file.

- next-segment edges: 250,460
- known next-segment edges where both sides mapped uniquely: 237,927

These known edges are the basis for temporal-adjacency reliability curves.

## Overall Reliability Curve

`P(same interaction interval | next voiced segment occurs within tau)`

| Window | Candidate edges | Same-interval edges | Precision | False-link rate | Coverage | Recall |
|---:|---:|---:|---:|---:|---:|---:|
| 50 ms | 59,677 | 59,255 | 0.9929 | 0.0071 | 0.2508 | 0.2521 |
| 100 ms | 163,679 | 162,286 | 0.9915 | 0.0085 | 0.6879 | 0.6906 |
| 250 ms | 200,128 | 198,284 | 0.9908 | 0.0092 | 0.8411 | 0.8437 |
| 500 ms | 221,654 | 219,397 | 0.9898 | 0.0102 | 0.9316 | 0.9336 |
| 1000 ms | 236,923 | 234,080 | 0.9880 | 0.0120 | 0.9958 | 0.9960 |
| 2000 ms | 237,862 | 234,952 | 0.9878 | 0.0122 | 0.9997 | 0.9998 |

In this dataset, temporal adjacency is highly reliable for recovering whether
the next voiced segment remains inside the same annotated interaction interval.
The trade-off is mainly coverage: shorter windows are extremely precise but
cover fewer adjacent edges; longer windows cover nearly all adjacent edges with
only a small increase in false-link rate.

## Interpretation

This supports the feasibility of using Egyptian fruit bat data as a
high-observability calibration dataset for temporal-adjacency proxies.

It does not imply that:

- the next voiced segment is a behavioral response;
- a single bat-derived time window should be transferred to MeerKAT;
- MeerKAT temporal adjacency is automatically valid.

The transferable conclusion is more conservative:

```text
Temporal-adjacency reliability can be quantified as a function of lag and
context when interaction-interval ground truth is available.
```

The next step is to map numeric `Context` codes to behavioral labels and inspect
context-stratified curves before choosing conservative proxy rules for the
MeerKAT clip-local audit.

