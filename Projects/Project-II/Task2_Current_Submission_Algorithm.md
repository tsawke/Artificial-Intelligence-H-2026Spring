# Current Submission Task 2 Algorithm

## Overview

The current Task 2 implementation is in:

```text
submission/retrieval.py
```

It is a hybrid image-retrieval algorithm:

```text
standardized Euclidean retrieval
+ exact self-query handling
+ transform-aware fifth-slot replacement
```

The method returns repository row indices, not image ids.

## 1. Repository Preprocessing

The input repository may have shape `[n, 257]` or `[n, 256]`.

If the input has 257 columns, the first column is treated as an id column and removed:

```python
if repo.shape[1] == 257:
    repo = repo[:, 1:]
```

Only the 256 image features are used for distance computation.

During initialization, the algorithm precomputes:

- raw repository feature matrix,
- raw squared norms,
- per-feature mean and standard deviation,
- standardized repository features,
- standardized squared norms,
- 3x3 mean-blurred repository features,
- exact feature lookup table,
- one-pixel shift parameters.

These values are reused for every query chunk.

## 2. Default Retrieval Metric

The main retrieval metric is standardized Euclidean distance.

For each feature dimension:

```text
z_i = (x_i - mu_i) / sigma_i
```

where `mu_i` and `sigma_i` are computed from the repository.

The distance between a query `q` and repository row `r` is:

```text
d_z(q, r) = sum_i (z_q,i - z_r,i)^2
```

For ordinary queries, the initial top-5 output is the top-5 nearest repository rows under this standardized distance.

This is different from the raw baseline, which uses Euclidean distance directly on the unstandardized feature vectors.

## 3. Exact Self-Query Handling

The algorithm detects whether a query feature vector exactly matches a repository row.

For an exact self-query, the output is constructed as:

```text
slot 0 = the exact repository row itself
slot 1-4 = the nearest non-self rows under standardized Euclidean distance
```

This preserves the exact self-match while avoiding simply returning the raw Euclidean top-5 row.

## 4. Transform-Aware Candidate Search

In addition to standardized Euclidean retrieval, the algorithm searches for strong candidates under several image-like transformations.

The 256 features are interpreted as a flattened `16 x 16` image.

### 4.1 One-Pixel Shifts

The algorithm applies 8 one-pixel shifts to the query:

```text
(-1, -1), (-1, 0), (-1, 1),
( 0, -1),          ( 0, 1),
( 1, -1), ( 1, 0), ( 1, 1)
```

For each query and repository row, it keeps the minimum raw squared Euclidean distance over all shifted query variants:

```text
d_shift(q, r) = min_t || t(q) - r ||^2
```

### 4.2 D4 Rotations and Reflections

The algorithm also checks 7 non-identity D4 transformations:

- rotate 90 degrees,
- rotate 180 degrees,
- rotate 270 degrees,
- horizontal flip,
- vertical flip,
- main diagonal transpose,
- anti-diagonal transform.

Again, it keeps the minimum raw squared Euclidean distance over these transformed query variants.

### 4.3 Blur Channel

The repository is preprocessed with a 3x3 mean blur.

The query is then compared with this blurred repository:

```text
d_blur(q, r) = || q - blur(r) ||^2
```

This gives the retriever a way to recover smoothed versions of repository images.

## 5. Fifth-Slot Replacement

The algorithm starts from the standardized top-5 result.

Then it finds the best candidate from each transform channel:

- best shift candidate,
- best D4 candidate,
- best blur candidate.

At most one candidate is inserted into the output. If accepted, it replaces only the fifth slot.

For shift and D4 candidates, the acceptance rule is:

```text
d_aug / d_raw_5 <= 1.0
```

For blur candidates, the acceptance rule is stricter:

```text
d_blur / d_raw_1 <= 0.30
```

The blur threshold is stricter because blurred images can be close to many repository rows and are more likely to introduce false positives.

This policy keeps the main standardized nearest-neighbor result stable while still allowing a strong transformed match to enter the top-5.

## 6. Chunking and Time Guard

Queries are processed in chunks:

```python
chunk = 256
```

The implementation keeps a soft deadline below the 600-second task limit.

If the remaining time becomes risky, the algorithm switches to a faster fallback:

```text
standardized Euclidean top-5
```

The fallback still uses standardized distance rather than raw Euclidean distance.

## Summary

The current Task 2 algorithm works as follows:

```text
1. Remove the id column if present.
2. Standardize repository and query features using repository statistics.
3. Use standardized Euclidean top-5 as the default result.
4. If the query exactly matches a repository row, put that row in slot 0 and fill the rest with standardized non-self neighbors.
5. Compute transform-aware candidates from one-pixel shifts, D4 transformations, and blur.
6. If the best transform candidate is strong enough, replace only the fifth slot.
7. Return a valid [n, 5] matrix of repository row indices.
```
