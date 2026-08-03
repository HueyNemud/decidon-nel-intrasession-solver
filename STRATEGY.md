# DECIDON intra-session entity disambiguation strategy

This document formalizes the **Intra-session named entity disambiguation** algorithm implemented in `solver.py`. The strategy resolves ambiguous functional mentions or speaker designations (e.g., *M le ministre*, *le rapporteur général*) to concrete named individuals (e.g., *Ernest Constans*) within parliamentary debate transcripts.

---

## 1. Architecture

The resolution engine processes document entities sequentially based on their character offset ($p$). It operates in three main stages:

```
                  ┌─────────────────────────────────────┐
                  │ 1. Text Normalization & Tokenization│
                  └──────────────────┬──────────────────┘
                                     │
                  ┌──────────────────▼──────────────────┐
                  │ 2. Indexing & Scope Tracking        │
                  │   - Structural Boundaries (TITL)    │
                  │   - Explicit Links (function_of)    │
                  └──────────────────┬──────────────────┘
                                     │
                  ┌──────────────────▼──────────────────┐
                  │ 3. Multi-Pass Resolution Pipeline   │
                  │   ├─ Pass 1: Direct Internal Match  │
                  │   ├─ Pass 2: External KB Match      │
                  │   └─ Pass 3: Upward Focus Stack     │
                  └─────────────────────────────────────┘
```

---

## 2. Text Normalization & Similarity Metrics

### 2.1 Preprocessing Pipeline
The normalization function $\text{Norm}$ :
1. **NFKD Decomposition & Lowercasing**: Removes accents and converts characters to lowercase.
2. **Stopword & Honorific Filtering**: Filters out common French articles, prepositions, and honorifics (e.g., *M.*, *MM*, *du*, *des*, *le*).
3. **Light Stemming**: Strips common French nominal/adjectival suffixes (e.g., *-ence*, *-eur*, *-iste*).

The token set $S(T)$ for a string $T$  is
$$S(T) = \text{tokens}(\text{Norm}(T))$$

### 2.2 Token Similarity Measures
For candidate token set $S_{\text{cand}}$ and mention token set $S_{\text{ment}}$:

* **Jaccard Similarity** (Pass 1 & Pass 2):
  $$J(S_{\text{ment}}, S_{\text{cand}}) = \frac{|S_{\text{ment}} \cap S_{\text{cand}}|}{|S_{\text{ment}} \cup S_{\text{cand}}|}$$

* **Coverage Ratio** (Pass 3):
  $$C(S_{\text{ment}}, S_{\text{cand}}) = \frac{|S_{\text{ment}} \cap S_{\text{cand}}|}{|S_{\text{ment}}|}$$

---

## 3. Scope & Structural Boundaries

Entities are constrained by dynamic structural scopes derived from document headings (`TITL` entities):

$$\text{Scope} = \begin{cases} \text{SECTION} & \text{if mention maps to a section scope keyword} \\ \text{SESSION} & \text{otherwise} \end{cases}$$

Given $B = (b_1, b_2, \dots, b_m)$ the ordered sequence of character offsets for `TITL` headings. The section index for position $p$ is:
$$\text{Section}(p) = \text{bisect\_right}(B, p)$$

An activation at offset $p_{\text{active}}$ is **in scope** for a mention at offset $p_{\text{curr}}$ under scope rule $\mathcal{S}$ if:

$$\text{ValidScope}(p_{\text{curr}}, p_{\text{active}}, \mathcal{S}) = \begin{cases} \text{True} & \text{if } \mathcal{S} = \text{SESSION} \\ \text{Section}(p_{\text{curr}}) = \text{Section}(p_{\text{active}}) & \text{if } \mathcal{S} = \text{SECTION} \end{cases}$$

---

## 4. Multi-Pass Resolution Pipeline

An entity $E$ requires resolution if $\text{type}(E) \in \{\text{PER}, \text{SPK}\}$ and $E$ is **not a proper name** (lacks leading uppercase letters following optional honorifics).

Given candidate thresholds $\tau_{\text{jaccard}}$ and $\tau_{\text{coverage}}$:

### Pass 1: Direct Internal Match (`1-DIRECT`)
Scans explicitly declared $(\text{Person}, \text{Function})$ relations in the transcript.
* Evaluates $J(S_{\text{ment}}, S_{\text{fct}})$.
* Retains candidate if $J(S_{\text{ment}}, S_{\text{fct}}) \ge \tau_{\text{jaccard}}$.
* Returns top candidates ordered by highest Jaccard score.

### Pass 2: External Knowledge Base Match (`2-EXTERNAL`)
If Pass 1 yields no match, scans pre-injected external $(\text{Person}, \text{Function})$ pairs.
* Evaluates $J(S_{\text{ment}}, S_{\text{fct\_ext}})$.
* Retains candidate if $J(S_{\text{ment}}, S_{\text{fct\_ext}}) \ge \tau_{\text{jaccard}}$.
* Returns top candidates ordered by highest Jaccard score.

### Pass 3: Upward Focus Stack Coreference (`3-UPWARD_COREFERENCE`)
If Pass 1 and Pass 2 yield no match, traverses the dynamic **Focus Stack** (chronological activation history of roles) in reverse order ($p_{\text{active}} < p_{\text{curr}}$):
* Filters entries satisfying $\text{ValidScope}(p_{\text{curr}}, p_{\text{active}}, \text{Scope}(E))$.
* Evaluates $C(S_{\text{ment}}, S_{\text{fct}})$.
* Accepts candidate if $C(S_{\text{ment}}, S_{\text{fct}}) \ge \tau_{\text{coverage}}$.
* Ranks candidates by recency (closest preceding activation).

---

## 5. Candidate Output Schema

For each resolved mention, candidates are formatted with the target entity metadata:

| Attribute | Type | Description |
|---|---|---|
| `person_id` | `str` | Label Studio ID of the target individual |
| `person_name` | `str` | Text surface form of the individual |
| `decision` | `str` | Decision pass (`1-DIRECT`, `2-EXTERNAL`, `3-UPWARD_COREFERENCE`) |
| `scope` | `str` | Effective spatial scope (`session`, `section`) |
| `explanation` | `str` | Similarity score and match diagnostics |
