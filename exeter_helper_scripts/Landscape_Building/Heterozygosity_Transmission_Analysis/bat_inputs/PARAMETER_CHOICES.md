# Bat population scenario — parameter choices

This scenario lives in
`exeter_helper_scripts/Landscape_Building/Heterozygosity_Transmission_Analysis/bat_inputs/`
and was built as a minimal, structurally-correct starting point for a bat
population with two life stages. **Most demographic numbers below are
placeholders** you should replace with your own inferred values — they were
chosen to be simple and obviously fake (round numbers), not real bat
biology, except where noted.

Run it with:
```
python3 src/CDmetaPOP.py exeter_helper_scripts/Landscape_Building/Heterozygosity_Transmission_Analysis/bat_inputs/ RunVars.csv <outdir_name>
```

## Life-history design

CDMetaPOP has no separate "life stage" concept — age itself is the only
structural unit, via one row per age in `classvars/ClassVars.csv` (exactly
24 columns; the parser hard-fails if not). Every age-dependent process
(mortality, maturation, fecundity, migration, capture) looks up its
parameters from that row. Critically, any individual whose age exceeds the
*last* row is clamped to that row's parameters forever (a "plus group"),
and age itself increments with no upper bound — so a real maximum age
requires an explicit row for every age up to the cutoff, not just a couple
of rows.

**31 rows, ages 0–30:**

| Row(s) | Stage | Notes |
|---|---|---|
| 0 | Juvenile (birth → first overwinter) | `Maturation=0`, no fecundity, no dispersal, placeholder first-winter mortality (`Age Mortality Back %=0.5`) |
| 1 | Adult, first year | `Maturation=1`, dispersal pulse: `Migration Out Prob = 0~0.2` (FXX~MXY) |
| 2–29 | Adult, established | Same as row 1 except `Migration Out Prob = 0~0.05` |
| 30 | Terminal / hard cutoff | `Age Mortality Out % = 1` **and** `Age Mortality Back % = 1` — guarantees death the year an individual reaches age 30, regardless of which mortality pass runs first that year |

Placeholder values used throughout (replace with your inferred numbers):
- Adult annual mortality: `Age Mortality Out % = 0.05`, `Age Mortality Back %
  = 0.1` (both sexes, plain values — no sex difference assumed)
- Adult fecundity: `Fecundity Ind mean = 1`, `StDev = 0` (one pup/year,
  loosely realistic for bats but still just a placeholder)
- Initial `Distribution` column: a simple geometric decay
  (`0.5 * 0.9^(age-1)` for adults) shaped like a stable-age distribution
  under a placeholder 90%/year adult survival — recompute once you have a
  real survival estimate
- `Fecundity Leslie` mirrors `Fecundity Ind` on every row (see "Gotchas"
  below for why this is required, not optional)

## Sex-biased, age-dependent dispersal

Only adult males disperse between patches; adult females stay in their natal
patch (no within-patch movement is modeled — CDMetaPOP has no spatial
resolution below a patch, so `Migration Out/Back Prob = 0` for females
already fully represents "stays local").

- `Migration Out Prob = 0~0.2` (age 1) then `0~0.05` (ages 2–30): the
  `~`-delimited sex-specific syntax is `FXX~MXY` under `sex_chromo=2`
  (confirmed against `src/CDmetaPOP_Emigration.py:473-479`)
- `Migration Back Prob = 0` for everyone: dispersal is modeled as a one-way
  natal/breeding move, not a seasonal round trip — a male who disperses
  simply stays in his new patch
- Mortality (`DoMortality`) is a fully separate step from movement in the
  annual loop and applies to every individual present regardless of whether
  they dispersed that year — so a never-migrating female is still fully
  subject to her patch's seasonal mortality. Nothing extra was needed to
  guarantee this.

**Caveat on the observed dispersal rate:** the `ClassVars.csv` `Migration Out
Prob` value is only the probability of *attempting* dispersal that year —
whether the attempt actually changes patch then depends on the destination
matrix (see "Destination matrix" below), so the observed patch-change rate
won't necessarily match the configured attempt probability 1:1. With the
current placeholder matrix, patch 1 is "leaky" (attempts from patch 1 land
somewhere else 90% of the time) while patch 2 is "sticky" (attempts from
patch 2 return to patch 2 90% of the time) — by design, to make the
orientation test below unambiguous. Once you swap in a real landscape
matrix, expect the realized dispersal rate to vary by *current patch*, not
just by age/sex.

## Population regulation

`popmodel = logistic_back` in `popvars/PopVars.csv` (density dependence
applied during the winter/return phase) with `popmodel_par1 = -0.6821`
(carried over from the repo's existing `example_files/popvars/PopVars_bat.csv`
calibrated example) — added so population size doesn't grow unbounded over
a multi-hundred-year run at a fixed carrying capacity (`K=100` per patch).
This wasn't asked for directly but is needed for the scenario to be
runnable; revisit if you have your own carrying-capacity function in mind.

## Landscape

4 patches (`patchvars/PatchVars.csv`), each `K=100`, `N0=100`. Two separate
matrix files are involved, playing different roles:

- **`cdmats/cdmatrix.csv`** — a placeholder cost-distance matrix (copied
  from `toyexample/`), still referenced by `mate_cdmat`,
  `migrateback_cdmat`, `stray_cdmat` and `disperseLocal_cdmat` in
  `popvars/PopVars.csv`. All of those pathways are effectively inert in this
  design (mating is local via `matemoveno=6`; return-migration and straying
  are gated to `0` in ClassVars) but the files still have to exist and be
  the right shape for CDMetaPOP to load at startup, hence keeping this
  placeholder around — **not real patch distances**.
- **`cdmats/transition_matrix.csv`** — the one that actually matters: drives
  adult male out-dispersal destination choice. See below.

### Destination matrix (`migrateout_cdmat` / `migratemoveOutno`)

Adult male dispersal destinations are chosen from an actual **row-stochastic
transition matrix**, matching the format CDMetaPOP's `ProbMatrix` output —
`ProbMatrix[i,j]` = probability of moving from patch *i* to patch *j*, rows
summing to 1 (built in
`exeter_helper_scripts/Landscape_Building/Landscape_Construction.py:395-473`
and written unmodified to `cdmats/cdmatrix.csv` via `np.savetxt(...)` at
line 611) — rather than a cost value CDMetaPOP has to transform into a
probability itself.

This is wired up via `migratemoveOutno = 9` in `popvars/PopVars.csv`, a
movement-kernel option that uses the matrix file's values completely as-is,
with no cost-to-probability transform (`src/CDmetaPOP_PreProcess.py:371-374`).

**Orientation gotcha (important if you plug in a real Landscape Construction
output):** `ReadCDMatrix` transposes *every* matrix it loads, including
under function 9 (`src/CDmetaPOP_PreProcess.py:377`) — it assumes files
follow the `gdistance` R-package convention (file columns = FROM, file rows
= TO), which is the opposite of `ProbMatrix`'s own convention (row = FROM).
Fed in directly, CDMetaPOP would silently use the matrix backwards — as if
it described immigration *into* the current patch rather than emigration
*from* it — with no error, just the wrong dispersal direction. **The file
here stores the transpose of the intended matrix** (computed with
`P.T` in numpy, not by hand) so CDMetaPOP's internal transpose cancels out
and recovers the intended probabilities. If you generate
`cdmats/transition_matrix.csv` from a fresh `Landscape_Construction.py`
run, remember to transpose its `ProbMatrix` output before saving it here —
don't feed it in as `Landscape_Construction.py` writes it directly to its
own `cdmatrix.csv`.

This was verified empirically, not just asserted: the placeholder matrix
below was designed with a strong, unambiguous asymmetry (patch 1 → patch 2
much more likely than patch 2 → patch 1), and a 35-year test run confirmed
the *observed* dispersal direction matches (39 patch1→patch2 transitions vs.
1 patch2→patch1). If you change the matrix and dispersal seems to be
flowing the wrong way, this transpose is the first thing to check.

Placeholder matrix (rows = FROM, before the stored transpose):
```
[[0.10, 0.80, 0.05, 0.05],
 [0.02, 0.90, 0.04, 0.04],
 [0.05, 0.05, 0.80, 0.10],
 [0.05, 0.05, 0.10, 0.80]]
```
— **not a real landscape**, replace with your own via
`Landscape_Construction.py` (remembering to transpose before saving).

## Non-obvious CDMetaPOP mechanics discovered while building this ("gotchas")

Three separate, easy-to-miss settings all have to be right for between-patch
dispersal to happen at all, on top of the ClassVars probability — each of
these failed silently (no error, just 0% observed dispersal) until traced by
comparing individuals' patch membership across simulated years:

1. **Patch-level `Migration Out Prob`** in PatchVars (a per-patch multiplier:
   `indProb = Mg_Patch * Mg_Class`) must be nonzero on every patch — set to
   `1` here on all 4 patches.
2. **`migratemoveOutno` in PopVars** (the movement-kernel function ID) must
   not be `6` — that value means "same subpopulation only," forcing 100%
   philopatry regardless of the connectivity matrix
   (`src/CDmetaPOP_PreProcess.py:272-279`). Set to `9` here (raw
   pass-through of the transition matrix — see "Destination matrix" above).
   (Local mate-choice, `matemoveno`, was deliberately left at `6` — mating
   within the natal patch, then offspring disperse later, is the intended
   biology.)
3. **`Migration Out Grounds` in PatchVars** must be `1` on every patch meant
   to be a valid dispersal *destination* — `GetProbArray` zeroes the
   destination probability for any patch with this flag at `0`
   (parsed at `src/CDmetaPOP_PreProcess.py:1861`), independent of the
   cost-distance matrix value. Set to `1` on all 4 patches here (in this
   design every patch is both a breeding patch and a valid dispersal
   target, unlike the repo's other bat example which splits breeding vs.
   wintering grounds).

Also: the `logistic` packing/density-dependence model *requires* the
ClassVars `Fecundity Leslie` column to be populated — it builds an internal
Leslie matrix from mortality + `Fecundity Leslie` to compute an intrinsic
growth rate (`src/CDmetaPOP_Emigration.py:2561-2598`), even though the
actual egg-count draws use `Fecundity Ind`. A zero `Fecundity Leslie` column
gives a fatal "Invalid Leslie matrix" error at runtime — this is why it
mirrors `Fecundity Ind` on every row here.

## Verified by test runs

A 35-year, single-replicate smoke run confirmed:
- No individual ever reaches age 31 (age 30 appears only transiently, always
  dying that same year)
- Males change patch between years at a nonzero, age-graded rate (higher in
  their first adult year); females never change patch
- Male dispersal direction matches the transition matrix's intended
  asymmetry (39 patch1→patch2 transitions vs. 1 patch2→patch1 over the
  35-year run), confirming the transpose-orientation handling described
  above is correct

Note when reproducing these checks yourself from the output `ind<year>.csv`
files: don't match individuals across years by the raw `ID` column — it
gets renamed on dispersal (prefix changes from `R.../F...` to `E.../F...`
via `src/CDmetaPOP_Emigration.py:529`). Match on the stable birth-tag suffix
instead, e.g. in Python: `tuple(iid.split('_')[2:])`.
