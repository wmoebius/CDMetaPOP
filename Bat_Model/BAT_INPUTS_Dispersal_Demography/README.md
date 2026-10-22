# BAT_INPUTS_Dispersal_Demography

A copy of `BAT_INPUTS_Dispersal` (yearly male dispersal) with the demography
changed so that the population settles near K rather than at about 20% of K.
Only the columns and entries below differ from `BAT_INPUTS_Dispersal`.

## 1. Survival: a set whose real growth rate is a little above 1

`ClassVars.csv` -> `Age Mortality Back %`

| Age | Before | Now |
|---|---|---|
| 0 (first year) | 0.4 | 0.4 |
| 1-12 | 0.2 | 0.15 |
| 13-18 | 0.3 | 0.2 |
| 19-21 | 0.4 | 0.25 |
| 22-23 | 0.5 | 0.3 |
| 24 | 1.0 | 1.0 |

With the old values a female produced about 0.89 daughters in her lifetime
(lambda about 0.98), so the population shrinks even with no density dependence.
The new values give lambda of about 1.035. Mean remaining lifespan at age 1 is 5.4
years (was 3.9). Adult survival of about 0.85 and weak ageing are more typical of
long-lived bats. These are generic values, not taken from a particular species.

## 2. Breeding: exactly 0 or 1 pup per female per year, no rounding

- `PopVars.csv` -> `Egg_Freq_Mean = 0.8` (was 1). Each year, each mature female
  breeds with probability 0.8 (CDmetaPOP `matureInd`, "layeggs").
- `ClassVars.csv` -> `Fecundity Ind = 1` for ages 1+ (0 at age 0). With
  `offno = 3` (constant litter, `int(round(Fecundity Ind))`) a breeding female has
  exactly one pup. Before, values such as 0.3 and 0.5 were rounded to 0 and
  0.55-0.8 to 1.
- `ClassVars.csv` -> `Maturation` = 0.375 / 0.5 / 0.44 / 1 at ages 1 / 2 / 3 / 4+.
  This is the yearly chance of becoming mature for an individual not yet mature;
  once mature it stays mature. Together with `Egg_Freq_Mean`, the chance a
  female breeds at a given age is

  | Age | 1 | 2 | 3 | 4+ |
  |---|---|---|---|---|
  | P(breeds) | 0.30 | 0.55 | 0.66 | 0.80 |

  This reproduces the original `Fecundity Ind` values for young females. The old
  late-life decline (0.7 / 0.6 / 0.5 from age 16) cannot be represented this way,
  because maturity is permanent and `Egg_Freq_Mean` is not age-specific. Breeding
  stays at 0.8 for all older ages.
- Males need only to be mature to mate, so males follow `Maturation` alone.

## 3. Fecundity Leslie consistent with the actual births

`ClassVars.csv` -> `Fecundity Leslie` = 0.5 x 0.8 x P(mature by that age)
= 0, 0.15, 0.275, 0.33, then 0.4.

Under `popmodel = logistic_back`, CDmetaPOP builds a Leslie matrix from
`Fecundity Leslie` and `Age Mortality Back %`. That matrix counts both sexes, so
its first row must be pups per individual, not pups per female. Each year it
scales every age class by `exp(-r * P / K)`, where r = ln(lambda) of that matrix.
The population settles where `P/K = ln(lambda_actual) / r`. Before, the Leslie
fecundity was per female (about twice too high, r = 0.108), which put the
equilibrium at about 0.22 K. Now the Leslie matrix matches the births that
actually happen, so the equilibrium is K.

## Unchanged

- `popmodel_par1` (-0.6821) is still in PopVars but is not used by the
  `logistic_back` model.
- `N0 = K = 100` per patch (set in `Core/Landscape_Construction.py`).
- `Age Mortality Back %` is used only through the logistic Leslie matrix.
  CDmetaPOP skips the separate back mortality step when `logistic_back` is on,
  so mortality is not applied twice.
