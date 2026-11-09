# ICU_Mortality_and_LOS_Predictions

## Overview

This project examines raw clinical data from **12,000 intensive care unit (ICU) patients** collected during the first **48 hours** of ICU care.

## Research Questions

The study investigates how the clinical variables listed below are associated with two patient outcomes:

1. **In-hospital death**
2. **Length of stay (days)**

## Input Variables

| Variable | Description | Unit / Range |
| --- | --- | --- |
| SAPS-I | Simplified Acute Physiology Score | — |
| SOFA | Sequential Organ Failure Assessment score | — |
| GCS | Glasgow Coma Scale score | 3–15 |
| Creatinine | Serum creatinine | mg/dL |
| Urine | Urine output | mL |
| NIMAP | Non-invasive mean arterial blood pressure | mmHg |
| BUN | Blood urea nitrogen | mg/dL |
| HR | Heart rate | bpm |
| RespRate | Respiratory rate | breaths/minute |

## Outcomes

| Outcome | Description |
| --- | --- |
| In-hospital death | Whether a patient died during the hospital stay. |
| Length of stay (days) | Days from ICU admission to the end of hospitalization, including any time spent in the hospital after ICU discharge. |

## Scope

All predictors are derived from the recorded data during the 48-hour observation period. Analyses should evaluate their relationship with mortality and hospital length of stay.

## Running the Project

Open a command-line terminal and change to the project root directory. Then run the following command to execute all project code in one step:

```bash
python run_all.py
```
