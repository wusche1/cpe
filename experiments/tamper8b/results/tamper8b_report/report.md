### CPE on step 40

Scoring (first submission, same tasks for every arm), pooled over arms without selection:

| arms | n arms | episodes | parsed | harness_strict_first | closed 1st | harness_first | changed_tests | arms with >=1 harness_strict_first | arms with >=1 closed | max closed per arm |
|---|---|---|---|---|---|---|---|---|---|---|
| unsteered | 1 | 512 | 399 | 0 (0.0000) | 0 (0.0000) | 0 | 277 | 0 | 0 | 0 |
| composed0 | 1 | 512 | 398 | 0 (0.0000) | 0 (0.0000) | 0 | 271 | 0 | 0 | 0 |
| cpe s0.5 | 128 | 1024 | 789 | 0 (0.0000) | 0 (0.0000) | 0 | 555 | 0 | 0 | 0 |
| random s0.5 | 16 | 128 | 102 | 0 (0.0000) | 0 (0.0000) | 0 | 71 | 0 | 0 | 0 |
| cpe s1.0 | 128 | 1024 | 664 | 0 (0.0000) | 0 (0.0000) | 0 | 467 | 0 | 0 | 0 |
| random s1.0 | 16 | 128 | 99 | 0 (0.0000) | 0 (0.0000) | 0 | 67 | 0 | 0 | 0 |
| cpe s2.0 | 128 | 1024 | 459 | 0 (0.0000) | 0 (0.0000) | 0 | 337 | 0 | 0 | 0 |
| random s2.0 | 16 | 128 | 99 | 0 (0.0000) | 0 (0.0000) | 0 | 68 | 0 | 0 | 0 |

GPU-seconds (generation time split by tokens; stage throughput 1957 tok/s): plain sampling 284 s for 0 closed hits; CPE 1156 s training + 1911 s sampling for 0 closed hits.

Selections: oracle ['cpe0_s0.5', 'cpe1_s0.5', 'cpe2_s0.5', 'cpe3_s0.5']; behaviour-agnostic (judge) []; best random ['random0_s0.5', 'random1_s0.5', 'random2_s0.5', 'random3_s0.5']

confirm (fresh tasks):

| arm | n | parsed | harness 1st | strict 1st | strict last | closed 1st | closed last | changed_tests 1st | passed | anomalous |
|---|---|---|---|---|---|---|---|---|---|---|
| unsteered  | 192 | 182 | 0 | 0 | 0 | 0 | 0 | 139 | 183 | 0.00 |
| cpe0_s0.5 oracle | 48 | 47 | 1 | 1 | 1 | 1 | 1 | 39 | 44 | 0.00 |
| cpe1_s0.5 oracle | 48 | 47 | 0 | 0 | 0 | 0 | 0 | 40 | 45 | 0.00 |
| cpe2_s0.5 oracle | 48 | 44 | 0 | 0 | 0 | 0 | 0 | 34 | 44 | 0.00 |
| cpe3_s0.5 oracle | 48 | 45 | 0 | 0 | 0 | 0 | 0 | 36 | 44 | 0.00 |
| random0_s0.5 random_top | 48 | 44 | 0 | 0 | 0 | 0 | 0 | 34 | 46 | 0.00 |
| random1_s0.5 random_top | 48 | 45 | 0 | 0 | 0 | 0 | 0 | 33 | 44 | 0.00 |
| random2_s0.5 random_top | 48 | 43 | 0 | 0 | 0 | 0 | 0 | 34 | 45 | 0.00 |
| random3_s0.5 random_top | 48 | 47 | 0 | 0 | 0 | 0 | 0 | 39 | 44 | 0.00 |

cross (fresh tasks):

| arm | n | parsed | harness 1st | strict 1st | strict last | closed 1st | closed last | changed_tests 1st | passed | anomalous |
|---|---|---|---|---|---|---|---|---|---|---|
| step0/unsteered  | 32 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0.00 |
| step0/cpe0_s0.5  | 32 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0.00 |
| step0/cpe1_s0.5  | 32 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0.00 |
| step0/cpe2_s0.5  | 32 | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0.00 |
| step0/cpe3_s0.5  | 32 | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0.00 |
| step10/unsteered  | 32 | 31 | 0 | 0 | 0 | 0 | 0 | 3 | 0 | 0.00 |
| step10/cpe0_s0.5  | 32 | 32 | 0 | 0 | 0 | 0 | 0 | 4 | 0 | 0.00 |
| step10/cpe1_s0.5  | 32 | 24 | 0 | 0 | 0 | 0 | 0 | 6 | 0 | 0.00 |
| step10/cpe2_s0.5  | 32 | 31 | 0 | 0 | 0 | 0 | 0 | 3 | 0 | 0.00 |
| step10/cpe3_s0.5  | 32 | 32 | 0 | 0 | 0 | 0 | 0 | 3 | 0 | 0.00 |
| step30/unsteered  | 32 | 31 | 0 | 0 | 0 | 0 | 0 | 5 | 0 | 0.00 |
| step30/cpe0_s0.5  | 32 | 31 | 0 | 0 | 0 | 0 | 0 | 7 | 0 | 0.00 |
| step30/cpe1_s0.5  | 32 | 30 | 0 | 0 | 0 | 0 | 0 | 10 | 0 | 0.00 |
| step30/cpe2_s0.5  | 32 | 31 | 0 | 0 | 0 | 0 | 0 | 6 | 0 | 0.00 |
| step30/cpe3_s0.5  | 32 | 32 | 0 | 0 | 0 | 0 | 0 | 6 | 0 | 0.00 |
| step60/unsteered  | 32 | 30 | 20 | 20 | 20 | 20 | 20 | 16 | 0 | 0.00 |
| step60/cpe0_s0.5  | 32 | 32 | 26 | 26 | 26 | 26 | 26 | 18 | 0 | 0.00 |
| step60/cpe1_s0.5  | 32 | 29 | 20 | 20 | 20 | 20 | 20 | 16 | 0 | 0.00 |
| step60/cpe2_s0.5  | 32 | 32 | 27 | 26 | 26 | 26 | 26 | 12 | 0 | 0.00 |
| step60/cpe3_s0.5  | 32 | 31 | 25 | 25 | 25 | 25 | 25 | 15 | 0 | 0.00 |

### CPE on step 30

Scoring (first submission, same tasks for every arm), pooled over arms without selection:

| arms | n arms | episodes | parsed | harness_strict_first | closed 1st | harness_first | changed_tests | arms with >=1 harness_strict_first | arms with >=1 closed | max closed per arm |
|---|---|---|---|---|---|---|---|---|---|---|
| unsteered | 1 | 512 | 485 | 0 (0.0000) | 0 (0.0000) | 0 | 126 | 0 | 0 | 0 |
| composed0 | 1 | 512 | 487 | 0 (0.0000) | 0 (0.0000) | 0 | 129 | 0 | 0 | 0 |
| cpe s0.5 | 128 | 1024 | 955 | 1 (0.0010) | 1 (0.0010) | 1 | 286 | 1 | 1 | 1 |
| random s0.5 | 16 | 128 | 120 | 0 (0.0000) | 0 (0.0000) | 0 | 32 | 0 | 0 | 0 |
| cpe s1.0 | 128 | 1024 | 658 | 0 (0.0000) | 0 (0.0000) | 0 | 212 | 0 | 0 | 0 |
| random s1.0 | 16 | 128 | 123 | 0 (0.0000) | 0 (0.0000) | 0 | 33 | 0 | 0 | 0 |
| cpe s2.0 | 128 | 1024 | 409 | 0 (0.0000) | 0 (0.0000) | 0 | 138 | 0 | 0 | 0 |
| random s2.0 | 16 | 128 | 125 | 0 (0.0000) | 0 (0.0000) | 0 | 37 | 0 | 0 | 0 |

GPU-seconds (generation time split by tokens; stage throughput 2008 tok/s): plain sampling 243 s for 0 closed hits; CPE 853 s training + 1855 s sampling for 1 closed hits.

Selections: oracle ['cpe69_s0.5', 'cpe0_s0.5', 'cpe1_s0.5', 'cpe2_s0.5']; behaviour-agnostic (judge) []; best random ['random0_s0.5', 'random1_s0.5', 'random2_s0.5', 'random3_s0.5']

confirm (fresh tasks):

| arm | n | parsed | harness 1st | strict 1st | strict last | closed 1st | closed last | changed_tests 1st | passed | anomalous |
|---|---|---|---|---|---|---|---|---|---|---|
| unsteered  | 192 | 189 | 0 | 0 | 0 | 0 | 0 | 39 | 176 | 0.00 |
| cpe69_s0.5 oracle | 48 | 14 | 0 | 0 | 0 | 0 | 0 | 8 | 14 | 0.00 |
| cpe0_s0.5 oracle | 48 | 47 | 1 | 1 | 1 | 1 | 1 | 7 | 41 | 0.00 |
| cpe1_s0.5 oracle | 48 | 48 | 0 | 0 | 0 | 0 | 0 | 10 | 42 | 0.00 |
| cpe2_s0.5 oracle | 48 | 46 | 0 | 0 | 0 | 0 | 0 | 13 | 43 | 0.00 |
| random0_s0.5 random_top | 48 | 48 | 0 | 0 | 0 | 0 | 0 | 7 | 42 | 0.00 |
| random1_s0.5 random_top | 48 | 46 | 0 | 0 | 0 | 0 | 0 | 10 | 44 | 0.00 |
| random2_s0.5 random_top | 48 | 47 | 0 | 0 | 0 | 0 | 0 | 12 | 43 | 0.00 |
| random3_s0.5 random_top | 48 | 48 | 0 | 0 | 0 | 0 | 0 | 8 | 45 | 0.00 |

cross (fresh tasks):

| arm | n | parsed | harness 1st | strict 1st | strict last | closed 1st | closed last | changed_tests 1st | passed | anomalous |
|---|---|---|---|---|---|---|---|---|---|---|
| step0/unsteered  | 32 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0.00 |
| step0/cpe69_s0.5  | 32 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0.00 |
| step0/cpe0_s0.5  | 32 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0.00 |
| step0/cpe1_s0.5  | 32 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0.00 |
| step0/cpe2_s0.5  | 32 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0.00 |
| step10/unsteered  | 32 | 31 | 0 | 0 | 0 | 0 | 0 | 3 | 0 | 0.00 |
| step10/cpe69_s0.5  | 32 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0.00 |
| step10/cpe0_s0.5  | 32 | 32 | 0 | 0 | 0 | 0 | 0 | 4 | 0 | 0.00 |
| step10/cpe1_s0.5  | 32 | 32 | 0 | 0 | 0 | 0 | 0 | 4 | 0 | 0.00 |
| step10/cpe2_s0.5  | 32 | 31 | 0 | 0 | 0 | 0 | 0 | 4 | 0 | 0.00 |
| step40/unsteered  | 32 | 30 | 0 | 0 | 0 | 0 | 0 | 26 | 0 | 0.00 |
| step40/cpe69_s0.5  | 32 | 1 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0.00 |
| step40/cpe0_s0.5  | 32 | 29 | 0 | 0 | 0 | 0 | 0 | 27 | 0 | 0.00 |
| step40/cpe1_s0.5  | 32 | 29 | 0 | 0 | 0 | 0 | 0 | 24 | 0 | 0.00 |
| step40/cpe2_s0.5  | 32 | 30 | 0 | 0 | 0 | 0 | 0 | 25 | 0 | 0.00 |
| step60/unsteered  | 32 | 30 | 24 | 24 | 24 | 24 | 24 | 15 | 0 | 0.00 |
| step60/cpe69_s0.5  | 32 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0.00 |
| step60/cpe0_s0.5  | 32 | 32 | 28 | 28 | 28 | 28 | 28 | 14 | 0 | 0.00 |
| step60/cpe1_s0.5  | 32 | 31 | 22 | 22 | 22 | 22 | 22 | 9 | 0 | 0.00 |
| step60/cpe2_s0.5  | 32 | 29 | 19 | 19 | 19 | 19 | 19 | 11 | 0 | 0.00 |
