#!/bin/zsh
# compare.sh TAG: logit_compare.py on TAG's dumps (CPU only): main vs pr, and within pr: step / verify vs batch
D=${0:A:h}; O=$D/out/$1/exact; C="nice python3 $D/tools/logit_compare.py"
${=C} $O/ld-main $O/ld-pr --no-gate --json $O/cmp-main-pr.json > $O/cmp-main-pr.txt 2>&1
${=C} $O/ld-pr/step $O/ld-pr/batch --no-gate --json $O/cmp-step-batch4.json > $O/cmp-step-batch4.txt 2>&1
${=C} $O/ld-pr/step $O/ld-pr-b2/batch --no-gate --json $O/cmp-step-batch2.json > $O/cmp-step-batch2.txt 2>&1
${=C} $O/ld-pr/verify $O/ld-pr/batch --no-gate --json $O/cmp-verify-batch4.json > $O/cmp-verify-batch4.txt 2>&1
${=C} $O/ld-pr/step $O/ld-pr/verify --no-gate --json $O/cmp-step-verify.json > $O/cmp-step-verify.txt 2>&1
echo CMPDONE
