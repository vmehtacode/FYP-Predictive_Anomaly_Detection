#!/bin/bash
# Quick validation test for V3 fixes

echo "========================================================================"
echo "VERSION 3 VALIDATION TEST"
echo "========================================================================"
echo ""
echo "Testing with 30 episodes to verify:"
echo "  1. Curriculum progresses: 0.0 → 0.1 → 0.2 → 0.3"
echo "  2. Model weights saved/loaded"
echo "  3. Final MAE ≈ Best MAE (within 10%)"
echo "  4. Training MAE > 0"
echo ""
echo "Expected runtime: ~10-15 minutes"
echo ""

cd "$(dirname "$0")/.."

echo "Starting V3 test..."
poetry run python scripts/run_large_scale_experiment_v3.py \
  --num-households 5 \
  --num-episodes 30 \
  --num-runs 1 \
  --output-dir results/test_v3 \
  2>&1 | tee results/test_v3.log

echo ""
echo "========================================================================"
echo "VALIDATION RESULTS"
echo "========================================================================"
echo ""

# Check 1: Curriculum progression
echo "1. Curriculum Progression:"
curriculum_count=$(grep -c "📈 Curriculum" results/test_v3.log)
if [ $curriculum_count -ge 2 ]; then
    echo "   ✅ PASS: Found $curriculum_count curriculum increases"
    grep "📈 Curriculum" results/test_v3.log | head -3
else
    echo "   ❌ FAIL: Only found $curriculum_count curriculum increase(s)"
fi
echo ""

# Check 2: Model weights  
echo "2. Model Weights Saving:"
weight_save_count=$(grep -c "✅ Saved PyTorch model weights" results/test_v3.log 2>/dev/null || echo "0")
if [ $weight_save_count -gt 0 ]; then
    echo "   ✅ PASS: Model weights saved ($weight_save_count times)"
else
    echo "   ❌ FAIL: Model weights not saved"
    # Check for warnings
    grep "Could not save\|has no model" results/test_v3.log 2>/dev/null | head -2 | sed 's/^/      /'
fi

if grep -q "✅ Loaded model weights" results/test_v3.log 2>/dev/null; then
    echo "   ✅ PASS: Model weights loaded"
else
    echo "   ⚠️  WARN: Model weights not loaded (may be metadata only)"
fi
echo ""

# Check 3: Final vs Best MAE
echo "3. Checkpoint Validation:"
if [ -f results/test_v3/metrics_summary.json ]; then
    poetry run python -c "
import json
d = json.load(open('results/test_v3/metrics_summary.json'))
r = d['detailed_results']['selfplay'][0]
best = r['best_val_mae']
final = r['val_mae']
diff_pct = abs(final - best) / best * 100
curriculum = r['final_curriculum_level']

print(f'   Best MAE:  {best:.4f} kWh (episode {r[\"best_episode\"]})')
print(f'   Final MAE: {final:.4f} kWh')
print(f'   Difference: {diff_pct:.1f}%')
print(f'   Curriculum: {curriculum:.1f}')
print()

if diff_pct < 10:
    print('   ✅ PASS: Final within 10% of best')
else:
    print(f'   ❌ FAIL: Final is {diff_pct:.1f}% different from best')

# Expected curriculum for 30 episodes: 0.2 (updates at 10, 20)
expected = min(0.9, (30 // 10) * 0.1)
if abs(curriculum - expected) < 0.05:
    print(f'   ✅ PASS: Curriculum at {curriculum:.1f} (expected {expected:.1f} for 30 episodes)')
else:
    print(f'   ❌ FAIL: Curriculum at {curriculum:.1f} (expected {expected:.1f})')
"
else
    echo "   ❌ FAIL: Results file not found"
fi
echo ""

# Check 4: Training MAE
echo "4. Training MAE:"
if grep -q "train_mae=0.0000" results/test_v3.log; then
    echo "   ⚠️  WARN: Some zero training MAEs detected"
else
    echo "   ✅ PASS: No zero training MAEs"
fi

# Show sample training MAEs
echo "   Sample values:"
grep "📊 Episode" results/test_v3.log | head -5 | sed 's/^/     /'
echo ""

echo "========================================================================"
echo "SUMMARY"
echo "========================================================================"
echo ""
echo "If all checks passed, run full experiment:"
echo ""
echo "  poetry run python scripts/run_large_scale_experiment_v3.py \\"
echo "    --num-households 50 \\"
echo "    --num-episodes 100 \\"
echo "    --num-runs 5 \\"
echo "    --output-dir results/large_scale_experiment_v3"
echo ""
echo "Estimated runtime: 8-12 hours (run overnight)"
echo ""

