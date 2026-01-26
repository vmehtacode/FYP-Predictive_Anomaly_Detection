#!/bin/bash
# Quick test script to validate all fixes work correctly

echo "========================================================================"
echo "TESTING FIXED SELF-PLAY EXPERIMENT"
echo "========================================================================"
echo ""
echo "This will run a quick test with:"
echo "  - 5 households (instead of 50)"
echo "  - 30 episodes (instead of 100)"  
echo "  - 1 run (instead of 5)"
echo ""
echo "Expected runtime: ~30 minutes"
echo ""
echo "What to look for:"
echo "  ✅ Curriculum increases: 0.0 → 0.1 → 0.2"
echo "  ✅ Training MAE > 0"
echo "  ✅ 'Loaded best model' message"
echo "  ✅ Final MAE ≈ Best MAE (within 10%)"
echo ""
read -p "Press Enter to start..."

cd "$(dirname "$0")/.."

poetry run python scripts/run_large_scale_experiment_FIXED.py \
  --num-households 5 \
  --num-episodes 30 \
  --num-runs 1 \
  --output-dir results/test_fixes \
  2>&1 | tee results/test_fixes.log

echo ""
echo "========================================================================"
echo "TEST COMPLETE"
echo "========================================================================"
echo ""
echo "Checking validation criteria..."
echo ""

# Check for curriculum progression
if grep -q "📈 Curriculum increased" results/test_fixes.log; then
    echo "✅ Curriculum progression detected"
else
    echo "❌ Curriculum did NOT progress - check logs"
fi

# Check for best model loading
if grep -q "✅ Loaded best model" results/test_fixes.log; then
    echo "✅ Best model checkpoint loaded"
else
    echo "❌ Best model NOT loaded - check logs"
fi

# Check for validation checks
if grep -q "VALIDATION CHECKS" results/test_fixes.log; then
    echo "✅ Validation checks ran"
    grep "Final val_mae:" results/test_fixes.log
    grep "Best val_mae:" results/test_fixes.log
    grep "Difference:" results/test_fixes.log
else
    echo "❌ Validation checks missing"
fi

echo ""
echo "Full log saved to: results/test_fixes.log"
echo "Results saved to: results/test_fixes/"
echo ""
echo "If all checks passed, run the full experiment with:"
echo ""
echo "  poetry run python scripts/run_large_scale_experiment_FIXED.py \\"
echo "    --num-households 50 \\"
echo "    --num-episodes 100 \\"
echo "    --num-runs 5 \\"
echo "    --output-dir results/large_scale_experiment_v2"
echo ""

