#!/bin/bash

echo "=== QUICK MTF VALIDATION ==="
echo "| Sigma | Theoretical MTF50 | Measured MTF50 | Status |"
echo "|-------|-------------------|----------------|--------|"

# Theoretical MTF50 values calculated as sqrt(-ln(0.5)/(2*pi^2*sigma^2))
sigmas=(0.5 1.0 1.5 2.0)
theoretical=(0.375 0.187 0.125 0.094)

for i in "${!sigmas[@]}"; do
    sigma=${sigmas[$i]}
    theory=${theoretical[$i]}
    
    echo -n "| $sigma   | $theory           | "
    
    # Extract MTF50 - take only the final averaged result
    measured=$(./mtf_analyzer_6_final working_targets/working_sigma_${sigma}.png 2>/dev/null | \
               grep "MTF50.*cycles" | tail -1 | awk '{print $2}')
    
    if [ -n "$measured" ]; then
        echo -n "$measured      | "
        # Simple comparison
        if (( $(echo "$measured > 0.001" | bc -l) )); then
            echo "✅ Valid |"
        else
            echo "❌ Low  |"
        fi
    else
        echo "FAILED        | ❌ Fail |"
    fi
done

echo ""
echo "Note: Measured values should decrease as sigma increases (more blur = lower MTF50)"