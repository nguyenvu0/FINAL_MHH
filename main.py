"""Main entry point for the Petri Net Analyzer."""

import argparse
import sys
from pathlib import Path

from task1_parser.pnml_parser import PNMLParser
from task2_explicit.explicit_reachability import ExplicitReachability
from task3_symbolic.symbolic_reachability import SymbolicReachability
from task4_deadlock.deadlock_detection import DeadlockDetector
from task5_optimization.optimization import ReachabilityOptimizer


def main() -> None:
    parser = argparse.ArgumentParser(description="Petri Net Analyzer - CO2011 Assignment")
    parser.add_argument("pnml_file", help="Path to the PNML file")
    args = parser.parse_args()

    filepath = Path(args.pnml_file)
    if not filepath.exists():
        print(f"Error: File '{filepath}' not found.")
        sys.exit(1)

    print(f"Loading PNML: {filepath.name}\n")

    # ============================================
    # Task 1: Parsing
    # ============================================
    try:
        net = PNMLParser.parse(str(filepath))
        net.print_summary()
    except Exception as e:
        print(f"Parsing failed: {e}")
        sys.exit(1)

    # ============================================
    # Task 2: Explicit Reachability (BFS & DFS)
    # ============================================
    print("\n" + "=" * 60)
    print("TASK 2: EXPLICIT REACHABILITY")
    print("=" * 60)
    
    # BFS
    explicit_bfs = ExplicitReachability(net)
    explicit_bfs.compute_bfs()
    explicit_bfs.print_results("BFS")
    
    # DFS
    print()  # Spacing
    explicit_dfs = ExplicitReachability(net)
    explicit_dfs.compute_dfs()
    explicit_dfs.print_results("DFS")

    # ============================================
    # Task 3: Symbolic Reachability (BDD)
    # ============================================
    print("\n" + "=" * 60)
    print("TASK 3: SYMBOLIC REACHABILITY (BDD)")
    print("=" * 60)
    
    sym = SymbolicReachability(net)
    sym.build_symbolic()
    
    print(f"BDD constructed: {'yes' if sym.bdd else 'no'}")
    print(f"Reachable markings (BDD): {sym.num_states}")
    print(f"Construction time: {sym.computation_time:.4f} seconds")
    
    # Verification
    if sym.num_states != explicit_bfs.num_states:
        print(f"\n⚠️  WARNING: State count mismatch!")
        print(f"   Explicit: {explicit_bfs.num_states}")
        print(f"   Symbolic: {sym.num_states}")
    else:
        print(f"\n✓ SUCCESS: BDD matches explicit ({sym.num_states} states)")

    # ============================================
    # Task 4: Deadlock Detection (ILP + BDD)
    # ============================================
    print("\n" + "=" * 60)
    print("TASK 4: DEADLOCK DETECTION (ILP + BDD)")
    print("=" * 60)
    
    detector = DeadlockDetector(net)
    deadlock = detector.detect_with_ilp_bdd(sym, max_candidates=500)
    
    print(f"Time: {detector.computation_time:.4f}s")
    if deadlock:
        print("Deadlock found:")
        for place, value in sorted(deadlock.items()):
            if value > 0:
                print(f"  {place}: {value}")
    else:
        print("No deadlock reachable.")

    # ============================================
    # Task 5: Optimization (ILP + BDD)
    # ============================================
    print("\n" + "=" * 60)
    print("TASK 5: OPTIMIZATION (ILP + BDD)")
    print("=" * 60)
    
    # Define objective: maximize sum of tokens
    weights = {p: 1 for p in net.places}
    print("Objective: Maximize sum of tokens (all weights = 1)")
    
    optimizer = ReachabilityOptimizer(net)
    best_marking = optimizer.optimize_with_ilp(sym, weights, maximize=True, max_candidates=500)
    
    print(f"Time: {optimizer.computation_time:.4f}s")
    print(f"Max value: {optimizer.optimal_value}")
    if best_marking:
        print("Max marking:", list(best_marking.values()))
    else:
        print("No optimal marking found.")
    
    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()