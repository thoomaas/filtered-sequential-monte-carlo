import os
import argparse
from tests.regression.runner import run_case
from tests.regression.utils import get_case_dirs

def run_all_tests(update_baseline=False):
    case_dirs = get_case_dirs()
    if not case_dirs:
        print("No test cases found in tests/regression/cases/")
        return

    print(f"Starting regression tests on {len(case_dirs)} cases...\n")
    
    results = {}
    for case_dir in case_dirs:
        case_name = os.path.basename(case_dir)
        print(f"--- Running Case: {case_name} ---")
        try:
            success = run_case(case_dir, update_baseline)
            results[case_name] = "PASS" if success else "FAIL"
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"ERROR in {case_name}: {e}")
            results[case_name] = f"ERROR: {e}"
        print("")

    print("="*30)
    print("REGRESSION TEST SUMMARY")
    print("="*30)
    passed = 0
    for name, status in results.items():
        print(f"{name:20}: {status}")
        if status == "PASS":
            passed += 1
    
    print("-" * 30)
    print(f"Total: {len(results)} | Passed: {passed} | Failed: {len(results) - passed}")
    print("="*30)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--update-baseline", action="store_true", help="Update all 'expected' results")
    args = parser.parse_args()
    run_all_tests(args.update_baseline)
