"""Print a greeting; used to test run_skill_script."""
import sys

print(f"hello {sys.argv[1] if len(sys.argv) > 1 else 'world'}")
