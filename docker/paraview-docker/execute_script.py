#!/usr/bin/env python3
"""
Execute a ParaView Python script on pvserver.

This script:
1. Connects to pvserver
2. Executes your ParaView Python script
3. Captures all output and errors

Usage:
    python execute_script.py --script example_volume_script.py
    python execute_script.py --script example_volume_script.py --host localhost --port 11111
"""

import argparse
import sys
import subprocess
import os
import tempfile

def create_wrapper_script(script_path, host, port):
    """
    Create a temporary wrapper script that:
    1. Connects to pvserver
    2. Executes the user's script
    3. Handles errors
    """
    with open(script_path, 'r') as f:
        user_script = f.read()

    wrapper = f'''
import sys
import traceback
from paraview.simple import *

print("[CLIENT] Connecting to pvserver at {host}:{port}...")
try:
    connection = Connect('{host}', {port})
    if connection is None:
        print("[CLIENT] ✗ Failed to connect to pvserver!")
        sys.exit(1)
    print(f"[CLIENT] ✓ Connected to pvserver: {{connection}}")
except Exception as e:
    print(f"[CLIENT] ✗ Connection error: {{e}}")
    traceback.print_exc()
    sys.exit(1)

print("[CLIENT] Executing script...")
print("="*70)

try:
    # Execute user script
{indent_code(user_script, 4)}

    print("="*70)
    print("[CLIENT] ✓ Script executed successfully")

except SystemExit as e:
    # Re-raise SystemExit to preserve exit codes
    raise

except Exception as e:
    print("="*70)
    print(f"[CLIENT] ✗ Error during script execution: {{e}}")
    traceback.print_exc()
    sys.exit(1)
'''

    return wrapper

def indent_code(code, spaces):
    """Indent code block by specified number of spaces"""
    indent = ' ' * spaces
    lines = code.split('\n')
    return '\n'.join(indent + line if line.strip() else '' for line in lines)

def execute_script(script_path, host, port, timeout):
    """
    Execute the ParaView Python script using pvpython.
    """
    print("="*70)
    print(f"EXECUTING SCRIPT: {script_path}")
    print(f"Target: {host}:{port}")
    print("="*70)

    try:
        # Create wrapper script
        wrapper_code = create_wrapper_script(script_path, host, port)

        # Write to temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as tmp:
            tmp.write(wrapper_code)
            tmp_path = tmp.name

        try:
            # Execute the wrapper script
            result = subprocess.run(
                ['pvpython', tmp_path],
                capture_output=True,
                text=True,
                timeout=timeout
            )

            print("\n" + "="*70)
            print("OUTPUT")
            print("="*70)

            if result.stdout:
                print(result.stdout)

            if result.stderr:
                print("\n" + "="*70)
                print("ERRORS/WARNINGS")
                print("="*70)
                print(result.stderr)

            print("\n" + "="*70)
            print(f"EXIT CODE: {result.returncode}")
            print("="*70)

            return result.returncode

        finally:
            # Clean up temp file
            try:
                os.unlink(tmp_path)
            except:
                pass

    except subprocess.TimeoutExpired:
        print(f"\n✗ ERROR: Script execution timed out ({timeout}s)")
        return 1
    except FileNotFoundError:
        print("\n✗ ERROR: 'pvpython' not found in PATH")
        print("Please ensure ParaView is installed and pvpython is accessible")
        return 1
    except Exception as e:
        print(f"\n✗ ERROR: Failed to execute script: {e}")
        import traceback
        traceback.print_exc()
        return 1

def main():
    parser = argparse.ArgumentParser(
        description='Execute ParaView script on pvserver'
    )
    parser.add_argument('--script', required=True,
                       help='Path to ParaView Python script to execute')
    parser.add_argument('--host', default='localhost',
                       help='pvserver host (default: localhost)')
    parser.add_argument('--port', type=int, default=11111,
                       help='pvserver port (default: 11111)')
    parser.add_argument('--timeout', type=int, default=60,
                       help='Execution timeout in seconds (default: 60)')

    args = parser.parse_args()

    # Check if script exists
    if not os.path.exists(args.script):
        print(f"✗ ERROR: Script not found: {args.script}")
        return 1

    # Execute the script
    exit_code = execute_script(args.script, args.host, args.port, args.timeout)

    print("\n" + "="*70)
    if exit_code == 0:
        print("✓ EXECUTION COMPLETED SUCCESSFULLY")
    else:
        print(f"✗ EXECUTION FAILED (exit code: {exit_code})")
    print("="*70)

    return exit_code

if __name__ == '__main__':
    sys.exit(main())
