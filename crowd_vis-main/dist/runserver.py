import subprocess
import os

def run_bash_script():
    bash_script_path = os.path.abspath("bash.sh")
    subprocess.run(["bash", bash_script_path])

if __name__ == "__main__":
    run_bash_script()