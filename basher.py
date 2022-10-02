import subprocess
import shlex

def run(command, workingDir="./"):
        result = subprocess.run(shlex.split(command), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, cwd=workingDir)
        return result
