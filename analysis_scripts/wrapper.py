# mypackage/myscript_wrapper.py
import subprocess
import os

def eval_vol(args):
    script_path = os.path.join(os.path.dirname(__file__), 'eval_vol.sh')
    subprocess.call(['bash', script_path])

def analyze(args):
    script_path = os.path.join(os.path.dirname(__file__), 'analyze.sh')
    subprocess.call(['bash', script_path])

def parse_pose(args):
    script_path = os.path.join(os.path.dirname(__file__), 'parse_pose.sh')
    subprocess.call(['bash', script_path])

def prepare(args):
    script_path = os.path.join(os.path.dirname(__file__), 'prepare.sh')
    subprocess.call(['bash', script_path])
