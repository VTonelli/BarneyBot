import os
import subprocess
import sys
import shlex

BASE_FOLDER = None

def set_folder(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return path

def run_command(line):
    args = shlex.split(line)
    subprocess.check_call(args)
    
def _install_requirements():
    print("pip install -r " + "\"" + os.path.join(BASE_FOLDER, "requirements.txt") + "\"")
    run_command("pip install -r " + "\"" + os.path.join(BASE_FOLDER, "requirements.txt") + "\"")
    
def _common_setup():
    global BASE_FOLDER
    _install_requirements()
    os.environ["HF_DATASETS_CACHE"] = os.path.join(BASE_FOLDER, "cache")    
    
def colab_setup(mount_folder):
    global BASE_FOLDER
    from google.colab import drive
    drive.mount('/content/drive', force_remount=True)
    BASE_FOLDER = mount_folder
    _common_setup()

def anaconda_manual_setup(base_folder, env_name):
    global BASE_FOLDER
    BASE_FOLDER = base_folder
    _common_setup()
    anaconda_base_folder = next(p for p in sys.path if p.endswith("Anaconda"))
    sys.path.insert(1, os.path.join(anaconda_base_folder, "envs", env_name,
                                    "Lib", "site-packages"))

def anaconda_auto_setup(base_folder):
    global BASE_FOLDER
    BASE_FOLDER = base_folder   
    _common_setup()
    