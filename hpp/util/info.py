import socket
import getpass
import subprocess
import os
from datetime import datetime
import time
import yaml
import pickle
import shutil
import re

def get_pc_and_version():
    '''Returns current PC's hostname, username and code's current version.'''
    hostname = socket.gethostname()
    username = getpass.getuser()

    # Store the version of the session
    commit_hash = subprocess.check_output(["git", "describe", '--always']).strip().decode('ascii')
    try:
        subprocess.check_output(["git", "diff", "--quiet"])
    except:
        commit_hash += '-dirty'
    version = commit_hash

    return hostname, username, version, os.getpid()

def port_is_in_use(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) == 0

def get_dir_size(dir = '.'):
    '''Returns the size of the given directory in bytes.'''
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(dir):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            # skip if it is symbolic link
            if not os.path.islink(fp):
                total_size += os.path.getsize(fp)
    return total_size

def bytes2human(bytes):
    '''Returns a string with human readable form of the given bytes.'''
    if bytes > 1024 * 1024 * 1024 * 1024:
        return str("{:.2f} TB".format(bytes / (1024 * 1024 * 1024 * 1024)))
    elif bytes > 1024 * 1024 * 1024:
        return str("{:.2f} GB".format(bytes / (1024 * 1024 * 1024)))
    elif bytes > 1024 * 1024:
        return str("{:.2f} MB".format(bytes / (1024 * 1024)))
    elif bytes > 1024:
        return str("{:.2f} KB".format(bytes / 1024))
    else:
        return str(bytes) + ' Bytes'

def get_now_timestamp():
    """
    Returns a timestamp for the current datetime as a string for using it in
    log file naming.
    """
    now_raw = datetime.now()
    return str(now_raw.year) + '.' + \
           '{:02d}'.format(now_raw.month) + '.' + \
           '{:02d}'.format(now_raw.day) + '.' + \
           '{:02d}'.format(now_raw.hour) + '.' \
           '{:02d}'.format(now_raw.minute) + '.' \
           '{:02d}'.format(now_raw.second) + '.' \
           '{:02d}'.format(now_raw.microsecond)

def transform_sec_to_timestamp(seconds):
    """
    Transforms seconds to a timestamp string in format: hours:minutes:seconds

    Parameters
    ----------
    seconds : float
        The seconds to transform

    Returns
    -------
    str
        The timestamp
    """
    hours, rem = divmod(seconds, 3600)
    minutes, seconds = divmod(rem, 60)
    return "{:0>2}h:{:0>2}m:{:05.2f}s".format(int(hours),int(minutes),seconds)

class Logger:
    def __init__(self, log_dir):
        self.log_dir = log_dir

        # Create the log directory
        if os.path.exists(log_dir):
            print('Directory ', log_dir, 'exists, do you want to remove it? (y/p/n)')
            answer = input('')
            if answer == 'y':
                shutil.rmtree(log_dir)
                os.mkdir(log_dir)
            elif answer == 'p':
                pass
            else:
                exit()

        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

    def log_data(self, data, filename):
        pickle.dump(data, open(os.path.join(self.log_dir, filename), 'wb'))

    def log_yml(self, dict, filename):
        with open(os.path.join(self.log_dir, filename + '.yml'), 'w') as stream:
            yaml.dump(dict, stream)

    def update(self):
        # self.exp_info['dir_size'] = get_dir_size(self.log_dir)
        # self.exp_info['time_elapsed'] = transform_sec_to_timestamp(time.time() - self.start_time)
        # self.log_yml(self.exp_info, 'exp_info')
        pass


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def info(*args):
    print(bcolors.OKBLUE + " ".join(map(str, args)) + bcolors.ENDC)

def warn(*args):
    print(bcolors.WARNING + bcolors.BOLD + "WARNING: " + " ".join(map(str, args)) + bcolors.ENDC)

def error(*args):
    print(bcolors.FAIL + bcolors.BOLD + "ERROR: " + " ".join(map(str, args)) + bcolors.ENDC)

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]