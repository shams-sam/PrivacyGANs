# Reference: https://github.com/human-analysis/kernel-adversarial-representation-learning/blob/master/utils.py

import os
import sys
import psutil
import signal


def setup_graceful_exit():
    # handle Ctrl-C signal
    signal.signal(signal.SIGINT, ctrl_c_handler)


def cleanup():
    # signal.signal(signal.SIGINT, signal.SIG_DFL)
    current_process = psutil.Process()
    children = current_process.children(recursive=True)
    for child in children:
        try:
            os.kill(int(child.pid), signal.SIGKILL)
        except OSError as ex:
            raise Exception(
                "wasn't able to kill the child process (pid:{}).".format(child.pid))
    #     # os.waitpid(child.pid, os.P_ALL)

    print('\x1b[?25h', end='', flush=True)  # show cursor
    sys.exit(0)


def ctrl_c_handler(*kargs):
    # try to gracefully terminate the program
    # signal.signal(signal.SIGINT, signal.SIG_DFL)
    cleanup()
