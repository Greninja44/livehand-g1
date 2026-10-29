import os
import sys

from .app import main

if __name__ == "__main__":
    rc = main()
    # main() has already stopped the threads, released the camera and closed all windows. MuJoCo 3.17's passive
    # viewer crashes during interpreter teardown on this setup (Python 3.14, GLFW; reproduced with a bare
    # launch_passive script: SIGSEGV / "double free"), so skip the teardown instead of reporting a bogus crash.
    sys.stdout.flush(); sys.stderr.flush()
    os._exit(rc or 0)
