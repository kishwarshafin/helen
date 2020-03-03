import subprocess as sp
import sys
import os
"""
This is a wrapper to call MarginPolish from the installed binary directory.
"""


def main():
    """
    Run MarginPolish from inside the binary directory
    :return:
    """
    command = os.path.join(os.path.dirname(__file__) + "/bin/marginPolish")
    sp.call([command] + sys.argv[1:])


if __name__ == '__main__':
    main()
