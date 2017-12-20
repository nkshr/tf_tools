from eval_info import eval_info
from eval_info import eval_info_comp

import argparse
import sys
import numpy as np
import math
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--left',
        type = str,
        default = './left'
    )
    parser.add_argument(
        '--right',
        type = str,
        default = './right'
    )
    parser.add_argument(
        '--result',
        type = str,
        default = './result.csv'
    )
    parser.add_argument(
        '--left_debug_dir',
        type = str,
        default = './debug/left'
    )
    parser.add_argument(
        '--right_debug_dir',
        type = str,
        default = './debug/right'
    )

    flags, unparsed = parser.parse_known_args() 

    einfo_comp = eval_info_comp.eval_info_comp()
    einfo_comp.read(flags.left, flags.right)
    einfo_comp.take_synthesis()
    einfo_comp.write(flags.result)
