#! /usr/bin/env python

#####################################################
# concatenate all header files into a single header #
#####################################################

import os
from subprocess import check_output as shell_call, CalledProcessError, STDOUT

with open(os.path.join('src','qmc.hpp'),'r') as f:
    main_header = f.readlines()

# get git commit id and write into header file
try:
    git_id = shell_call(['git','rev-parse','HEAD'], cwd=os.path.split(os.path.abspath(__file__))[0], stderr=STDOUT).strip()
except CalledProcessError:
    git_id = 'UNKNOWN'
# in python3, the return type of `shell_call` may be `bytes` but we need `str`
if not isinstance(git_id, str):
    git_id = git_id.decode()
main_header.insert(0,'// qmc single header, generated from commit "' + git_id + '"\n\n')

# insert qmc headers into main header
for idx,line in enumerate(main_header):
    if not line.startswith('#include "qmc_'):
        continue

    assert line.endswith('"\n')
    include_filename = line[len('#include "'):-len('"\n')]

    with open(os.path.join('src',include_filename),'r') as include_header:
        main_header[idx] = '/*\n * begin included header "' + include_filename + '"\n */\n' \
                         + include_header.read() + '\n' \
                         + '/*\n * end included header "' + include_filename + '"\n */\n\n'

# output single header file
with open('qmc.hpp','w') as out:
    out.writelines(main_header)
