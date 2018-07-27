#! /usr/bin/env python

#####################################################
# concatenate all header files into a single header #
#####################################################

import os
import datetime
from subprocess import check_output as shell_call, CalledProcessError, STDOUT

# get git commit id and write into header file
try:
    git_id = shell_call(['git','rev-parse','HEAD'], cwd=os.path.split(os.path.abspath(__file__))[0], stderr=STDOUT).strip()
except CalledProcessError:
    git_id = 'UNKNOWN'
# in python3, the return type of `shell_call` may be `bytes` but we need `str`
if not isinstance(git_id, str):
    git_id = git_id.decode()

def flatten(input_list):
    new_list = []
    for item in input_list:
        if type(item) == type([]):
            new_list.extend(flatten(item))
        else:
            new_list.append(item)
    return new_list

def parse_header(header_filename, header_path, known_headers=set()):
    with open(os.path.join(header_path,header_filename), 'r') as f:
        header = f.readlines()

    for idx, line in enumerate(header):
        if not line.startswith('#include "'):
            continue

        assert line.endswith('"\n')
        include_path, include_filename = os.path.split(line[len('#include "'):-len('"\n')])
        if os.path.abspath(os.path.join(header_path,include_path,include_filename)) not in known_headers:
            print(os.path.abspath(os.path.join(header_path,include_path,include_filename)))
            known_headers.add(os.path.abspath(os.path.join(header_path,include_path,include_filename)))
            include = parse_header(include_filename, os.path.join(header_path, include_path))
            header[idx] = include
        else:
            header[idx] = '// (Included Above): ' + line

    return header

main_header = parse_header('qmc.hpp','src')
# Add top banner
main_header.insert(0,
"""/*
 * Qmc Single Header
 * Commit: """ + git_id + """
 * Generated: """ +  datetime.datetime.now().strftime('%d-%m-%Y %X') + """
 *
 * ----------------------------------------------------------
 * This file has been merged from multiple headers.
 * Please don't edit it directly
 * ----------------------------------------------------------
 */
""")

# output single header file
with open('qmc.hpp','w') as out:
    out.writelines(flatten(main_header))
