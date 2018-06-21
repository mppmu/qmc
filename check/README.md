# How to run the tests

Each test can be run by executing 'make <dirname>'. 
'make runtests' can be invoked to run all tests.
'make summarize' gives a one-line summary for each example:

make runtests:
   This command runs 'make test' for all subdirectories.

make summarize:
   By running "make summarize" the user obtains a printout of the status
   of all integrals for which 'make test' has been run earlier.
   Note that this command does not run any test.

make clean:
   Delete all generated files; i.e. run 'make clean'
   in every test directory.

make <dirname>:
   Run 'make test' in the directory <dirname>.


# Further options

Make's "-j<jmake>" option controls how many recipes make executes in parallel. 

In order to change the c++ compiler, you can set the environment variables CXX. 
For example, to run an with the clang++ compiler type
'CXX=clang++ make <example>'.

# Adding new tests

All subdirectories are considered when invoking 'make runtests'. 
A test directory should have a "Makefile" that defines the
targets 'test' and 'clean':

make test:
    Should compile and run a test program. A log should be written into the file
    "test.log". If all tests have succeeded, the file "test.log" should end with
    the line "@@@ SUCCESS @@@". Otherwise, "test.log" should end with the line
    "@@@ FAILURE @@@".

make clean:
    Should delete all temporary and log files that are generated on 'make test'.
