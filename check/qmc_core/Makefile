include ./Makefile.conf
include ../test_suite/Makefile.conf

.PHONY : compile
compile : $(TEST_NAME)

.PHONY : test
test : test.log
test.log : $(TEST_NAME)
	./$< 2>&1 >test.log \
	&& printf "\n@@@ SUCCESS @@@" >> test.log || printf "\n@@@ FAILURE @@@" >> test.log

.PHONY : clean
clean :
	rm -rf *.o $(TEST_NAME)
	echo > test.log
