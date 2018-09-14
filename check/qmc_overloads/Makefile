include ./Makefile.conf
include ../test_suite/Makefile.conf

.PHONY : compile
compile : $(TEST_NAME)

.PHONY : test
test : test.log
test.log : $(TEST_NAME)
	./$< 2>&1 >test.log \
	&& printf "\n@@@ SUCCESS @@@" >> test.log || printf "\n@@@ FAILURE @@@" >> test.log

ifneq ($(CXX), nvcc)

.PHONY : coverage
coverage : coverage.html

coverage.log : $(TEST_NAME) test.log
	$(GCOV) $(GCOVFLAGS) $< 2>&1 >coverage.log

coverage.html : %.html : %.log
	$(GCOVR) $(GCOVRFLAGS) -o $@

endif

.PHONY : clean
clean :
	rm -rf coverage.log test.log *.o *.gcov *.dSYM *.gcda *.gcno *.html $(TEST_NAME)
