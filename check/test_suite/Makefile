include ./Makefile.conf

.DEFAULT_GOAL = catch.o

catch.o : catch.cpp catch.hpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

.PHONY : clean very-clean
clean :
	rm -rf coverage.log *.o *.gcov *.dSYM *.gcda *.gcno *.html

very-clean  : clean
