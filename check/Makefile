# Setup
# =====
.DEFAULT_GOAL=runtests

# Running Tests
# =============

# Running all
# -----------
ALL_EXAMPLES=$(shell ls -d */test.log | sed s/\\/test.log//)
.PHONY : runtests
runtests : $(ALL_EXAMPLES)

# General rule
# ------------
.PHONY .SILENT : $(ALL_EXAMPLES)
$(ALL_EXAMPLES) : test_suite/catch.o
	echo ------------------------------------------------------------
	echo "START `date +'%s'`"
	echo ------------------------------------------------------------
	echo "TEST $@"
	echo ------------------------------------------------------------
	printf "Running 'make clean' in $@ ...\n"
	$(MAKE) -C "$@" clean \
	|| printf "make clean failed\n@@@ FAILURE @@@\n" \
		> "$@/test/test.log" \
	printf "Running 'make test' in $@...\n"
	$(MAKE) -C "$@" test \
	|| printf "make test failed\n@@@ FAILURE @@@\n" \
		> "$@/test/test.log"
	echo ------------------------------------------------------------
	echo "DONE `date +'%s'`"
	echo ------------------------------------------------------------

test_suite/catch.o :
	$(MAKE) -C test_suite catch.o

# Printing Results
# ================
.SILENT .PHONY : summarize
summarize :
	printf "\n"
	printf "\033[0;32m"
	find . -type f -name 'test.log' -exec grep -Hl "@@@ SUCCESS @@@" \{} \
		   2>/dev/null \; | \
		sed 's/\.\/\([A-Za-z0-9_-]*\)\/test.log/+ \1 (succeeded)/' | \
		sort
	printf "\033[0m"
	printf "\033[1;31m"
	find . -type f -name 'test.log' -exec grep -Hl "@@@ FAILURE @@@" \{} \
		   2>/dev/null \; | \
		sed 's/\.\/\([A-Za-z0-9_-]*\)\/test.log/+ \1 (FAILED)/' | \
		sort
	find . -type f -name 'test.log' -exec grep -HL "@@@ FAILURE\|SUCCESS @@@" \{} \; | \
		sed 's/\.\/\([A-Za-z0-9_-]*\)\/test.log/+ \1 (UNKNOWN)/' | \
		sort
	printf "\033[0m"
	printf "\n"

	# return nonzero exit status if any example does not report success
	for dir in $(ALL_EXAMPLES); do grep -HL "@@@ SUCCESS @@@" $$dir/test.log 2>&1 >/dev/null || exit 1 ; done

# Cleaning up
# ===========
.PHONY : clean
clean :
	for f in * ; \
	do \
	if test -d "$$f" -a -f "$$f/Makefile" ; \
	then \
		$(MAKE) -C $$f clean ; \
	fi \
	done