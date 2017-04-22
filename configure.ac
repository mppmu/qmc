AC_INIT([Qmc], [0.1])
AC_CONFIG_SRCDIR([src/qmc.hpp])

AM_INIT_AUTOMAKE

LT_INIT

AC_CONFIG_MACRO_DIRS([acinclude.d])

AC_PROG_CXX
AX_CXX_COMPILE_STDCXX_11([noext]) # noext: use -std=c++11 rather than -std=gnu++11

AC_OUTPUT(Makefile src/Makefile check/Makefile)
