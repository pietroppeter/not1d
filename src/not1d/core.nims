# Build options for core.nim, read by nim whoever invokes it (nimlang's hatch
# hook runs `nim c`). NOT1D_SORT=pdqsort builds with ot1d's C++ pdqsort
# instead of the Nim radix sort: that needs Nim's C++ backend.
if getEnv("NOT1D_SORT") == "pdqsort":
  switch("backend", "cpp")
  switch("define", "pdqsort")
  # nimlang points Nim's C compiler at `zig cc`; use it for C++ too (zig cc
  # compiles .cpp files as C++), with zig's libc++ linked in statically, so
  # the module is portable and cross-compiles like the C build.
  if get("clang.exe") != "":
    switch("clang.cpp.exe", get("clang.exe"))
    switch("clang.cpp.linkerexe", get("clang.linkerexe"))
    switch("passC", "-lc++")
    switch("passL", "-lc++")
