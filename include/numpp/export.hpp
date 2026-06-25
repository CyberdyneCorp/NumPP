#pragma once

// Public-symbol export macro. Default visibility is hidden (set in CMake), so
// only NUMPP_API-tagged symbols are exported from the shared library.
#if defined(_WIN32)
#  if defined(NUMPP_BUILDING)
#    define NUMPP_API __declspec(dllexport)
#  else
#    define NUMPP_API __declspec(dllimport)
#  endif
#else
#  define NUMPP_API __attribute__((visibility("default")))
#endif
