#ifndef UTILS_INCLUDE
#define UTILS_INCLUDE

namespace utils {
  template<typename T> inline const T& min( const T& a, const T& b ) { return( a <= b ? a : b ); }
  template<typename T> inline const T& max( const T& a, const T& b ) { return( a >= b ? a : b ); }
};

#endif // UTILS_INCLUDE

