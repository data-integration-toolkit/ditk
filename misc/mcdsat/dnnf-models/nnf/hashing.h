#ifndef HASHING_H
#define HASHING_H

#include <ext/hash_map>
#include <ext/hash_set>

namespace hashing = ::__gnu_cxx;

namespace __gnu_cxx
{
  template<typename T> class hash<const T*>
  {
  public:
    size_t operator()( const T* p ) const { return( size_t(p) ); }
  };
};

#endif // HASHING_H
