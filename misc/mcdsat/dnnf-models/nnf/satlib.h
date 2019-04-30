#ifndef SATLIB_INCLUDE
#define SATLIB_INCLUDE

#include <iostream>

class CSolver;
namespace nnf { class Manager; };

namespace satlib
{
  void read_cnf_file( std::istream &is, CSolver &manager );
  void read_nnf_file( std::istream &is, nnf::Manager &manager );
};

#endif // SATLIB_INCLUDE
