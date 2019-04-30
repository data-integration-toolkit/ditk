#include <nnf.h>
#include <satlib.h>

#include <stdio.h>
#include <sys/resource.h>
#include <iostream>
#include <fstream>
#include <functional>
#include <set>

int verbosity_level = 0;
std::vector<int> fvars;

void
banner( std::ostream &os )
{
  //os << "Enumerate models from NNF." << std::endl;
}

void
usage( std::ostream &os )
{
  os << "usage: mincost [-m] [-w] <costfile> <nnf>" << std::endl;
}

inline float
read_time_in_seconds( void )
{
  struct rusage r_usage;
  getrusage( RUSAGE_SELF, &r_usage );
  return( (float)r_usage.ru_utime.tv_sec +
	  (float)r_usage.ru_stime.tv_sec +
	  (float)r_usage.ru_utime.tv_usec / (float)1000000 +
	  (float)r_usage.ru_stime.tv_usec / (float)1000000 );
}

void
extract_variable_group( std::istream &is, std::set<int> &group )
{
  size_t n;
  is >> n;
  for( size_t i = 0; i < n; ++i ) {
    int var;
    is >> var;
    group.insert(var);
  }
}

void
extract_spairs( std::istream &is, std::vector<int> &spairs )
{
  size_t n;
  is >> n;
  for( size_t i = 0; i < n; ++i ) {
    int p, q;
    is >> p >> q;
    spairs.push_back( (p<<16)+q );
  }
}

int
main( int argc, char **argv )
{
  float i_seconds, l_seconds, seconds;
  bool use_mp = false;
  bool output = false;
  banner(std::cout);

  // parse arguments
  ++argv;
  --argc;
  if( argc == 0 ) {
   print_usage:
    usage(std::cout);
    exit(-1);
  }
  for( ; argc && ((*argv)[0] == '-'); --argc, ++argv ) {
    switch( (*argv)[1] ) {
      case 'm':
        use_mp = true;
        break;
      case 'w':
        output = true;
        break;
      case '?':
      default:
        goto print_usage;
    }
  }

  if( argc != 2 ) goto print_usage;
  std::string cost_file = argv[0];
  std::string nnf_file = argv[1];

  // create managers and set start time
  nnf::Manager *nnf_theory = new nnf::Manager(); //0,true);
  i_seconds = l_seconds = read_time_in_seconds();

  // read nnfs from files
  std::ifstream nnf_is(nnf_file.c_str());
  if( !nnf_is.is_open() ) {
    std::cout << "main: error opening file '" << nnf_file << "'" << std::endl;
    exit(-1);
  }

  try {
    std::cout << "main: reading file '" << nnf_file << "'" << std::flush;
    satlib::read_nnf_file(nnf_is,*nnf_theory);
  }
  catch( int e ) {
      std::cout << std::endl << "main: error reading nnf file '" << nnf_file << "'" << std::endl;
      exit(-1);
  }
  seconds = read_time_in_seconds();
  std::cout << " : " << seconds-l_seconds << " seconds" << std::endl;
  l_seconds = seconds;
#if 1
  std::cout << "main: #nodes=" << nnf_theory->count_nodes()
            << ", #edges=" << nnf_theory->count_edges();
  if( use_mp ) {
#ifdef MP
    std::cout << ", #models=" << *(nnf_theory->mp_count_models());
#else
    std::cout << ", #models=<mp support not compiled>";
#endif
  }
  else {
    std::cout << ", #models=" << nnf_theory->count_models();
  }
  seconds = read_time_in_seconds();
  std::cout << " : " << seconds-l_seconds << " seconds" << std::endl;
#endif

  if( output ) {
    std::cout << "--- mincost begin ---" << std::endl;
    nnf_theory->mincost_recursively(std::cout);
    std::cout << "---- mincost end ----" << std::endl;
  }

  // cleanup: don't needed since finishing...
  //nnf_theory->unregister_use( nnf_theory->root() );
  //delete nnf_theory;

  // total time
  //seconds = read_time_in_seconds();
  //std::cout << "main: total time " << seconds-i_seconds << " seconds" << std::endl;
  return(0);
}

