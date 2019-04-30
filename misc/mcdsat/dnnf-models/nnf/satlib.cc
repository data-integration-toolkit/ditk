//#include "cnf.h"
#include "nnf.h"
#include "satlib.h"

#include <set>
#include <vector>
#include <stdio.h>
#include <ctype.h>
#include <fstream>
#include <string>
#include <string.h>

namespace satlib {

bool
read_line( std::istream &is, std::string &line )
{
  char c;
  line.clear();
  while( is.get(c) ) {
    if( c == '\n' ) return(true);
    line += c;
  }
  return(false);
}

#if 0
void
read_cnf_file( std::istream &is, CSolver &manager )
{
  int numvars, numclauses, n = 0;
  std::string line;

  while( read_line(is,line) ) {
    if( line[0] == 'c' )
      continue;
    else if( line[0] == 'p' ) {
      sscanf( line.c_str(), "p cnf %d %d", &numvars, &numclauses );
      manager.set_variable_number(numvars);
    }
    else if( line[0] == '%' )
      break;
    else {
      ++n;
      std::set<int> clause_vars;
      std::set<int> clause_lits;
      char *str = strdup(line.c_str());
      char *ptr = strtok(str," ");
      do {
        int lit = atoi( ptr );
        if( lit != 0 ) {
          cnf::VarIndex var = (lit<0?-lit:lit);
          int sign = lit < 0;
          clause_vars.insert( var );
          clause_lits.insert( (var<<1) + sign );
        }
      } while( (ptr = strtok(0," ")) );
      free(str);

      // add clause
      if( clause_vars.size() > 0 ) {
        std::vector<int> *tmp = new std::vector<int>;
        for( std::set<int>::iterator it = clause_lits.begin(); it != clause_lits.end(); ++it )
          tmp->push_back(*it);
        manager.add_orig_clause( &tmp->begin()[0], tmp->size(), 0 );
      }
      clause_lits.clear();
      clause_vars.clear();
    }
  }
  if( n != numclauses ) throw 1;
}
#endif

void
read_nnf_file( std::istream &is, nnf::Manager &manager )
{
  size_t n = 0;
  int num_vars, num_nodes, num_edges;
  std::string line;
  const nnf::Node **nodes = 0;

  while( read_line(is,line) ) {
    if( line[0] == 'c' )
      continue;
    else if( line[0] == 'n' ) {
      sscanf( line.c_str(), "nnf %d %d %d", &num_nodes, &num_edges, &num_vars );
      manager.reserve(num_nodes);
      manager.set_num_vars(num_vars);
      nodes = new const nnf::Node*[num_nodes];
      for( int i = 0; i < num_nodes; ++i ) nodes[i] = 0;
    }
    else if( line[0] == '%' )
      break;
    else {
      size_t id = n++;
      char *str = strdup(line.c_str());
      char *ptr = strtok(str," ");
      char type = toupper(*ptr);
      ptr = strtok(0, " ");

      if( type == 'L' ) {
        int lit = atoi(ptr);
        assert( lit != 0 );
        unsigned var = (lit > 0 ? (lit<<1) : ((-lit)<<1)+1);
        free(str);
        nodes[id] = nnf::make_variable(&manager,var);
      }
      else if( (type == 'A') || (type == 'O') ) {
        if( type == 'O' ) ptr = strtok(0," "); 
        size_t sz = atoi(ptr);
        if( sz == 0 ) {
          nodes[id] = nnf::make_value(&manager,(type=='A'?true:false));
        }
        else {
          ptr = strtok(0," ");
          std::vector<const nnf::Node*> children;
          for( size_t i = 0; i < sz; ++i ) {
            children.push_back( nodes[atoi(ptr)] );
            ptr = strtok(0," ");
          }
          assert( ptr == 0 );
          free(str);
          if( type == 'A' )
            nodes[id] = nnf::make_and(&manager,children.size(),(const nnf::Node**)&children[0]);
          else
            nodes[id] = nnf::make_or(&manager,children.size(),(const nnf::Node**)&children[0]);
        }
      }
    }
  }
  manager.set_root(nodes[n-1]);
  manager.register_use(manager.root());
  manager.set_sorted();
  for( size_t i = 0; i < n; ++i ) manager.unregister_use(nodes[i]);
  delete[] nodes;
}

}; // satlib namespace

