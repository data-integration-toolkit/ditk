#include "nnf.h"
#include "satlib.h"
#include <fstream>
#include <stdio.h>
#include <unistd.h>
#include <math.h>

#define INF 0x7f
#define MIN(a,b) ((a)<(b)?(a):(b))

namespace nnf {

size_t
Manager::count_edges() const
{
  for( size_t i = 0, isz = node_pools_.size(); i < isz; ++i ) {
    NodePool *pool = node_pools_[i];
    for( size_t j = 0, jsz = pool->boundary_; j < jsz; ++j ) {
      const Node *n = &pool->table_[j];
      assert( n->cache_.first_ == 0 );
      if( (n->type_ == And) || (n->type_ == Or) ) {
        size_t count = 0;
        for( const NodePtr *p = n->children_; *p != 0; ++p ) {
          count += 1 + (size_t)(*p)->cache_.first_;
          (*p)->cache_.set();
        }
        n->cache_.set((const void*)count);
      }
    }
  }
  size_t count = (size_t)root_->cache_.first_;
  clean_node_cache();
  return(count);
}

void
Manager::sort()
{
  if( sorted_ ) return;
  char tname[20] = "/tmp/nnf.XXXXXX";
  char *fname = mktemp(tname);

  if( verbose_ ) std::cout << "nnf: sorting: dumping" << std::flush;
  std::ofstream os(fname); 
  dump(os);
  os.close();

  clear();

  if( verbose_ ) std::cout << ", reading" << std::flush;
  std::ifstream is(fname);
  satlib::read_nnf_file(is,*this);
  is.close();

  if( verbose_ ) std::cout << std::endl;
  unlink(fname);
}

// count nnf models:
//   if output = 0, just count all models, else count models for each literal and store in output
//   if litmap != 0, count models compatible with literals in litmap

float
Manager::count_models( float *output, const int *litmap ) const
{
  // first pass (bottom-up): compute model count in each node
  for( size_t i = 0, isz = node_pools_.size(); i < isz; ++i ) {
    NodePool *pool = node_pools_[i];
    for( size_t j = 0, jsz = pool->boundary_; j < jsz; ++j ) {
      const Node *n = &pool->table_[j];
      if( n->ref_ > 0 ) {
        float count = -1;
        if( (n->type_ == And) || (n->type_ == Or) ) {
          for( const NodePtr *p = n->children_; *p != 0; ++p ) {
            float c = *(float*)&(*p)->cache_.first_;
            count = (count==-1?c:(n->type_==And?count*c:count+c));
          }
        }
        else if( n->type_ == Value )
          count = ((int)n->children_?1:0);
        else if( n->type_ == Variable ) {
          if( litmap && litmap[(int)n->children_^1] )
            count = 0;
          else
            count = 1;
        }
        n->cache_.set( *(float**)&count );
      }
    }
  }
  float result = *(float*)&root_->cache_.first_;
  if( !output ) {
    clean_node_cache();
    return( result );
  }

#if 0
  //std::cout << "MC=" << *(float*)&n->cache_.first_ << ":" << result << std::endl;
  //dump( std::cout );
  //for( size_t i = 0; i < num_; ++i )
  //std::cout << *(float*)&nodes_[i].cache_.second_ << std::endl;
  //std::cout << std::endl;

  // initialize output
  for( size_t i = 0; i <= num_vars_; ++ i ) output[i] = -1;

  // second pass (top-down): differentiate with respect to each node (only for complete count)
  for( size_t i = num_; i > 0; --i )
    {
      float pd = 0;
      const Node *curr = &nodes_[i-1];
      float count = *(float*)&curr->cache_.first_;
      if( (curr->parents_->size() == 0) || (count < 0) )
	pd = 1;
      else
	{
	  for( NodeVector::const_iterator pi = curr->parents_->begin(); pi != curr->parents_->end(); ++pi )
	    {
	      assert( (nodes_[*pi].type_ == And) || (nodes_[*pi].type_ == Or) );
	      float cpd = *(float*)&nodes_[*pi].cache_.second_;
	      if( nodes_[*pi].type_ == And )
		{
		  if( nodes_[*pi].left_ == i-1 )
		    cpd *= *(float*)&nodes_[nodes_[*pi].right_].cache_.first_;
		  else
		    cpd *= *(float*)&nodes_[nodes_[*pi].left_].cache_.first_;
		}
	      pd += cpd;
	    }
	}
      curr->cache_.second_ = *(const float**)&pd;
      if( curr->type_ == Variable )
	{
	  if( (curr->left_ % 2 == 0) && (!litmap || !litmap[curr->left_^1]) )
	    output[curr->left_>>1] = pd;
	  else if( output[curr->left_>>1] == -1 )
	    output[curr->left_>>1] = 0;
	}
    }
  output[0] = *(float*)&n->cache_.first_;
#endif
  return( result );
}

#ifdef MP
MP_num*
Manager::mp_count_models( MP_num *output, const int *litmap ) const
{
  // first pass (bottom-up): compute model count in each node
  for( size_t i = 0, isz = node_pools_.size(); i < isz; ++i ) {
    NodePool *pool = node_pools_[i];
    for( size_t j = 0, jsz = pool->boundary_; j < jsz; ++j ) {
      const Node *n = &pool->table_[j];
      if( n->ref_ > 0 ) {
        MP_num *count = new MP_num(-1);
        if( (n->type_ == And) || (n->type_ == Or) ) {
          for( const NodePtr *p = n->children_; *p != 0; ++p ) {
            MP_num *c = (MP_num*)(*p)->cache_.first_;
            if( *count == -1 )
              *count = *c;
            else if( n->type_ == And )
              *count *= (*c);
            else
              *count += (*c);
          }
        }
        else if( n->type_ == Value )
          *count = ((int)n->children_?1:0);
        else if( n->type_ == Variable ) {
          if( litmap && litmap[(int)n->children_^1] )
            *count = 0;
          else
            *count = 1;
        }
        n->cache_.set( (const void*)count );
      }
    }
  }
  MP_num *result = new MP_num(*(MP_num*)root_->cache_.first_);
  if( !output ) {
    for( size_t i = 0, isz = node_pools_.size(); i < isz; ++i ) {
      NodePool *pool = node_pools_[i];
        for( size_t j = 0, jsz = pool->boundary_; j < jsz; ++j ) {
          const Node *n = &pool->table_[j];
          if( n->ref_ > 0 ) {
            delete (MP_num*)n->cache_.first_;
            n->cache_.set();
          }
        }
    }
    return( result );
  }

#if 0
  //std::cout << "MC=" << *(float*)&n->cache_.first_ << ":" << result << std::endl;
  //dump( std::cout );
  //for( size_t i = 0; i < num_; ++i )
  //std::cout << *(float*)&nodes_[i].cache_.second_ << std::endl;
  //std::cout << std::endl;

  // initialize output
  for( size_t i = 0; i <= num_vars_; ++ i ) output[i] = -1;

  // second pass (top-down): differentiate with respect to each node (only for complete count)
  for( size_t i = num_; i > 0; --i )
    {
      float pd = 0;
      const Node *curr = &nodes_[i-1];
      float count = *(float*)&curr->cache_.first_;
      if( (curr->parents_->size() == 0) || (count < 0) )
	pd = 1;
      else
	{
	  for( NodeVector::const_iterator pi = curr->parents_->begin(); pi != curr->parents_->end(); ++pi )
	    {
	      assert( (nodes_[*pi].type_ == And) || (nodes_[*pi].type_ == Or) );
	      float cpd = *(float*)&nodes_[*pi].cache_.second_;
	      if( nodes_[*pi].type_ == And )
		{
		  if( nodes_[*pi].left_ == i-1 )
		    cpd *= *(float*)&nodes_[nodes_[*pi].right_].cache_.first_;
		  else
		    cpd *= *(float*)&nodes_[nodes_[*pi].left_].cache_.first_;
		}
	      pd += cpd;
	    }
	}
      curr->cache_.second_ = *(const float**)&pd;
      if( curr->type_ == Variable )
	{
	  if( (curr->left_ % 2 == 0) && (!litmap || !litmap[curr->left_^1]) )
	    output[curr->left_>>1] = pd;
	  else if( output[curr->left_>>1] == -1 )
	    output[curr->left_>>1] = 0;
	}
    }
  output[0] = *(float*)&n->cache_.first_;
#endif
  return( result );
}
#endif // MP

#if 0
float
Manager::compute_value_sorted_smooth( const Node *n, float *output, const float *values, const int *litmap ) const
{
  float result;
  assert( sorted_ && smooth_ );

  // first pass (bottom-up): compute model count in each node
  for( size_t i = 0; i < num_; ++i ) {
    float count = 0, pcount = 0;
    const Node *curr = &nodes_[i];
    if( (curr->type_ == And) || (curr->type_ == Or) ) {
      assert( (i > curr->left_) && (i > curr->right_) );
      float left = *(float*)&nodes_[curr->left_].cache_.first_;
      float right = *(float*)&nodes_[curr->right_].cache_.first_;
      float pleft = *(float*)&nodes_[curr->left_].cache_.second_;
      float pright = *(float*)&nodes_[curr->right_].cache_.second_;
      count = (curr->type_==And?left*right:left+right);
      if( (pleft < 0) && (pright < 0) )
        pcount = -1;
      else {
        pcount = (curr->type_==And?1:0);
        if( pleft >= 0 ) pcount = pleft;
        if( pright >= 0 ) pcount = (curr->type_==And?pcount*pright:pcount+pright);
        if( (curr->type_ == Or) && (pcount == 0) && ((pleft < 0) || (pright < 0)) )
          pcount = -1;
      }
    }
    else if( curr->type_ == Value )
      count = pcount = (curr->left_?1:0);
    else if( curr->type_ == Variable ) {
      count = pcount = (!values?1:values[curr->left_]);
      if( litmap && litmap[curr->left_^1] )
        count = pcount = 0;
    }
    curr->cache_.set( true, *(const float**)&count, *(const float**)&pcount );
    //std::cout << "insert for " << i << " " << count << " " << pcount << std::endl;
  }
  result = *(float*)&n->cache_.second_;
  if( !output ) return( result );
  //std::cout << "MC=" << *(float*)&n->cache_.first_ << ":" << result << std::endl;
  //dump( std::cout );
  //for( size_t i = 0; i < num_; ++i )
  //std::cout << "cache[" << i << "]=" << *(float*)&nodes_[i].cache_.second_ << std::endl;
  //std::cout << std::endl;

  // initialize output
  for( size_t i = 0; i <= num_vars_; ++ i ) output[i] = -1;

  // second pass (top-down): differentiate with respect to each node (only for complete count)
  for( size_t i = num_; i > 0; --i ) {
    float pd = 0;
    const Node *curr = &nodes_[i-1];
    float count = *(float*)&curr->cache_.first_;
    if( (curr->parents_->size() == 0) || (count < 0) )
      pd = 1;
    else {
      for( NodeVector::const_iterator pi = curr->parents_->begin(); pi != curr->parents_->end(); ++pi ) {
        assert( (nodes_[*pi].type_ == And) || (nodes_[*pi].type_ == Or) );
        float cpd = *(float*)&nodes_[*pi].cache_.second_;
        if( nodes_[*pi].type_ == And ) {
          if( nodes_[*pi].left_ == i-1 )
            cpd *= *(float*)&nodes_[nodes_[*pi].right_].cache_.first_;
          else
            cpd *= *(float*)&nodes_[nodes_[*pi].left_].cache_.first_;
        }
        pd += cpd;
      }
    }
    curr->cache_.second_ = *(const float**)&pd;
    if( curr->type_ == Variable ) {
      if( !litmap || !litmap[curr->left_^1] )
        output[curr->left_>>1] = pd;
      else if( output[curr->left_>>1] == -1 )
        output[curr->left_>>1] = 0;
    }
  }
  output[0] = *(float*)&n->cache_.first_;
  return( result );
}
#endif

int
Manager::mincost_recursively( const Node *n, Model &m) const
{
  bool stat = true;
  if( n->type_ == And ) {
    int mm=INF;

    for( NodePtr *p = n->children_; (*p != 0); ++p ) {
      mm=MIN(mm,mincost_recursively(*p,m));
    }

    return mm;
  }
  else if( n->type_ == Or ) {
    int sum=0;

    for( NodePtr *p = n->children_; (*p != 0); ++p ) {
      sum+=mincost_recursively(*p,m);
    }

    return sum;
  }
  else if( n->type_ == Variable ) {
      //if n vista, devolver costo, else 0
      return 0;
  }
  else if( n->type_ == Value ) {
    return 0;
  }
}

std::pair<bool,const Node*>
Manager::enumerate_models_recursively( const Node *n, Model &m, const Node *last_or ) const
{
  bool stat = true;
  const Node *lor = last_or;
  if( n->type_ == And ) {
    for( NodePtr *p = n->children_; stat && (*p != 0); ++p ) {
      std::pair<bool,const Node*> rc = enumerate_models_recursively(*p,m,lor);
      stat = rc.first;
      lor = rc.second;
    }
  }
  else if( n->type_ == Or ) {
    int next = (int)n->cache_.first_;
    n->cache_.second_ = lor;
    std::pair<bool,const Node*> rc = enumerate_models_recursively(n->children_[next],m,n);
    stat = rc.first;
    lor = rc.second;
  }
  else if( n->type_ == Variable ) {
    m.insert((int)n->children_);
  }
  else if( n->type_ == Value ) {
    stat = ((int)n->children_ != 0);
  }
  return(std::make_pair(stat,lor));
}

void
Manager::enumerate_models( std::ostream &os, bool all ) const
{
  Model m;
  bool next = true;
  while( next ) {
    // generate model
    std::pair<bool,const Node*> rc = enumerate_models_recursively(root_,m,0);
    if( rc.first ) {
      m.print(os,all);
      os << std::endl;
    }
    m.clear();

    // advance state
    next = false;
    for( const Node *n = rc.second; !next && (n != 0); n = (const Node*)n->cache_.second_ ) {
      int i = 1+(int)n->cache_.first_;
      if( n->children_[i] == 0 )
        n->cache_.first_ = 0;
      else {
        n->cache_.first_ = (const void*)i;
        next = true;
      }
    }
  }
}

void
Manager::enumerate_models( ModelList &models ) const
{
  Model m;
  bool next = true;
  while( next ) {
    // generate model
    std::pair<bool,const Node*> rc = enumerate_models_recursively(root_,m,0);
    if( rc.first ) models.push_back( new Model(m) );
    m.clear();

    // advance state
    next = false;
    for( const Node *n = rc.second; !next && (n != 0); n = (const Node*)n->cache_.second_ ) {
      int i = 1+(int)n->cache_.first_;
      if( n->children_[i] == 0 )
        n->cache_.first_ = 0;
      else {
        n->cache_.first_ = (const void*)i;
        next = true;
      }
    }
  }
}

void
Manager::sorted_dump( std::ostream &os ) const
{
  size_t index = 0;
  for( size_t i = 0, isz = node_pools_.size(); i < isz; ++i ) {
    NodePool *pool = node_pools_[i];
    for( size_t j = 0, jsz = pool->boundary_; j < jsz; ++j ) {
      const Node *n = &pool->table_[j];
      if( n->ref_ > 0 ) {
        n->cache_.set( (const void*)index++ );
        if( n->type_ == Value )
          os << ((int)n->children_==0?"O 0 0":"A 0") << std::endl;
        else if( n->type_ == Variable )
          os << "L " << ((int)n->children_%2?-(((int)n->children_)>>1):((int)n->children_)>>1) << std::endl;
        else if( (n->type_ == And) || (n->type_ == Or) ) {
          size_t size = 0;
          for( const NodePtr *p = n->children_; *p != 0; ++p, ++size );
          os << (n->type_==And?"A ":"O 0 ") << size;
          for( const NodePtr *p = n->children_; *p != 0; ++p )
            os << " " << (size_t)(*p)->cache_.first_;
          os << std::endl;
        }
      }
    }
  }
}

void
Manager::recursive_dump( std::ostream &os, const Node *n, size_t &index ) const
{
  if( (int)n->cache_.second_ == 1 ) return;
  if( n->type_ == Value )
    os << ((int)n->children_==0?"O 0 0":"A 0") << std::endl;
  else if( n->type_ == Variable )
    os << "L " << ((int)n->children_%2?-(((int)n->children_)>>1):((int)n->children_)>>1) << std::endl;
  else if( (n->type_ == And) || (n->type_ == Or) ) {
    size_t size = 0;
    for( const NodePtr *p = n->children_; *p != 0; ++p, ++size )
      recursive_dump( os, *p, index );
    os << (n->type_==And?"A ":"O 0 ") << size;
    for( const NodePtr *p = n->children_; *p != 0; ++p )
      os << " " << (size_t)(*p)->cache_.first_;
    os << std::endl;
  }
  n->cache_.set( (const void*)index++, (const void*)1 );
}

}; // nnf namespace

