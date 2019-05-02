#ifndef NNF_INCLUDE
#define NNF_INCLUDE

#include "utils.h"
#include "hashing.h"
#include <iostream>
#include <iomanip>
#include <list>
#include <set>
#include <vector>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

#ifdef MP
#include <gmp.h>
class MP_num {
  mpz_t z_;
public:
  MP_num( int n = 0 ) { mpz_init_set_si(z_,n); }
  MP_num( const MP_num &a ) { mpz_set(z_,a.z_); }
  ~MP_num() { mpz_clear(z_); }
  void operator=( const MP_num &a ) { mpz_set(z_,a.z_); }
  void operator=( int n ) { mpz_set_si(z_,n); }
  bool operator==( const MP_num &a ) { return( mpz_cmp(z_,a.z_) == 0 ); }
  bool operator==( int n ) { return( mpz_cmp_si(z_,n) == 0 ); }
  void operator+=( const MP_num &a ) { mpz_add(z_,z_,a.z_); }
  void operator*=( const MP_num &a ) { mpz_mul(z_,z_,a.z_); }
  void print( std::ostream &os ) const { mpz_out_str(stdout,10,z_); }
};
inline std::ostream& operator<<( std::ostream &os, const MP_num &a ) { a.print(os); return(os); }
#endif

namespace nnf {

  template<typename T> struct Pool {
    T *table_;
    size_t size_;
    size_t used_;
    size_t boundary_;
    std::list<std::pair<size_t,T*> > free_list_;
    size_t size_free_list_;
    Pool( size_t size ) : size_(size), used_(0), boundary_(0), size_free_list_(0) { table_ = new T[size]; }
    ~Pool() { delete[] table_; }
    bool verify() const { return( (boundary_ <= size_) && (used_ + size_free_list_ == boundary_) ); }
    bool from_here( const T *e ) const { return( (table_ <= e) && (e < &table_[size_]) ); }
    T* allocate( size_t n = 1 ) { if( n+boundary_ <= size_ ) { T *e = &table_[boundary_]; boundary_ += n; used_ += n; return(e); } return(0); }
    void clear() { used_ = 0; boundary_ = 0; free_list_.clear(); size_free_list_ = 0; }
    void return_to_free_list( T *e, size_t size = 1 ) { free_list_.push_front(std::make_pair(size,e)); used_ -= size; size_free_list_ += size; }
    T* get_from_free_list( size_t size = 1 )
    { 
      for( typename std::list<std::pair<size_t,T*> >::iterator fi = free_list_.begin(), end = free_list_.end(); fi != end; ++fi ) {
        if( (*fi).first == size ) {
          T *e = (*fi).second;
          used_ += size;
          size_free_list_ -= size;
          free_list_.erase(fi);
          return(e);
        }
      }
      return(0);
    }
    void stat( std::ostream &os ) const { os << "size=" << size_ << ", used=" << used_ << ", boundary=" << boundary_ << ", free-list=" << size_free_list_ << std::endl; }
  };

  class Model : public std::set<int> {
  public:
    void print( std::ostream &os, bool all = false ) const { for( const_iterator mi = begin(); mi != end(); ++mi ) { if( all ) os << ((*mi)%2?-((*mi)>>1):(*mi)>>1) << " "; else if( (*mi)%2 == 0 ) os << ((*mi)>>1) << " "; } }
  };
  class ModelList : public std::list<Model*> {
  public:
    void print( std::ostream &os, bool all = false ) const { for( const_iterator mi = begin(); mi != end(); ++mi ) { (*mi)->print(os); os << std::endl; } }
  };

  struct NodeCache {
    const void *first_;
    const void *second_;
    NodeCache() : first_(0), second_(0) { }
    void set( const void *f = 0, const void *s = 0 ) { first_ = f; second_ = s; }
  };

  enum NodeType { Null, And, Or, Value, Variable };

  struct Node;
  class Manager;
  typedef Node* NodePtr;

  struct Node {
    NodeType type_;
    NodePtr *children_;
    mutable size_t nparents_;
    mutable size_t szparents_;
    mutable NodePtr *parents_;
    mutable NodeCache cache_;
    mutable size_t ref_;
    Node( Manager *man = 0, size_t sizep = 0 );
    ~Node() { }
    void push_parent( Manager &man, Node *n ) const;
    void clean_cache() const { cache_.first_ = cache_.second_ = 0; }
    void clean() { type_ = Null; children_ = 0; nparents_ = 0; szparents_ = 0; parents_ = 0; ref_ = 0; cache_.set(); }
    bool is_clean() const { return( (type_ == Null) && !children_ && !parents_ && !cache_.first_ && !cache_.second_ && !ref_ ); }
    void print( std::ostream &os ) const
    {
      os << "node: id=" << this << ", type=" << type_ << std::flush;
      if( type_ == Value )
        os << ", value=" << (int)children_;
      else if( type_ == Variable )
        os << ", literal=" << (int)children_;
      else if( (type_ == And) || (type_ == Or) ) {
        os << ", children={";
        if( children_ ) for( const NodePtr *p = children_; *p != 0; ++p ) os << *p << ",";
        os << "}, parents={";
        if( parents_ ) for( const NodePtr *p = parents_; *p != 0; ++p ) os << *p << ",";
        os << "}";
      }
      os << std::endl;
    }
  };

  struct NodePool : public Pool<Node> {
    NodePool( size_t size ) : Pool<Node>(size) { }
  };
  struct NodePtrPool : public Pool<NodePtr> {
    NodePtrPool( size_t size ) : Pool<NodePtr>(size) { }
  };

  class LiteralCache : public __gnu_cxx::hash_map<int,const Node*> {
    mutable size_t lookups_, hits_;
    typedef __gnu_cxx::hash_map<int,const Node*> basetype;
  public:
    LiteralCache() : lookups_(0), hits_(0) { }
    ~LiteralCache() { }
    size_t lookups() const { return(lookups_); }
    size_t hits() const { return(hits_); }
    float hit_rate() const { return((float)hits_/(float)lookups_); }
    const Node* lookup( int lit ) const { ++lookups_; const_iterator hi = find(lit); if( hi == end() ) return(0); ++hits_; return((*hi).second); }
    void insert( const Node *n ) { assert( n->type_ == Variable ); basetype::insert( std::make_pair((int)n->children_,n) ); }
    void remove( const Node *n ) { iterator hi = find((int)n->children_); erase(hi); }
    void clear() { basetype::clear(); lookups_ = 0; hits_ = 0; }
  };
  class MultipleCache {
    mutable size_t lookups_, hits_;
  public:
    MultipleCache() : lookups_(0), hits_(0) { }
    ~MultipleCache() { }
    size_t lookups() const { return(lookups_); }
    size_t hits() const { return(hits_); }
    float hit_rate() const { return((float)hits_/(float)lookups_); }
    const Node* lookup( size_t size, const Node **children ) const { ++lookups_; return 0; }
    void insert( const Node *n ) { }
    void remove( const Node *n ) { }
    void clear() { lookups_ = 0; hits_ = 0; }
  };
 
  class Manager {
    size_t num_vars_;
    mutable unsigned ref_count_;
    mutable float inc_rate_;

    bool sorted_;
    bool verbose_;

    LiteralCache var_cache_;
    MultipleCache or_cache_;
    MultipleCache and_cache_;

    const Node *root_;
    mutable const Node *false_node_;
    mutable const Node *true_node_;

  protected:
    std::vector<NodePool*> node_pools_;
    std::vector<NodePtrPool*> node_ptr_pools_;

    friend const Node* make_and( Manager *man, size_t size, const Node **conjuncts );
    friend const Node* make_or( Manager *man, size_t size, const Node **disjuncts );
    friend const Node* make_value( Manager *man, unsigned var );
    friend const Node* make_value( Manager *man, bool value );

  public:
    Manager( size_t num_vars = 0, bool verbose = false, size_t space = 1024 )
      : num_vars_(num_vars), ref_count_(0), inc_rate_(2), sorted_(false), verbose_(verbose), root_(0)
    {
      if( verbose_ ) std::cout << "nnf: new manager: id=" << this << std::endl;
      if( space == 0 ) space = 1024;
      node_pools_.push_back( new NodePool(space) );
      if( verbose_ ) std::cout << "nnf: creating node pool: id=" << node_pools_.back() << ", size=" << space << std::endl;
      node_ptr_pools_.push_back( new NodePtrPool(space) );
      if( verbose_ ) std::cout << "nnf: creating node_ptr pool: id=" << node_ptr_pools_.back() << ", size=" << space << std::endl;
      false_node_ = 0;
      true_node_ = 0;
    }
    virtual ~Manager()
    {
      if( false_node_ ) unregister_use( false_node_ );
      if( true_node_ ) unregister_use( true_node_ );
      if( verbose_ ) std::cout << "nnf: delete manager: id=" << this << ", ref_count=" << ref_count_ << std::endl;
    }

    void set_verbose() { verbose_ = true; }
    size_t num_vars() const { return(num_vars_); }
    void set_num_vars( size_t num_vars ) { num_vars_ = num_vars; }
    size_t num() const { size_t n = 0; for( size_t i = 0; i < node_pools_.size(); ++i ) n += node_pools_[i]->size_; return(n); }
    size_t size() const { size_t n = 0; for( size_t i = 0; i < node_pools_.size(); ++i ) n += node_pools_[i]->used_; return(n); }
    bool sorted() const { return(sorted_); }
    void set_sorted( bool flag = true ) { sorted_ = flag; }
    const Node* root() const { return(root_); }
    void set_root( const Node *root ) { root_ = root; }
    float set_inc_rate( float irate ) { float r = inc_rate_; inc_rate_ = irate; return(r); }

#if 0
    bool verify_integrity() const
    {
      for( size_t i = 0; i < num_nodes_; ++i ) {
        const Node *n = nodes_[i];
        if( n->ref_ < n->parents_->size() ) return(false);
      }
      return(true);
    }
#endif

    void clear()
    {
      for( size_t i = 0, isz = node_pools_.size(); i < isz; ++i ) node_pools_[i]->clear();
      for( size_t i = 0, isz = node_ptr_pools_.size(); i < isz; ++i ) node_ptr_pools_[i]->clear();
      num_vars_ = 0;
      ref_count_ = 0;
      sorted_ = false;
      var_cache_.clear();
      or_cache_.clear();
      and_cache_.clear();
      root_ = 0;
      false_node_ = 0;
      true_node_ = 0;
    }
    void register_use( const Node *n ) { ++n->ref_; ++ref_count_; }
    void unregister_use( const Node *n )
    {
      assert( (n->ref_ > 0) && (ref_count_ > 0) );
      --ref_count_;
      if( --n->ref_ == 0 ) {
        if( (n->type_ == And) || (n->type_ == Or) )
          for( const NodePtr *p = n->children_; *p != 0; ++p ) unregister_use(*p);
        remove_from_cache(n);
        return_to_free_list((Node*)n);
        assert( n->is_clean() );
      }
    }

    void node_ptr_pool_stat( std::ostream &os ) const
    {
      for( size_t i = 0, sz = node_ptr_pools_.size(); i < sz; ++i ) {
        os << "  " << std::setw(2) << i << ": ";
        node_ptr_pools_[i]->stat(os);
      }
    }
    NodePtr* allocate_ptr( size_t size )
    {
      assert( !node_ptr_pools_.empty() );
      NodePtr *p = 0;
      for( size_t i = 0, sz = node_ptr_pools_.size(); i < sz; ++i ) {
        p = node_ptr_pools_[i]->allocate(size);
        if( p != 0 ) return(p);
      }
      size_t sz = 2*node_ptr_pools_.back()->size_;
      while( sz < size ) sz *= 2;
      node_ptr_pools_.push_back( new NodePtrPool(sz) );
      if( verbose_ ) {
        std::cout << "nnf: creating node_ptr pool: id=" << node_ptr_pools_.back() << ", size=" << sz << std::endl
                  << "node_ptr pools stat:" << std::endl;
        node_ptr_pool_stat(std::cout);
      }
      p = node_ptr_pools_.back()->allocate(size);
      assert( p != 0 );
      return(p);
    }
    void return_ptr_to_free_list( NodePtr *p, size_t size )
    {
      for( size_t i = 0, sz = node_ptr_pools_.size(); i < sz; ++i )
        if( node_ptr_pools_[i]->from_here(p) ) { node_ptr_pools_[i]->return_to_free_list(p,size); break; }
    }
    NodePtr* get_ptr( size_t size = 1 )
    {
      NodePtr *p = 0;
      for( size_t i = 0, sz = node_ptr_pools_.size(); (i < sz) && (p == 0); ++i ) {
        assert( node_ptr_pools_[i]->verify() );
        p = node_ptr_pools_[i]->get_from_free_list(size);
      }
      if( p == 0 ) p = allocate_ptr(size);
      for( size_t i = 0; i < size; ++i ) p[i] = 0;
      return(p);
    }

    void node_pool_stat( std::ostream &os ) const
    {
      for( size_t i = 0, sz = node_pools_.size(); i < sz; ++i ) {
        os << "  " << std::setw(2) << i << ": ";
        node_pools_[i]->stat(os);
      }
    }
    Node* allocate( size_t size = 1 )
    {
      assert( !node_pools_.empty() );
      Node *n = 0;
      for( size_t i = 0, sz = node_pools_.size(); i < sz; ++i ) {
        n = node_pools_[i]->allocate(size);
        if( n != 0 ) return(n);
      }
      size_t sz = utils::max((size_t)(inc_rate_*node_pools_.back()->size_),size);
      node_pools_.push_back( new NodePool(sz) );
      if( verbose_ ) {
        std::cout << "nnf: creating node pool: id=" << node_pools_.back() << ", size=" << sz << std::endl
                  << "node pools stat:" << std::endl;
        node_pool_stat(std::cout);
      }
      n = node_pools_.back()->allocate();
      assert( n != 0 );
      return(n);
    }
    void reserve( size_t size = 0 )
    {
      size_t count = 0;
      for( size_t i = 0, isz = node_pools_.size(); i < isz; ++i ) {
        NodePool *pool = node_pools_[i];
        count += pool->size_ - pool->boundary_ + pool->size_free_list_;
      }
      if( count < size ) {
        node_pools_.push_back( new NodePool(size-count) );
        if( verbose_ ) {
          std::cout << "nnf: creating node pool: id=" << node_pools_.back() << ", size=" << size-count << std::endl
                    << "node pools stat:" << std::endl;
          node_pool_stat(std::cout);
        }
      }
    }
    void return_to_free_list( Node *n )
    {
      for( size_t i = 0, sz = node_pools_.size(); i < sz; ++i )
        if( node_pools_[i]->from_here(n) ) { node_pools_[i]->return_to_free_list(n); break; }
      size_t size = 0;
      if( ((n->type_ == And) || (n->type_ == Or)) && n->children_ ) {
        for( NodePtr *p = n->children_; *p != 0; ++p, ++size ) *p = 0;
        return_ptr_to_free_list( n->children_, 1+size );
      }
      if( n->parents_ ) {
        for( NodePtr *p = n->parents_; *p != 0; ++p, ++size ) *p = 0;
        return_ptr_to_free_list( n->parents_, n->szparents_ );
      }
      n->clean();
    }
    Node* get_node()
    {
      Node *n = 0;
      for( size_t i = 0, sz = node_pools_.size(); (i < sz) && (n == 0); ++i ) {
        assert( node_pools_[i]->verify() );
        n = node_pools_[i]->get_from_free_list();
      }
      if( n == 0 ) n = allocate();
      n->clean();
      register_use(n);
      return(n);
    }
    const Node* get_node( NodeType type, unsigned value )
    {
      Node *n = get_node();
      //if( verbose_ ) std::cout << "nnf: new variable/value node: id=" << n << std::endl;
      n->type_ = type;
      n->children_ = (NodePtr*)value;
      return(n);
    }
    const Node* get_node( NodeType type, size_t size, Node **children )
    {
      assert( (type == And) || (type == Or) );
      Node *n = get_node();
      //if( verbose_ ) std::cout << "nnf: new and/or node: id=" << n << std::endl;
      n->type_ = type;
      n->children_ = get_ptr(1+size);
      for( size_t i = 0; i < size; ++i ) {
        n->children_[i] = children[i];
        children[i]->push_parent(*this,n);
        register_use(children[i]);
      }
      n->children_[size] = 0;
      return(n);
    }

    const Node* cache_lookup( NodeType type, unsigned variable ) const
    {
      assert( type == Variable );
      return( var_cache_.lookup(variable) );
    }
    const Node* cache_lookup( NodeType type, size_t size, const Node **children ) const
    {
      assert( (type == And) || (type == Or) );
      return( type == And ? and_cache_.lookup(size,children) : or_cache_.lookup(size,children) );
    }
    void insert_in_cache( const Node *n )
    {
      assert( (n->type_ == And) || (n->type_ == Or) || (n->type_ == Variable) );
      if( n->type_ == Variable )
        var_cache_.insert(n);
      else if( n->type_ == And )
        and_cache_.insert(n);
      else
        or_cache_.insert(n);
    }
    void remove_from_cache( const Node *n )
    {
      if( n->type_ == Variable )
        var_cache_.remove(n);
      else if( n->type_ == And )
        and_cache_.remove(n);
      else
        or_cache_.remove(n);
    }
    void clean_node_cache() const
    {
      for( size_t i = 0, isz = node_pools_.size(); i < isz; ++i ) {
        NodePool *pool = node_pools_[i];
        for( size_t j = 0, jsz = pool->boundary_; j < jsz; ++j ) {
          pool->table_[j].clean_cache();
        }
      }
    }

    const Node* true_node() { if( !true_node_ ) true_node_ = get_node(Value,true); return(true_node_); }
    const Node* false_node() { if( !false_node_ ) false_node_ = get_node(Value,false); return(false_node_); }
    bool is_constant_false( const Node* n ) const { return( (n->type_ == Value) && ((int)n->children_ == 0) ); }
    bool is_constant_true( const Node* n ) const { return( (n->type_ == Value) && ((int)n->children_ == 1) ); }

    size_t count_nodes() const
    {
      size_t n = 0;
      for( size_t i = 0, sz = node_pools_.size(); i < sz; ++i ) n += node_pools_[i]->used_;
      return(n);
    }
    size_t count_edges() const;
    void sort();

    //void smooth();

    bool satisfiable() const { return( !is_constant_false(root_) ); }
    float count_models( float *output = 0, const int *litmap = 0 ) const;
#ifdef MP
    MP_num* mp_count_models( MP_num *output = 0, const int *litmap = 0 ) const;
#endif

    int mincost_recursively( const Node *n, Model &m) const;
    std::pair<bool,const Node*> enumerate_models_recursively( const Node *n, Model &m, const Node *last_or ) const;
    void enumerate_models( std::ostream &os, bool all = false ) const;
    void enumerate_models( ModelList &models ) const;
    //void enumerate_min_models( ModelList &models ) const;
    //float compute_value( float *output = 0, const float *values = 0, const int *litmap = 0, node nref = null_node ) const;

    void sorted_dump( std::ostream &os ) const;
    void recursive_dump( std::ostream &os, const Node *n, size_t &index ) const;
    void dump( std::ostream &os ) const
    {
      os << "nnf " << count_nodes() << " " << count_edges() << " " << num_vars() << std::endl;
      if( sorted_ )
        sorted_dump(os);
      else {
        size_t index = 0;
        recursive_dump(os,root_,index);
      }
      clean_node_cache();
    }
    void stats( std::ostream &os ) const
    {
      os << "nnf: #nodes=" << count_nodes() << ", #edges=" << count_edges() << ", ref_count=" << ref_count_ << std::endl;
      os << "node pools:" << std::endl;
      node_pool_stat(os);
      os << "node_ptr pools:" << std::endl;
      node_ptr_pool_stat(os);
    }
  };

  inline Node::Node( Manager *man, size_t sizep )
    : type_(Null), children_(0), nparents_(0), szparents_(sizep), parents_(0), ref_(0)
  {
    if( man && (szparents_ > 0) ) parents_ = man->get_ptr(szparents_);
  }
  inline void Node::push_parent( Manager &man, Node *n ) const
  {
    assert( ((nparents_ == 0) && (szparents_ == 0)) || (nparents_ < szparents_) );
    if( szparents_ == 0 ) {
      szparents_ = 2;
      parents_ = man.get_ptr(szparents_);
    }
    else if( 1+nparents_ == szparents_ ) {
      NodePtr *p = parents_;
      ++szparents_;
      parents_ = man.get_ptr(szparents_);
      for( size_t i = 0; i < nparents_; ++i ) {
        parents_[i] = p[i];
        p[i] = 0;
      }
      man.return_ptr_to_free_list(p,szparents_-1);
    }
    parents_[nparents_++] = n;
  }

  inline const Node* make_and( Manager *man, size_t size, const Node **conjuncts )
  {
    size_t sz = 0;
    std::vector<const Node*> children;
    for( size_t i = 0; i < size; ++i ) {
      if( man->is_constant_false(conjuncts[i]) ) {
        man->register_use(man->false_node());
        return(man->false_node());
      }
      else if( !man->is_constant_true(conjuncts[i]) ) {
        children.push_back(conjuncts[i]);
        ++sz;
      }
    }
    if( children.empty() ) {
      man->register_use(man->true_node());
      return(man->true_node());
    }
    else {
      const Node *n = man->cache_lookup( And, sz, (const Node**)&children[0] );
      if( n != 0 )
        man->register_use(n);
      else {
        n = man->get_node( And, sz, (Node**)&children[0] );
        man->insert_in_cache(n);
      }
      return(n);
    }
  }
  inline const Node* make_or( Manager *man, size_t size, const Node **disjuncts )
  {
    size_t sz = 0;
    std::vector<const Node*> children;
    for( size_t i = 0; i < size; ++i ) {
      if( man->is_constant_true(disjuncts[i]) ) {
        man->register_use(man->true_node());
        return(man->true_node());
      }
      else if( !man->is_constant_false(disjuncts[i]) ) {
        children.push_back(disjuncts[i]);
        ++sz;
      }
    }
    if( children.empty() ) {
      man->register_use(man->false_node());
      return(man->false_node());
    }
    else {
      const Node *n = man->cache_lookup( Or, sz, (const Node**)&children[0] );
      if( n != 0 )
        man->register_use(n);
      else {
        n = man->get_node( Or, sz, (Node**)&children[0] );
        man->insert_in_cache(n);
      }
      return(n);
    }
  }
  inline const Node* make_value( Manager *man, bool value )
  {
    if( value ) {
      man->register_use(man->true_node());
      return(man->true_node());
    }
    else {
      man->register_use(man->false_node());
      return(man->false_node());
    }
  }
  inline const Node* make_variable( Manager *man, unsigned var )
  {
    const Node *n = man->cache_lookup( Variable, var );
    if( n != 0 )
      man->register_use(n);
    else {
      n = man->get_node( Variable, var );
      man->insert_in_cache(n);
    }
    return(n);
  }

  inline void dump( const Manager *man, std::ostream &os ) { man->dump(os); }

}; // nnf namespace

inline std::ostream& operator<<( std::ostream &os, const nnf::Model &m ) { m.print(os); return(os); }
inline std::ostream& operator<<( std::ostream &os, const nnf::ModelList &ml ) { ml.print(os); return(os); }

#endif // NNF_INCLUDE

