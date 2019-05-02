import pandas as pd
import hashlib
import struct
import random, copy, struct
import warnings
import numpy as np
import pickle
from collections import defaultdict
import os
import random
import string
from abc import ABCMeta, abstractmethod
from scipy.stats import pearsonr, spearmanr
import text_similarity
ABC = ABCMeta('ABC', (object,), {}) # compatible with Python 2 *and* 3
try:
    import redis
except ImportError:
    redis = None


def ordered_storage(config, name=None):
    '''Return ordered storage system based on the specified config.
    The canonical example of such a storage container is
    ``defaultdict(list)``. Thus, the return value of this method contains
    keys and values. The values are ordered lists with the last added
    item at the end.
    Args:
        config (dict): Defines the configurations for the storage.
            For in-memory storage, the config ``{'type': 'dict'}`` will
            suffice. For Redis storage, the type should be ``'redis'`` and
            the configurations for the Redis database should be supplied
            under the key ``'redis'``. These parameters should be in a form
            suitable for `redis.Redis`. The parameters may alternatively
            contain references to environment variables, in which case
            literal configuration values should be replaced by dicts of
            the form::
                {'env': 'REDIS_HOSTNAME',
                 'default': 'localhost'}
            For a full example, see :ref:`minhash_lsh_at_scale`
        name (bytes, optional): A reference name for this storage container.
            For dict-type containers, this is ignored. For Redis containers,
            this name is used to prefix keys pertaining to this storage
            container within the database.
    '''
    tp = config['type']
    if tp == 'dict':
        return DictListStorage(config)
    if tp == 'redis':
        return RedisListStorage(config, name=name)


def unordered_storage(config, name=None):
    '''Return an unordered storage system based on the specified config.
    The canonical example of such a storage container is
    ``defaultdict(set)``. Thus, the return value of this method contains
    keys and values. The values are unordered sets.
    Args:
        config (dict): Defines the configurations for the storage.
            For in-memory storage, the config ``{'type': 'dict'}`` will
            suffice. For Redis storage, the type should be ``'redis'`` and
            the configurations for the Redis database should be supplied
            under the key ``'redis'``. These parameters should be in a form
            suitable for `redis.Redis`. The parameters may alternatively
            contain references to environment variables, in which case
            literal configuration values should be replaced by dicts of
            the form::
                {'env': 'REDIS_HOSTNAME',
                 'default': 'localhost'}
            For a full example, see :ref:`minhash_lsh_at_scale`
        name (bytes, optional): A reference name for this storage container.
            For dict-type containers, this is ignored. For Redis containers,
            this name is used to prefix keys pertaining to this storage
            container within the database.
    '''
    tp = config['type']
    if tp == 'dict':
        return DictSetStorage(config)
    if tp == 'redis':
        return RedisSetStorage(config, name=name)


class Storage(ABC):
    '''Class for key, value containers where the values are sequences.'''
    def __getitem__(self, key):
        return self.get(key)

    def __delitem__(self, key):
        return self.remove(key)

    def __len__(self):
        return self.size()

    def __iter__(self):
        for key in self.keys():
            yield key

    def __contains__(self, item):
        return self.has_key(item)

    @abstractmethod
    def keys(self):
        '''Return an iterator on keys in storage'''
        return []

    @abstractmethod
    def get(self, key):
        '''Get list of values associated with a key
        
        Returns empty list ([]) if `key` is not found
        '''
        pass

    def getmany(self, *keys):
        return [self.get(key) for key in keys]

    @abstractmethod
    def insert(self, key, *vals, **kwargs):
        '''Add `val` to storage against `key`'''
        pass

    @abstractmethod
    def remove(self, *keys):
        '''Remove `keys` from storage'''
        pass

    @abstractmethod
    def remove_val(self, key, val):
        '''Remove `val` from list of values under `key`'''
        pass

    @abstractmethod
    def size(self):
        '''Return size of storage with respect to number of keys'''
        pass

    @abstractmethod
    def itemcounts(self, **kwargs):
        '''Returns the number of items stored under each key'''
        pass

    @abstractmethod
    def has_key(self, key):
        '''Determines whether the key is in the storage or not'''
        pass

    def status(self):
        return {'keyspace_size': len(self)}

    def empty_buffer(self):
        pass


class OrderedStorage(Storage):

    pass


class UnorderedStorage(Storage):

    pass


class DictListStorage(OrderedStorage):
    '''This is a wrapper class around ``defaultdict(list)`` enabling
    it to support an API consistent with `Storage`
    '''
    def __init__(self, config):
        self._dict = defaultdict(list)

    def keys(self):
        return self._dict.keys()

    def get(self, key):
        return self._dict.get(key, [])

    def remove(self, *keys):
        for key in keys:
            del self._dict[key]

    def remove_val(self, key, val):
        self._dict[key].remove(val)

    def insert(self, key, *vals, **kwargs):
        self._dict[key].extend(vals)

    def size(self):
        return len(self._dict)

    def itemcounts(self, **kwargs):
        '''Returns a dict where the keys are the keys of the container.
        The values are the *lengths* of the value sequences stored
        in this container.
        '''
        return {k: len(v) for k, v in self._dict.items()}

    def has_key(self, key):
        return key in self._dict


class DictSetStorage(UnorderedStorage, DictListStorage):
    '''This is a wrapper class around ``defaultdict(set)`` enabling
    it to support an API consistent with `Storage`
    '''
    def __init__(self, config):
        self._dict = defaultdict(set)

    def get(self, key):
        return self._dict.get(key, set())

    def insert(self, key, *vals, **kwargs):
        self._dict[key].update(vals)


if redis is not None:
    class RedisBuffer(redis.client.Pipeline):
        ''' 
        A bufferized version of `redis.pipeline.Pipeline`.
        The only difference from the conventional pipeline object is the
        ``_buffer_size``. Once the buffer is longer than the buffer size,
        the pipeline is automatically executed, and the buffer cleared.
        
        '''

        def __init__(self, connection_pool, response_callbacks, transaction, buffer_size,
                     shard_hint=None):
            self._buffer_size = buffer_size
            super(RedisBuffer, self).__init__(
                connection_pool, response_callbacks, transaction,
                shard_hint=shard_hint)

        @property
        def buffer_size(self):
            return self._buffer_size

        @buffer_size.setter
        def buffer_size(self, value):
            self._buffer_size = value

        def execute_command(self, *args, **kwargs):
            if len(self.command_stack) >= self._buffer_size:
                self.execute()
            super(RedisBuffer, self).execute_command(*args, **kwargs)


    class RedisStorage:
        '''Base class for Redis-based storage containers.
        Args:
            config (dict): Redis storage units require a configuration
                of the form::
                    storage_config={
                        'type': 'redis',
                        'redis': {'host': 'localhost', 'port': 6379}
                    }
                one can refer to system environment variables via::
                    storage_config={
                        'type': 'redis',
                        'redis': {
                            'host': {'env': 'REDIS_HOSTNAME',
                                     'default':'localhost'},
                            'port': 6379}
                        }
                    }
            name (bytes, optional): A prefix to namespace all keys in
                the database pertaining to this storage container.
                If None, a random name will be chosen.
        '''

        def __init__(self, config, name=None):
            self.config = config
            self._buffer_size = 50000
            redis_param = self._parse_config(self.config['redis'])
            self._redis = redis.Redis(**redis_param)
            self._buffer = RedisBuffer(self._redis.connection_pool,
                                       self._redis.response_callbacks,
                                       transaction=True,
                                       buffer_size=self._buffer_size)
            if name is None:
                name = _random_name(11)
            self._name = name

        @property
        def buffer_size(self):
            return self._buffer_size

        @buffer_size.setter
        def buffer_size(self, value):
            self._buffer_size = value
            self._buffer.buffer_size = value

        def redis_key(self, key):
            return self._name + key

        def _parse_config(self, config):
            cfg = {}
            for key, value in config.items():
                # If the value is a plain str, we will use the value
                # If the value is a dict, we will extract the name of an environment
                # variable stored under 'env' and optionally a default, stored under
                # 'default'.
                # (This is useful if the database relocates to a different host
                # during the lifetime of the LSH object)
                if isinstance(value, dict):
                    if 'env' in value:
                        value = os.getenv(value['env'], value.get('default', None))
                cfg[key] = value
            return cfg

        def __getstate__(self):
            state = self.__dict__.copy()
            # We cannot pickle the connection objects, they get recreated
            # upon unpickling
            state.pop('_redis')
            state.pop('_buffer')
            return state

        def __setstate__(self, state):
            self.__dict__ = state
            # Reconnect here
            self.__init__(self.config, name=self._name)


    class RedisListStorage(OrderedStorage, RedisStorage):
        def __init__(self, config, name=None):
            RedisStorage.__init__(self, config, name=name)

        def keys(self):
            return self._redis.hkeys(self._name)

        def redis_keys(self):
            return self._redis.hvals(self._name)

        def status(self):
            status = self._parse_config(self.config['redis'])
            status.update(Storage.status(self))
            return status

        def get(self, key):
            return self._get_items(self._redis, self.redis_key(key))

        def getmany(self, *keys):
            pipe = self._redis.pipeline()
            pipe.multi()
            for key in keys:
                self._get_items(pipe, self.redis_key(key))
            return pipe.execute()

        @staticmethod
        def _get_items(r, k):
            return r.lrange(k, 0, -1)

        def remove(self, *keys):
            self._redis.hdel(self._name, *keys)
            self._redis.delete(*[self.redis_key(key) for key in keys])

        def remove_val(self, key, val):
            redis_key = self.redis_key(key)
            self._redis.lrem(redis_key, val)
            if not self._redis.exists(redis_key):
                self._redis.hdel(self._name, redis_key)

        def insert(self, key, *vals, **kwargs):
            # Using buffer=True outside of an `insertion_session`
            # could lead to inconsistencies, because those
            # insertion will not be processed until the
            # buffer is cleared
            buffer = kwargs.pop('buffer', False)
            if buffer:
                self._insert(self._buffer, key, *vals)
            else:
                self._insert(self._redis, key, *vals)

        def _insert(self, r, key, *values):
            redis_key = self.redis_key(key)
            r.hset(self._name, key, redis_key)
            r.rpush(redis_key, *values)

        def size(self):
            return self._redis.hlen(self._name)

        def itemcounts(self):
            pipe = self._redis.pipeline()
            pipe.multi()
            ks = self.keys()
            for k in ks:
                self._get_len(pipe, self.redis_key(k))
            d = dict(zip(ks, pipe.execute()))
            return d

        @staticmethod
        def _get_len(r, k):
            return r.llen(k)

        def has_key(self, key):
            return self._redis.hexists(self._name, key)

        def empty_buffer(self):
            self._buffer.execute()
            # To avoid broken pipes, recreate the connection
            # objects upon emptying the buffer
            self.__init__(self.config, name=self._name)


    class RedisSetStorage(UnorderedStorage, RedisListStorage):
        def __init__(self, config, name=None):
            RedisListStorage.__init__(self, config, name=name)

        @staticmethod
        def _get_items(r, k):
            return r.smembers(k)

        def remove_val(self, key, val):
            redis_key = self.redis_key(key)
            self._redis.srem(redis_key, val)
            if not self._redis.exists(redis_key):
                self._redis.hdel(self._name, redis_key)

        def _insert(self, r, key, *values):
            redis_key = self.redis_key(key)
            r.hset(self._name, key, redis_key)
            r.sadd(redis_key, *values)

    def query(self, minhash):
        '''
        Giving the MinHash of the query set, retrieve 
        the keys that references sets with Jaccard
        similarities greater than the threshold.
        
        Args:
            minhash (datasketch.MinHash): The MinHash of the query set. 
        Returns:
            `list` of unique keys.
        '''
        if len(minhash) != self.h:
            raise ValueError("Expecting minhash with length %d, got %d"
                    % (self.h, len(minhash)))
        candidates = set()
        for (start, end), hashtable in zip(self.hashranges, self.hashtables):
            H = self._H(minhash.hashvalues[start:end])
            for key in hashtable.get(H):
                candidates.add(key)
        if self.prepickle:
            return [pickle.loads(key) for key in candidates]
        else:
            return list(candidates)

    def __contains__(self, key):
        '''
        Args:
            key (hashable): The unique identifier of a set.
        Returns: 
            bool: True only if the key exists in the index.
        '''
        if self.prepickle:
            key = pickle.dumps(key)
        return key in self.keys

    def remove(self, key):
        '''
        Remove the key from the index.
        Args:
            key (hashable): The unique identifier of a set.
        '''
        if self.prepickle:
            key = pickle.dumps(key)
        if key not in self.keys:
            raise ValueError("The given key does not exist")
        for H, hashtable in zip(self.keys[key], self.hashtables):
            hashtable.remove_val(H, key)
            if not hashtable.get(H):
                hashtable.remove(H)
        self.keys.remove(key)

    def is_empty(self):
        '''
        Returns:
            bool: Check if the index is empty.
        '''
        return any(t.size() == 0 for t in self.hashtables)

    @staticmethod
    def _H(hs):
        return bytes(hs.byteswap().data)

    def _query_b(self, minhash, b):
        if len(minhash) != self.h:
            raise ValueError("Expecting minhash with length %d, got %d"
                    % (self.h, len(minhash)))
        if b > len(self.hashtables):
            raise ValueError("b must be less or equal to the number of hash tables")
        candidates = set()
        for (start, end), hashtable in zip(self.hashranges[:b], self.hashtables[:b]):
            H = self._H(minhash.hashvalues[start:end])
            if H in hashtable:
                for key in hashtable[H]:
                    candidates.add(key)
        if self.prepickle:
            return {pickle.loads(key) for key in candidates}
        else:
            return candidates

    def get_counts(self):
        '''
        
        Returns a list of length ``self.b`` with elements representing the
        number of keys stored under each bucket for the given permutation.
        
        '''
        counts = [
            hashtable.itemcounts() for hashtable in self.hashtables]
        return counts

    def get_subset_counts(self, *keys):
        '''
        Returns the bucket allocation counts (see :func:`~datasketch.MinHashLSH.get_counts` above)
        restricted to the list of keys given.
        Args:
            keys (hashable) : the keys for which to get the bucket allocation
                counts
        '''
        if self.prepickle:
            key_set = [pickle.dumps(key) for key in set(keys)]
        else:
            key_set = list(set(keys))
        hashtables = [unordered_storage({'type': 'dict'}) for _ in
                      range(self.b)]
        Hss = self.keys.getmany(*key_set)
        for key, Hs in zip(key_set, Hss):
            for H, hashtable in zip(Hs, hashtables):
                hashtable.insert(H, key)
        return [hashtable.itemcounts() for hashtable in hashtables]


class MinHashLSHInsertionSession:
    '''
        Context manager for batch insertion of documents into a MinHashLSH.
    '''

    def __init__(self, lsh, buffer_size):
        self.lsh = lsh
        self.lsh.buffer_size = buffer_size

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def close(self):
        self.lsh.keys.empty_buffer()
        for hashtable in self.lsh.hashtables:
            hashtable.empty_buffer()

    def insert(self, key, minhash, check_duplication=True):
        '''
        Insert a unique key to the index, together
        with a MinHash (or weighted MinHash) of the set referenced by
        the key.
        Args:
            key (hashable): The unique identifier of the set.
            minhash (datasketch.MinHash): The MinHash of the set.
        '''
        self.lsh._insert(key, minhash, check_duplication=check_duplication,
                         buffer=True)









def sha1_hash32(data):
    """

    A 32-bit hash function based on SHA1.
    Args:
        data (bytes): the data to generate 32-bit integer hash from.
    Returns:
        int: an integer hash value that can be encoded using 32 bits.
    
    """
    return struct.unpack('<I', hashlib.sha1(data).digest()[:4])[0]

def sha1_hash64(data):
    """A 32-bit hash function based on SHA1.
    Args:
        data (bytes): the data to generate 64-bit integer hash from.
    Returns:
        int: an integer hash value that can be encoded using 64 bits.
    """
    return struct.unpack('<Q', hashlib.sha1(data).digest()[:8])[0]


# Load csv



# The size of a hash value in number of bytes
hashvalue_byte_size = len(bytes(np.int64(42).data))

# http://en.wikipedia.org/wiki/Mersenne_prime
_mersenne_prime = (1 << 61) - 1
_max_hash = (1 << 32) - 1
_hash_range = (1 << 32)

class MinHash(text_similarity.TextSemanticSimilarity):
    '''MinHash is a probabilistic data structure for computing
    `Jaccard similarity`_ between sets.
    Args:
        num_perm (int, optional): Number of random permutation functions.
            It will be ignored if `hashvalues` is not None.
        seed (int, optional): The random seed controls the set of random
            permutation functions generated for this MinHash.
        hashfunc (optional): The hash function used by this MinHash.
            It takes the input passed to the `update` method and
            returns an integer that can be encoded with 32 bits.
            The default hash function is based on SHA1 from hashlib_.
        hashobj (**deprecated**): This argument is deprecated since version
            1.4.0. It is a no-op and has been replaced by `hashfunc`.
        hashvalues (`numpy.array` or `list`, optional): The hash values is
            the internal state of the MinHash. It can be specified for faster
            initialization using the existing state from another MinHash.
        permutations (optional): The permutation function parameters. This argument
            can be specified for faster initialization using the existing
            state from another MinHash.
    Note:
        To save memory usage, consider using :class:`datasketch.LeanMinHash`.
    Note:
        Since version 1.1.1, MinHash will only support serialization using
        `pickle`_. ``serialize`` and ``deserialize`` methods are removed,
        and are supported in :class:`datasketch.LeanMinHash` instead.
        MinHash serialized before version 1.1.1 cannot be deserialized properly
        in newer versions (`need to migrate? <https://github.com/ekzhu/datasketch/issues/18>`_).
    Note:
        Since version 1.1.3, MinHash uses Numpy's random number generator
        instead of Python's built-in random package. This change makes the
        hash values consistent across different Python versions.
        The side-effect is that now MinHash created before version 1.1.3 won't
        work (i.e., ``jaccard``, ``merge`` and ``union``)
        with those created after.
    .. _`Jaccard similarity`: https://en.wikipedia.org/wiki/Jaccard_index
    .. _hashlib: https://docs.python.org/3.5/library/hashlib.html
    .. _`pickle`: https://docs.python.org/3/library/pickle.html
    '''

    def __init__(self, num_perm=128, seed=1,
            hashfunc=sha1_hash32,
            hashobj=None, # Deprecated.
            hashvalues=None, permutations=None):
        if hashvalues is not None:
            num_perm = len(hashvalues)
        if num_perm > _hash_range:
            # Because 1) we don't want the size to be too large, and
            # 2) we are using 4 bytes to store the size value
            raise ValueError("Cannot have more than %d number of\
                    permutation functions" % _hash_range)
        self.seed = seed
        # Check the hash function.
        if not callable(hashfunc):
            raise ValueError("The hashfunc must be a callable.")
        self.hashfunc = hashfunc
        # Check for use of hashobj and issue warning.
        if hashobj is not None:
            warnings.warn("hashobj is deprecated, use hashfunc instead.",
                    DeprecationWarning)
        # Initialize hash values
        if hashvalues is not None:
            self.hashvalues = self._parse_hashvalues(hashvalues)
        else:
            self.hashvalues = self._init_hashvalues(num_perm)
        # Initalize permutation function parameters
        if permutations is not None:
            self.permutations = permutations
        else:
            generator = np.random.RandomState(self.seed)
            # Create parameters for a random bijective permutation function
            # that maps a 32-bit hash value to another 32-bit hash value.
            # http://en.wikipedia.org/wiki/Universal_hashing
            self.permutations = np.array([(generator.randint(1, _mersenne_prime, dtype=np.uint64),
                                           generator.randint(0, _mersenne_prime, dtype=np.uint64))
                                          for _ in range(num_perm)], dtype=np.uint64).T
        if len(self) != len(self.permutations[0]):
            raise ValueError("Numbers of hash values and permutations mismatch")

    def _init_hashvalues(self, num_perm):
        return np.ones(num_perm, dtype=np.uint64)*_max_hash

    def _parse_hashvalues(self, hashvalues):
        return np.array(hashvalues, dtype=np.uint64)

    def update(self, b):
        '''Update this MinHash with a new value.
        The value will be hashed using the hash function specified by
        the `hashfunc` argument in the constructor.
        Args:
            b: The value to be hashed using the hash function specified.
        Example:
            To update with a new string value (using the default SHA1 hash
            function, which requires bytes as input):
            .. code-block:: python
                minhash = Minhash()
                minhash.update("new value".encode('utf-8'))
            We can also use a different hash function, for example, `pyfarmhash`:
            .. code-block:: python
                import farmhash
                def _hash_32(b):
                    return farmhash.hash32(b)
                minhash = MinHash(hashfunc=_hash_32)
                minhash.update("new value")
        '''
        hv = self.hashfunc(b)
        a, b = self.permutations
        phv = np.bitwise_and((a * hv + b) % _mersenne_prime, np.uint64(_max_hash))
        self.hashvalues = np.minimum(phv, self.hashvalues)

    def jaccard(self, other):
        '''Estimate the `Jaccard similarity`_ (resemblance) between the sets
        represented by this MinHash and the other.
        Args:
            other (datasketch.MinHash): The other MinHash.
        Returns:
            float: The Jaccard similarity, which is between 0.0 and 1.0.
        '''
        if other.seed != self.seed:
            raise ValueError("Cannot compute Jaccard given MinHash with\
                    different seeds")
        if len(self) != len(other):
            raise ValueError("Cannot compute Jaccard given MinHash with\
                    different numbers of permutation functions")
        return np.float(np.count_nonzero(self.hashvalues==other.hashvalues)) /\
                np.float(len(self))

    def count(self):
        '''Estimate the cardinality count based on the technique described in
        `this paper <http://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=365694>`_.
        Returns:
            int: The estimated cardinality of the set represented by this MinHash.
        '''
        k = len(self)
        return np.float(k) / np.sum(self.hashvalues / np.float(_max_hash)) - 1.0

    def merge(self, other):
        '''Merge the other MinHash with this one, making this one the union
        of both.
        Args:
            other (datasketch.MinHash): The other MinHash.
        '''
        if other.seed != self.seed:
            raise ValueError("Cannot merge MinHash with\
                    different seeds")
        if len(self) != len(other):
            raise ValueError("Cannot merge MinHash with\
                    different numbers of permutation functions")
        self.hashvalues = np.minimum(other.hashvalues, self.hashvalues)

    def digest(self):
        '''Export the hash values, which is the internal state of the
        MinHash.
        Returns:
            numpy.array: The hash values which is a Numpy array.
        '''
        return copy.copy(self.hashvalues)

    def is_empty(self):
        '''
        Returns:
            bool: If the current MinHash is empty - at the state of just
                initialized.
        '''
        if np.any(self.hashvalues != _max_hash):
            return False
        return True

    def clear(self):
        '''
        Clear the current state of the MinHash.
        All hash values are reset.
        '''
        self.hashvalues = self._init_hashvalues(len(self))

    def copy(self):
        '''
        :returns: datasketch.MinHash -- A copy of this MinHash by exporting its state.
        '''
        return MinHash(seed=self.seed, hashfunc=self.hashfunc,
                hashvalues=self.digest(),
                permutations=self.permutations)

    def __len__(self):
        '''
        :returns: int -- The number of hash values.
        '''
        return len(self.hashvalues)

    def __eq__(self, other):
        '''
        :returns: bool -- If their seeds and hash values are both equal then two are equivalent.
        '''
        return type(self) is type(other) and \
            self.seed == other.seed and \
            np.array_equal(self.hashvalues, other.hashvalues)

    def read_dataset(self, data_file):
        '''
            Read dataset given in data_file
            Input Parameters : 
                - data_file                 :  Dataset to be read
            Returns : 
                - input_df( dataframe )     :  Dataframe that stores the input dataset, used for manipulation  

        '''
        input_df = pd.read_excel(data_file)
        return input_df
        
        

    @classmethod
    def union(cls, *mhs):
        '''
            Create a MinHash which is the union of the MinHash objects passed as arguments.
            Input Parameters:
                *mhs:The MinHash objects to be united. The argument list length is variable,
                but must be at least 2.
            Returns:
                datasketch.MinHash: A new union MinHash.
        '''

        if len(mhs) < 2:
            raise ValueError("Cannot union less than 2 MinHash")
        num_perm = len(mhs[0])
        seed = mhs[0].seed
        if any((seed != m.seed or num_perm != len(m)) for m in mhs):
            raise ValueError("The unioning MinHash must have the\
                    same seed and number of permutation functions")
        hashvalues = np.minimum.reduce([m.hashvalues for m in mhs])
        permutations = mhs[0].permutations
        return cls(num_perm=num_perm, seed=seed, hashvalues=hashvalues,
                permutations=permutations)

    def generate_embeddings(self, input_list):

        '''
            Generates embeddings for each of the word in the input_list
            Input Parameters :
                - Input_list :       List of words
                - Returns    :       List of embeddings

        '''
        embeddings_list=[]
        for each_word in input_list:
            m = MinHash(num_perm=128)
            m.update(each_word.encode('utf8'))
            embeddings_list.append(m.hashvalues)
        return embeddings_list

    def load_model(self):

        '''
            No Load Model - No training required

        '''
        pass
    def save_model(self):

        '''
            No save model - No training required

        '''
        pass

    def train(self):

        '''
            No training needed

        '''
        pass


    def evaluate_util(self,file_name=""):
        '''

            Utility for evaluation of the file given
            Input Parameters : file name for evaluation
            Returns          : {Pearson Correlation Coefficient,
                               Spearman Correlation Coefficient}


        '''

        input_list=[]
        predict_similarity_list=[]
        actual_similarity_list=[]

        '''
            Evaluation on SemEval 2017 Dataset

        '''

        if file_name=='semEval':

            df=pd.read_csv("Datasets//sts-test.csv",error_bad_lines=False,sep='\t',usecols=range(7),engine='python')
            for index, row in df.iterrows():
                predicted_similarity_score=self.predict(row[5],row[6])
                predict_similarity_list.append(predicted_similarity_score)
                actual_similarity_list.append(row[4]/5.0)


        '''
            Evaluation on Sick 2017 Dataset

        '''

        if file_name=='sick':
            input_df= self.read_dataset("Datasets/sick.xlsx") 
            for index, row in input_df.iterrows():
                predicted_similarity_score=self.predict(row['sentence_A'],row['sentence_B'])
                actual_similarity_list.append(row['relatedness_score']/5.0)
                predict_similarity_list.append(predicted_similarity_score)


        '''
            Evaluation on SemEval 2014 Dataset

        '''

        if file_name=='semEval2014':
           input_df=self.read_dataset('Datasets/semEval2014.xlsx')
           for index, row in input_df.iterrows():
                predicted_similarity_score=self.predict(row[0],row[1])
                predict_similarity_list.append(predicted_similarity_score)
                actual_similarity_list.append(row[2]/5.0)

        return self.evaluate(actual_similarity_list,predict_similarity_list)
    

    def predict(self, data_X, data_Y):
        '''
            Predicts the similarity between data_X, data_Y
            Input parameters       : data_X - sentence1
                                   : data_Y - sentence2

            Returns                : similarity_score [0-1]

        '''
        m1 = MinHash(num_perm=128)
        m2 = MinHash(num_perm=128)
        set1=set(data_X.split(' '))
        set2=set(data_Y.split(' '))
        for d in set1:
            m1.update(d.encode('utf8'))
        for d in set2:
            m2.update(d.encode('utf8'))
        return m1.jaccard(m2)

    def evaluate(self, actual_list, predicted_list):
        '''
            Evaluate between actual_list and predicted list
            Input parameters        : actual_list ( List of actual similarity values )
                                    : predicted_list ( list of predicted similarity values )
            Returns                 : - Pearson Correlation Coefficient
                                      - Spearman Correlation Coefficient

        '''

        evaluation_score, p_val=pearsonr(actual_list,predicted_list)
        spearman, p_val=spearmanr(actual_list,predicted_list)
        return evaluation_score,spearman    


def _random_name(length):
    # For use with Redis, we return bytes
    return ''.join(random.choice(string.ascii_lowercase)
                   for _ in range(length)).encode('utf8')






_integration_precision = 0.001
def _integration(f, a, b):
    p = _integration_precision
    area = 0.0
    x = a
    while x < b:
        area += f(x+0.5*p)*p
        x += p
    return area, None

try:
    from scipy.integrate import quad as integrate
except ImportError:
    # For when no scipy installed
    integrate = _integration


def _false_positive_probability(threshold, b, r):
    _probability = lambda s : 1 - (1 - s**float(r))**float(b)
    a, err = integrate(_probability, 0.0, threshold) 
    return a


def _false_negative_probability(threshold, b, r):
    _probability = lambda s : 1 - (1 - (1 - s**float(r))**float(b))
    a, err = integrate(_probability, threshold, 1.0)
    return a


def _optimal_param(threshold, num_perm, false_positive_weight,
        false_negative_weight):
    '''
    Compute the optimal `MinHashLSH` parameter that minimizes the weighted sum
    of probabilities of false positive and false negative.
    '''
    min_error = float("inf")
    opt = (0, 0)
    for b in range(1, num_perm+1):
        max_r = int(num_perm / b)
        for r in range(1, max_r+1):
            fp = _false_positive_probability(threshold, b, r)
            fn = _false_negative_probability(threshold, b, r)
            error = fp*false_positive_weight + fn*false_negative_weight
            if error < min_error:
                min_error = error
                opt = (b, r)
    return opt



# Sample workflow of the class :
def main():
    print('-------------------EVALUATION------------------------')
    print('-----------------------------------------------------')
    print('------------------SICK DATASET-----------------------')
    print('Using HashFunc : xxhash.xxh32')
    minHash=MinHash()
    print(minHash.evaluate_util('sick'))
    print('-----------------------------')
    print('Using HashFunc : mmh3.hash123')
    minHash=MinHash()
    print(minHash.evaluate_util('sick'))
    print('-----------------------------------------------------')
    print('-----------------SemEval 2014 DATASET----------------')
    print('Using HashFunc : xxhash.xxh32')
    minHash=MinHash()
    print(minHash.evaluate_util('semEval2014'))
    print('-----------------------------')
    print('Using HashFunc : mmh3.hash123')
    minHash=MinHash()
    print(minHash.evaluate_util('semEval2014'))
    print('------------------------------------------------------')
    print('----------------SemEval 2017 DATASET------------------')
    print('Using HashFunc : xxhash.xxh32')
    minHash=MinHash()
    print(minHash.evaluate_util('semEval'))
    print('------------------------------------------------------')
    print('Using HashFunc : mmh3.hash123')
    minHash=MinHash()
    print(minHash.evaluate_util('semEval'))
    print('-------------------------------------------------------')



if __name__ == '__main__':
    main()


