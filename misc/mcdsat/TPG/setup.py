#!/usr/bin/env python

""" Setup script for Toy Parser Generator

Just run "python setup.py install" to install TPG


Toy Parser Generator: A Python parser generator
Copyright (C) 2002 Christophe Delord
 
This library is free software; you can redistribute it and/or
modify it under the terms of the GNU Lesser General Public
License as published by the Free Software Foundation; either
version 2.1 of the License, or (at your option) any later version.

This library is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public
License along with this library; if not, write to the Free Software
Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA

For further information about TPG you can visit
http://christophe.delord.free.fr/en/tpg

"""

import os
import sys
import operator
from glob import glob

# BEFORE importing distutils, remove MANIFEST. distutils doesn't properly
# update it when the contents of directories change.
if os.path.exists('MANIFEST'): os.remove('MANIFEST')

from distutils.core import setup

# Update the documentation when building a source dist
if 'sdist' in sys.argv[1:]:

    to_update = [
        # Precompiled TPG parser
        ( 'tpg.py', ['tpg.pyg'],
            "tpg tpg.pyg"
        ),
        # Documentation / Tutorial
#        ( 'doc/tpg.html', ['doc/*.tex'],
#            #"cd doc && htlatex tpg html,2 && web2png -t && rm tpg.dvi"
#            "cd doc && htlatex tpg html,2,png \" -g.png\" && rm tpg.dvi"
#        ),
        ( 'doc/tpg.pdf', ['doc/*.tex'],
            "cd doc && pdflatex tpg && pdflatex tpg && pdflatex tpg"
        ),
#        ( 'doc/tpg.dvi', ['doc/*.tex'],
#            "cd doc && latex tpg && latex tpg && latex tpg"
#        ),
    ]

    def target_outdated(target, deps):
        try:
            target_time = os.path.getmtime(target)
        except os.error:
            return 1
        for dep in deps:
            dep_time = os.path.getmtime(dep)
            if dep_time > target_time:
                return 1
        return 0

    def target_update(target, deps, cmd):
        deps = reduce(operator.add, map(glob, deps), [])
        if target_outdated(target, deps):
            os.system(cmd)

    for target in to_update:
        target_update(*target)

# tpg.py contains version, authors, license, url, keywords, etc.
import tpg

# Call the setup() routine which does most of the work
setup(name             = tpg.__tpgname__,
      version          = tpg.__version__,
      description      = tpg.__description__,
      long_description = tpg.__long_description__,
      author           = tpg.__author__,
      author_email     = tpg.__email__,
      url              = tpg.__url__,
      maintainer       = tpg.__author__,
      maintainer_email = tpg.__email__,
      license          = tpg.__license__,
      platforms        = ['Linux', 'Unix', 'Mac OSX', 'Windows XP/2000/NT', 'Windows 95/98/ME'],
      keywords         = ['Parsing', 'Parser', 'Generator', 'Python'],
      py_modules       = ['tpg'],
      scripts          = ['tpg'],
)

