#!/usr/bin/env python

import re
import sys
import unittest

import tpg

print "*"*70
print "*"
print "* Unit tests for %(__name__)s %(__version__)s (%(__date__)s)"%tpg.__dict__
print "*"
print "* Platform : %s"%sys.platform.replace('\n', ' ')
print "* Version  : %s"%sys.version.replace('\n', ' ')
print "*"
print "* Please report bug to %(__author__)s (%(__email__)s)"%tpg.__dict__
print "* for further detail read %(__url__)s"%tpg.__dict__
print "*"
print "*"*70

for PARSER, VERBOSE in ( (tpg.Parser, None),
                         (tpg.VerboseParser, 0),
                         (tpg.VerboseParser, 1),
                         (tpg.VerboseParser, 2),
                       ):
    for LEXER in tpg.TPGParser.Options.option_dict['lexer'][0].keys():

        print "*"*70
        if VERBOSE is None:
            print "* %s %s"%(PARSER.__name__, LEXER)
        else:
            print "* %s verbose=%s %s"%(PARSER.__name__, VERBOSE, LEXER)
        print "*"*70

        class LexerOptionsTestCase(unittest.TestCase):

            class WordBounded(PARSER):
                __doc__ = r"""
                    set lexer = %(LEXER)s

                    set word_boundary = True

                    separator spaces '\s+' ;

                    token a 'a' ;
                    token b 'b' ;
                    token w '\w+' ;

                    START/lst ->            $ lst = []
                        (   a/t             $ lst.append(t)
                        |   b/t             $ lst.append(t)
                        |   w/t             $ lst.append(t)
                        )*
                        ;
                """%tpg.Py()
                verbose = VERBOSE

            def testWordBound(self):
                p = self.WordBounded()
                self.assertEquals(p('abcabc a b c'), ['abcabc', 'a', 'b', 'c'])
                self.assertEquals(p('abc abc a b c'), ['abc', 'abc', 'a', 'b', 'c'])

            class NotWordBounded(PARSER):
                __doc__ = r"""
                    set lexer = %(LEXER)s

                    set word_boundary = False

                    separator spaces '\s+' ;

                    token a 'a' ;
                    token b 'b' ;
                    token w '\w+' ;

                    START/lst ->            $ lst = []
                        (   a/t             $ lst.append(t)
                        |   b/t             $ lst.append(t)
                        |   w/t             $ lst.append(t)
                        )*
                        ;
                """%tpg.Py()
                verbose = VERBOSE

            def testNotWordBound(self):
                p = self.NotWordBounded()
                if LEXER in ('Lexer', 'CacheLexer'):
                    self.assertEquals(p('abcabc a b c'), ['abcabc', 'a', 'b', 'c'])
                    self.assertEquals(p('abc abc a b c'), ['abc', 'abc', 'a', 'b', 'c'])
                else:
                    self.assertEquals(p('abcabc a b c'), ['a', 'b', 'cabc', 'a', 'b', 'c'])
                    self.assertEquals(p('abc abc a b c'), ['a', 'b', 'c', 'a', 'b', 'c', 'a', 'b', 'c'])

            class WordBounded2(PARSER):
                __doc__ = r"""
                    set lexer = %(LEXER)s

                    set word_boundary = True

                    separator spaces '\s+' ;

                    token abc 'abc' ;
                    token def 'def:' ;
                    token ghi ':ghi' ;
                    token other '\w+|:' ;

                    START/lst ->            $ lst = []
                        (   abc/t           $ lst.append(t)
                        |   def/t           $ lst.append(t)
                        |   ghi/t           $ lst.append(t)
                        |   other
                        )*
                        ;
                """%tpg.Py()
                verbose = VERBOSE

            def testWordBounded2(self):
                p = self.WordBounded2()
                self.assertEquals(p("abc :titi: :def: toto :ghi:"), ['abc', 'def:', ':ghi'])
                self.assertEquals(p(":abc:def:ghi:"), ['abc', 'def:'])
                self.assertEquals(p(":abc:ghi:def:"), ['abc', ':ghi', 'def:'])
                self.assertEquals(p("::abc::def::ghi::"), ['abc', 'def:', ':ghi'])
                self.assertEquals(p("::abc::ghi::def::"), ['abc', ':ghi', 'def:'])

            class IgnoreCase(PARSER):
                __doc__ = r"""
                    set lexer = %(LEXER)s

                    set lexer_ignorecase = True

                    separator spaces '\s+' ;

                    token a 'a' ;
                    token b 'B' ;

                    START/lst ->            $ lst = []
                        (   a/t             $ lst.append(t)
                        |   b/t             $ lst.append(t)
                        )*
                        ;
                """%tpg.Py()
                verbose = VERBOSE

            def testIgnoreCase(self):
                p = self.IgnoreCase()
                self.assertEquals(p('a A b B'), ['a', 'A', 'b', 'B'])

            class NotIgnoreCase(PARSER):
                __doc__ = r"""
                    set lexer = %(LEXER)s

                    set lexer_ignorecase = False

                    separator spaces '\s+' ;

                    token a 'a' ;
                    token b 'B' ;

                    START/lst ->            $ lst = []
                        (   a/t             $ lst.append(t)
                        |   b/t             $ lst.append(t)
                        )*
                        ;
                """%tpg.Py()
                verbose = VERBOSE

            def testNotIgnoreCase(self):
                p = self.NotIgnoreCase()
                if LEXER in ('ContextSensitiveLexer',):
                    self.assertRaises(tpg.SyntacticError, p, 'a A b B')
                else:
                    self.assertRaises(tpg.LexicalError, p, 'a A b B')
                self.assertEquals(p('a B a B'), ['a', 'B', 'a', 'B'])

            class Multiline(PARSER):
                __doc__ = r"""
                    set lexer = %(LEXER)s

                    set lexer_multiline = True

                    token b '^b' ;
                    token e 'e$' ;

                    token w '\w' ;
                    
                    separator spaces '\s+' ;

                    START/$nb,nw,ne$ ->         $ nb, nw, ne = 0, 0, 0
                        (   b                   $ nb += 1
                        |   e                   $ ne += 1
                        |   w                   $ nw += 1
                        )*
                        ;
                """%tpg.Py()
                verbose = VERBOSE

            def testMultiline(self):
                p = self.Multiline()
                self.assertEquals(p('b b w e e\nb b w e e\nb b w e e'), (3, 9, 3))

            class NotMultiline(PARSER):
                __doc__ = r"""
                    set lexer = %(LEXER)s

                    set lexer_multiline = False

                    token b '^b' ;
                    token e 'e$' ;

                    token w '\w' ;
                    
                    separator spaces '\s+' ;

                    START/$nb,nw,ne$ ->         $ nb, nw, ne = 0, 0, 0
                        (   b                   $ nb += 1
                        |   e                   $ ne += 1
                        |   w                   $ nw += 1
                        )*
                        ;
                """%tpg.Py()
                verbose = VERBOSE

            def testNotMultiline(self):
                p = self.NotMultiline()
                self.assertEquals(p('b b w e e\nb b w e e\nb b w e e'), (1, 13, 1))

            class DotAll(PARSER):
                __doc__ = r"""
                    set lexer = %(LEXER)s

                    set lexer_dotall = True

                    token ch '.' ;
                    token nl '\n' ;

                    START/$c,n$ ->              $ c, n = 0, 0
                        (   ch                  $ c += 1
                        |   nl                  $ n += 1
                        )*
                        ;
                """%tpg.Py()
                verbose = VERBOSE

            def testDotAll(self):
                p = self.DotAll()
                self.assertEquals(p('a\nb\nc\n'), (6, 0))

            class NotDotAll(PARSER):
                __doc__ = r"""
                    set lexer = %(LEXER)s

                    set lexer_dotall = False

                    token ch '.' ;
                    token nl '\n' ;

                    START/$c,n$ ->              $ c, n = 0, 0
                        (   ch                  $ c += 1
                        |   nl                  $ n += 1
                        )*
                        ;
                """%tpg.Py()
                verbose = VERBOSE

            def testNotDotAll(self):
                p = self.NotDotAll()
                self.assertEquals(p('a\nb\nc\n'), (3, 3))

            class Verbose(PARSER):
                __doc__ = (r"""
                    set lexer = %(LEXER)s

                    set lexer_verbose = True

                    token foobar "foo bar" ;
                    token foo "foo" ;
                    token bar "bar" ;

                    token triple_quoted_1 '''
                        "{3}
                        (   \\.
                        |   "{0,2} [^"\\]+
                        )*
                        "{3}
                    ''' ;

                """ + r'''
                    token triple_quoted_2 """
                        '{3}
                        (   \\.
                        |   '{0,2} [^'\\]+
                        )*
                        '{3}
                    """ ;
                ''' + r"""

                    separator spaces '\s+' ;

                    START/lst ->                $ lst = []
                        (   foobar              $ lst.append(1)
                        |   foo                 $ lst.append(2)
                        |   bar                 $ lst.append(3)
                        |   triple_quoted_1     $ lst.append(4)
                        |   triple_quoted_2     $ lst.append(5)
                        )*
                        ;
                """)%tpg.Py()
                verbose = VERBOSE

            def testVerbose(self):
                p = self.Verbose()
                self.assertEquals(p('''
                    foobar foo bar foo  bar """ hello 'world' """ \''' ''hello'' ""universe"" \'''
                '''), [1, 2, 3, 2, 3, 4, 5])

            class NotVerbose(PARSER):
                __doc__ = r"""
                    set lexer = %(LEXER)s

                    set lexer_verbose = False

                    token foobar "foo bar" ;
                    token foo "foo" ;
                    token bar "bar" ;

                    separator spaces '\s+' ;

                    START/lst ->                $ lst = []
                        (   foobar              $ lst.append(1)
                        |   foo                 $ lst.append(2)
                        |   bar                 $ lst.append(3)
                        )*
                        ;
                """%tpg.Py()
                verbose = VERBOSE

            def testNotVerbose(self):
                p = self.NotVerbose()
                if LEXER in ('ContextSensitiveLexer',):
                    self.assertRaises(tpg.SyntacticError, p, 'foobar foo bar foo  bar')
                else:
                    self.assertRaises(tpg.LexicalError, p, 'foobar foo bar foo  bar')
                self.assertEquals(p('foo bar foo  bar'), [1, 2, 3])

        class LexersTestCase(unittest.TestCase):

            class Parser(PARSER):
                __doc__ = r"""
                    set lexer = %(LEXER)s
                    set lexer = NamedGroupLexer

                    token float '\d+\.\d+' $ float
                    token int '\d+' int
                    token text '\w+' ;
                    token str '".*?"' $ lambda s: s[1:-1]
                    token brackets '\{(.|\n)*?\}' "{}"

                    separator spaces '\s+' ;
                    separator comments '\(.*?\)' ;

                    START/$t, type(t)$ ->
                        (   float/t
                        |   int/t
                        |   text/t
                        |   str/t
                        |   brackets/t
                        )
                        ;

                    POSITIONS/lst ->                $ lst = []
                        (                           $ line, column = self.line(), self.column()
                            (   text text int       $ lst.append(None)
                            |   text int            $ lst.append(None)
                            |   text/t              $ lst.append((t, line, column))
                            |   brackets/t          $ lst.append((t, line, column))
                            )
                        )*
                        ;   
                """%tpg.Py()
                verbose = VERBOSE

            def testLexers(self):
                p = self.Parser()
                self.assertEquals(p('  314'), (314, int))
                self.assertEquals(p('3.14   '), (3.14, float))
                self.assertEquals(p(' (one identifier) cool'), ('cool', str))
                self.assertEquals(p('"Super cool"  (!!!) '), ('Super cool', str))
                self.assertEquals(p('  { ... }  (!!!) '), ('{}', str))

            def testPositions(self):
                p = self.Parser()
                s = r"""a b c
                d e f
                { ...
                ... } y
                z
                """
                self.assertEquals(p.parse('POSITIONS',s), [('a', 1, 1), ('b', 1, 3), ('c', 1, 5),
                                                           ('d', 2, 17), ('e', 2, 19), ('f', 2, 21),
                                                           ('{}', 3, 17), ('y', 4, 23), ('z', 5, 17),
                                                          ])

        class BacktrackingTestCase(unittest.TestCase):

            class Parser(PARSER):
                __doc__ = r"""
                    set lexer = %(LEXER)s

                    set word_boundary = False

                    START/n ->
                        (   'a' 'b' 'c' 'd'         $ n = 1
                        |   'a' 'b' 'd' 'c'         $ n = 2
                        |   'a'
                            (   'c' 'b' 'd'         $ n = 3
                            |   'c' 'd' 'b'         $ n = 4
                            |   'd'
                                (   'b' 'c'         $ n = 5
                                |   'c' 'b'         $ n = 6
                                )
                            )
                        |   'b'
                            (   'a'
                                (   'c' 'd'         $ n = 7
                                |   'd' 'c'         $ n = 8
                                )
                            |   'c'
                                (   'a' 'd'         $ n = 9
                                |   'd' 'a'         $ n = 10
                                )
                            )
                        |   'e'                     $ n = 11
                        )
                        ;
                    """%tpg.Py()

            def testBacktracking(self):
                p = self.Parser()
                self.assertEquals(p('abcd'), 1)
                self.assertEquals(p('abdc'), 2)
                self.assertEquals(p('acbd'), 3)
                self.assertEquals(p('acdb'), 4)
                self.assertEquals(p('adbc'), 5)
                self.assertEquals(p('adcb'), 6)
                self.assertEquals(p('bacd'), 7)
                self.assertEquals(p('badc'), 8)
                self.assertEquals(p('bcad'), 9)
                self.assertEquals(p('bcda'), 10)
                self.assertEquals(p('e'), 11)
                self.assertRaises(tpg.SyntacticError, p, 'abce')
                self.assertRaises(tpg.SyntacticError, p, 'abec')
                self.assertRaises(tpg.SyntacticError, p, 'aebc')
                self.assertRaises(tpg.SyntacticError, p, 'eabd')
                self.assertRaises(tpg.SyntacticError, p, 'cabd')

        class ExtractTestCase(unittest.TestCase):

            class Parser(PARSER):
                __doc__ = r"""
                    set lexer = %(LEXER)s

                    separator spaces '\s+' ;

                    START/lst ->            $ lst = []
                        @start
                        (
                            '\('
                                @start1
                                '\w+'*
                                @stop1
                            '\)'            $ lst.append(self.extract(start1, stop1))
                        )*
                        @stop               $ lst.append(self.extract(start, stop))
                        ;
                """%tpg.Py()

            def testExtract(self):
                p = self.Parser()
                self.assertEquals(p(''), [''])
                self.assertEquals(p('() ()'), ['', '', '() ()'])
                self.assertEquals(p('  ( ) (  )   '), ['', '', '( ) (  )'])
                self.assertEquals(p('(a b)  (c  d)'), ['a b', 'c  d', '(a b)  (c  d)'])
                self.assertEquals(p('  (a b)  (  c  d   )   '), ['a b', 'c  d', '(a b)  (  c  d   )'])

        class AxiomTestCase(unittest.TestCase):

            class Parser(PARSER):
                __doc__ = r"""
                    set lexer = %(LEXER)s

                    token word '\w+' ;

                    START/"Axiom == START" -> ;

                    SYMBOL1/"Axiom == SYMBOL1" -> ;

                    SYMBOL2/"Axiom == SYMBOL2" -> ;
                """%tpg.Py()

            def testAxiom(self):
                p = self.Parser()
                self.assertEquals(p(''), "Axiom == START")
                self.assertEquals(p.parse('START', ''), "Axiom == START")
                self.assertEquals(p.parse('SYMBOL1', ''), "Axiom == SYMBOL1")
                self.assertEquals(p.parse('SYMBOL2', ''), "Axiom == SYMBOL2")
                self.assertRaises(AttributeError, p.parse, 'SYMBOL3', '')
                self.assertRaises(tpg.SyntacticError, p, 'x')
                self.assertRaises(tpg.SyntacticError, p.parse ,'START', 'x')
                self.assertRaises(tpg.SyntacticError, p.parse ,'SYMBOL1', 'x')
                self.assertRaises(tpg.SyntacticError, p.parse ,'SYMBOL2', 'x')
                self.assertRaises(AttributeError, p.parse ,'SYMBOL3', 'x')

        class TokenInfoTestCase(unittest.TestCase):

            class Parser(PARSER):
                __doc__ = r"""
                    set lexer = %(LEXER)s
                    set lexer_dotall = True

                    separator spaces '\s+' ;

                    token tok "\(.*?\)" ;

                    START/lst ->                $ lst = []
                        (                       $ lst.append((self.line(), self.column()))
                            @t                  $ lst.append((t.line, t.column))
                            tok                 $ lst.append((self.line(t), self.column(t)))
                            @t                  $ lst.append((t.line, t.column))
                        )*
                        ;
                """%tpg.Py()

            def testTokenInfo(self):
                p = self.Parser()
                if LEXER in ('ContextSensitiveLexer', ):
                    self.assertEquals(p(''), [(1,1), (1,1)])
                    self.assertEquals(p(' ( ) (\n) (w)'), [(1,1), (1,1), (1,1), (1,2),
                                                           (1,2), (1,2), (1,2), (1,6),
                                                           (1,6), (1,6), (1,6), (2,3),
                                                           (2,3), (2,3)
                                                          ])
                else:
                    self.assertEquals(p(''), [(1,1), (1,1)])
                    self.assertEquals(p(' ( ) (\n) (w)'), [(1,2), (1,2), (1,2), (1,6),
                                                           (1,6), (1,6), (1,6), (2,3),
                                                           (2,3), (2,3), (2,3), (2,6),
                                                           (2,6), (2,6)
                                                          ])

        class CheckErrorTestCase(unittest.TestCase):

            class Parser(PARSER):
                __doc__ = r"""
                    set lexer = %(LEXER)s

                    separator spaces '\s+' ;
                    token int '\-?\d+' int ;

                    POSITIVE_1/i -> int/i check $i>0$ ;
                    POSITIVE_2/i -> int/i $self.check(i>0)$ ;
                    POSITIVE_3/i -> int/i ( check $i>0$ | error "i <= 0" ) ;
                    POSITIVE_4/i -> int/i ( check $i>0$ | error $"%%d <= 0"%%i$ ) ;
                    POSITIVE_5/i -> int/i ( check $i>0$ | $self.error("%%d <= 0"%%i)$ );
                    POSITIVE_6/i -> int/i ( check $i>0$ | $i=None$ ) ;
                    POSITIVE_7/i -> int/i ( $self.check(i>0)$ | $i=None$ ) ;
                """%tpg.Py()

            def testCheckError(self):
                p = self.Parser()
                for x in ("1", "18"):
                    self.assertEquals(p.parse('POSITIVE_1', x), int(x))
                    self.assertEquals(p.parse('POSITIVE_2', x), int(x))
                    self.assertEquals(p.parse('POSITIVE_3', x), int(x))
                    self.assertEquals(p.parse('POSITIVE_4', x), int(x))
                    self.assertEquals(p.parse('POSITIVE_5', x), int(x))
                    self.assertEquals(p.parse('POSITIVE_6', x), int(x))
                    self.assertEquals(p.parse('POSITIVE_7', x), int(x))
                for x in ("0", "-36"):
                    self.assertRaises(tpg.SyntacticError, p.parse, 'POSITIVE_1', x)
                    self.assertRaises(tpg.SyntacticError, p.parse, 'POSITIVE_2', x)
                    self.assertRaises(tpg.SemanticError, p.parse, 'POSITIVE_3', x)
                    self.assertRaises(tpg.SemanticError, p.parse, 'POSITIVE_4', x)
                    self.assertRaises(tpg.SemanticError, p.parse, 'POSITIVE_5', x)
                    self.assertEquals(p.parse('POSITIVE_6', x), None)
                    self.assertEquals(p.parse('POSITIVE_7', x), None)

        class RepetitionTestCase(unittest.TestCase):

            class Parser(PARSER):
                __doc__ = r"""
                    set lexer = %(LEXER)s

                    separator spaces '\s+' ;

                    token w '\w+' ;

                    STAR/n -> $n=0$ ( w $n=n+1$ )* ;
                    PLUS/n -> $n=0$ ( w $n=n+1$ )+ ;
                    QUES/n -> $n=0$ ( w $n=n+1$ )? ;
                    REP1/n -> $n=0$ ( w $n=n+1$ ){} ;
                    REP2/n -> $n=0$ ( w $n=n+1$ ){3} ;
                    REP3/n -> $n=0$ ( w $n=n+1$ ){,} ;
                    REP4/n -> $n=0$ ( w $n=n+1$ ){,3} ;
                    REP5/n -> $n=0$ ( w $n=n+1$ ){2,} ;
                    REP6/n -> $n=0$ ( w $n=n+1$ ){2,5} ;
                """%tpg.Py()

            def testStar(self):
                p = self.Parser()
                self.assertEquals(p.parse('STAR', ''), 0)
                self.assertEquals(p.parse('STAR', 'a'), 1)
                self.assertEquals(p.parse('STAR', 'a b'), 2)
                self.assertEquals(p.parse('STAR', 'a b c d e f'), 6)

            def testPlus(self):
                p = self.Parser()
                self.assertRaises(tpg.SyntacticError, p.parse, 'PLUS', '')
                self.assertEquals(p.parse('PLUS', 'a'), 1)
                self.assertEquals(p.parse('PLUS', 'a b'), 2)
                self.assertEquals(p.parse('PLUS', 'a b c d e f'), 6)

            def testQuestion(self):
                p = self.Parser()
                self.assertEquals(p.parse('QUES', ''), 0)
                self.assertEquals(p.parse('QUES', 'a'), 1)
                self.assertRaises(tpg.SyntacticError, p.parse, 'QUES', 'a b')
                self.assertRaises(tpg.SyntacticError, p.parse, 'QUES', 'a b c d e f')

            def testREP1(self):
                p = self.Parser()
                self.assertEquals(p.parse('REP1', ''), 0)
                self.assertRaises(tpg.SyntacticError, p.parse, 'REP1', '1')
                self.assertRaises(tpg.SyntacticError, p.parse, 'REP1', '1 2')
                self.assertRaises(tpg.SyntacticError, p.parse, 'REP1', '1 2 3 4 5 6')

            def testREP2(self):
                p = self.Parser()
                self.assertRaises(tpg.SyntacticError, p.parse, 'REP2', '1')
                self.assertRaises(tpg.SyntacticError, p.parse, 'REP2', '1 2')
                self.assertEquals(p.parse('REP2', '1 2 3'), 3)
                self.assertRaises(tpg.SyntacticError, p.parse, 'REP2', '1 2 3 4')
                self.assertRaises(tpg.SyntacticError, p.parse, 'REP2', '1 2 3 4 5 6')

            def testREP3(self):
                p = self.Parser()
                self.assertEquals(p.parse('REP3', ''), 0)
                self.assertEquals(p.parse('REP3', '1'), 1)
                self.assertEquals(p.parse('REP3', '1 2'), 2)
                self.assertEquals(p.parse('REP3', '1 2 3'), 3)
                self.assertEquals(p.parse('REP3', '1 2 3 4'), 4)
                self.assertEquals(p.parse('REP3', '1 2 3 4 5'), 5)
                self.assertEquals(p.parse('REP3', '1 2 3 4 5 6'), 6)

            def testREP4(self):
                p = self.Parser()
                self.assertEquals(p.parse('REP4', ''), 0)
                self.assertEquals(p.parse('REP4', '1'), 1)
                self.assertEquals(p.parse('REP4', '1 2'), 2)
                self.assertEquals(p.parse('REP4', '1 2 3'), 3)
                self.assertRaises(tpg.SyntacticError, p.parse, 'REP4', '1 2 3 4')
                self.assertRaises(tpg.SyntacticError, p.parse, 'REP4', '1 2 3 4 5')
                self.assertRaises(tpg.SyntacticError, p.parse, 'REP4', '1 2 3 4 5 6')

            def testREP5(self):
                p = self.Parser()
                self.assertRaises(tpg.SyntacticError, p.parse, 'REP5', '')
                self.assertRaises(tpg.SyntacticError, p.parse, 'REP5', '1')
                self.assertEquals(p.parse('REP5', '1 2'), 2)
                self.assertEquals(p.parse('REP5', '1 2 3'), 3)
                self.assertEquals(p.parse('REP5', '1 2 3 4 5 6'), 6)

            def testREP6(self):
                p = self.Parser()
                self.assertRaises(tpg.SyntacticError, p.parse, 'REP6', '')
                self.assertRaises(tpg.SyntacticError, p.parse, 'REP6', '1')
                self.assertEquals(p.parse('REP6', '1 2'), 2)
                self.assertEquals(p.parse('REP6', '1 2 3'), 3)
                self.assertEquals(p.parse('REP6', '1 2 3 4'), 4)
                self.assertEquals(p.parse('REP6', '1 2 3 4 5'), 5)
                self.assertRaises(tpg.SyntacticError, p.parse, 'REP6', '1 2 3 4 5 6')

        class ArgsTestCase(unittest.TestCase):

            class Parser(PARSER):
                __doc__ = r"""
                    set lexer = %(LEXER)s

                    separator spaces '\s+' ;

                    START<a, b, *args, **kws>/<a, b, args, kws> -> S<a, b, *args, **kws>/<a, b, args, kws> ;
                    S<a, b, *args, **kws>/<a, b, args, kws> -> ;

                """%tpg.Py()

            def testArgs(self):
                p = self.Parser()
                self.assertRaises(TypeError, p, '', 'a')
                self.assertEquals(p('', 'a', 'b'), ('a', 'b', (), {}))
                self.assertEquals(p('', 'a', 'b', 'c'), ('a', 'b', ('c',), {}))
                self.assertEquals(p('', 'a', 'b', 'c', 'd'), ('a', 'b', ('c', 'd'), {}))
                self.assertEquals(p('', 'a', 'b', 'c', 'd', e='e'), ('a', 'b', ('c', 'd'), {'e':'e'}))
                self.assertEquals(p('', 'a', 'b', 'c', 'd', e='e', f='f'), ('a', 'b', ('c', 'd'), {'e':'e', 'f':'f'}))
                self.assertEquals(p('', 'a', 'b', **({'e':'e', 'f':'f'})), ('a', 'b', (), {'e':'e', 'f':'f'}))
                self.assertEquals(p('', 'a', 'b', *('c', 'd')), ('a', 'b', ('c', 'd'), {}))
                self.assertEquals(p('', 'a', 'b', *('c', 'd'), **({'e':'e', 'f':'f'})), ('a', 'b', ('c', 'd'), {'e':'e', 'f':'f'}))

        class PyExprTestCase(unittest.TestCase):

            class Parser(PARSER):
                __doc__ = r"""
                    set lexer = %(LEXER)s

                    START/x ->
                        $ foo = "*foo*"
                        (   "1" IDENT<foo>/x1 IDENT<314>/x2 $ x = x1, x2 $
                        |   "2" STRING<"2">/x
                        |   "3" CODE<$"a"+"b"$>/x
                        )
                        ;

                    IDENT<i>/i -> ;
                    STRING<s>/"a string" -> check $s=="2"$ ;
                    CODE<x>/$1+2$ -> check $x=="ab"$ ;
                """%tpg.Py()

            def testExpr(self):
                p = self.Parser()
                self.assertEquals(p("1"), ("*foo*", 314))
                self.assertEquals(p("2"), "a string")
                self.assertEquals(p("3"), 3)

        class EmptyChoiceTestCase(unittest.TestCase):

            def OK(self):
                class Parser(PARSER):
                    __doc__ = r"""
                        set lexer = %(LEXER)s

                        A ->
                            (   x       # not empty
                            |   x y     # not empty
                            )
                            ;

                        B ->
                            (   x       # not empty
                            |   x y     # not empty
                            |           # empty
                            )
                            ;

                        C ->
                            (   x       # not empty
                            |   x y     # not empty
                            |   ( )     # empty
                            )
                            ;
                    """%tpg.Py()

            def NOK1(self):
                class Parser(PARSER):
                    __doc__ = r"""
                        set lexer = %(LEXER)s

                        A ->
                            (           # empty !!!
                            |   x       # not empty
                            |   x y     # not empty
                            )
                            ;

                    """%tpg.Py()

            def NOK2(self):
                class Parser(PARSER):
                    __doc__ = r"""
                        set lexer = %(LEXER)s

                        B ->
                            (   ( ( ) ) # empty !!!
                            |   x       # not empty
                            |   x y     # not empty
                            )
                            ;

                    """%tpg.Py()

            def NOK3(self):
                class Parser(PARSER):
                    __doc__ = r"""
                        set lexer = %(LEXER)s

                        C ->
                            (   x       # not empty
                            |           # empty !!!
                            |   x y     # not empty
                            )
                            ;

                    """%tpg.Py()

            def NOK4(self):
                class Parser(PARSER):
                    __doc__ = r"""
                        set lexer = %(LEXER)s

                        D ->
                            (   x       # not empty
                            |   ( )     # empty !!!
                            |   x y     # not empty
                            )
                            ;

                    """%tpg.Py()

            def testEmptyLast(self):
                self.assertEquals(self.OK(), None)

            def testEmptyNonLast(self):
                self.assertRaises(tpg.SyntacticError, self.NOK1)
                self.assertRaises(tpg.SyntacticError, self.NOK2)
                self.assertRaises(tpg.SyntacticError, self.NOK3)
                self.assertRaises(tpg.SyntacticError, self.NOK4)

        try:
            unittest.main()
        except SystemExit, failed:
            if failed.args[0]:
                raise

