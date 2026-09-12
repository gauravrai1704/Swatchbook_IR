"""
A standalone, dependency-free implementation of the Porter Stemming
Algorithm (Porter, 1980, "An algorithm for suffix stripping").

We implement it ourselves (rather than relying on nltk) so the whole
project runs with zero third-party dependencies -- important for a
reproducible course submission that a TA can run on any machine.

This is the classic "Porter1" algorithm: 5 sequential suffix-stripping
steps driven by measuring the "measure" (m) of the stem -- the number
of consonant-vowel (VC) sequences -- and by simple suffix conditions.
"""

VOWELS = "aeiou"


class PorterStemmer:
    """A lightweight implementation of the classic Porter stemming algorithm.

    The goal is to reduce word variants to a common root form so that search can match
    related terms such as "running", "runs", and "runner" under a single stem. This
    is especially important in information retrieval, where exact-string matching alone
    would miss semantically related terms.
    """

    def __init__(self):
        # `b` is the current working word; `k` is the last valid index of the word.
        # `j` is used as a temporary boundary while suffix rules are being checked.
        self.b = ""
        self.k = 0
        self.j = 0

    def _cons(self, i):
        """Return True if index i contains a consonant in the current word.

        We follow the Porter definition: y counts as a consonant only when it is not
        preceded by another consonant, which makes the stemming rules less brittle.
        """
        ch = self.b[i]
        if ch in VOWELS:
            return False
        if ch == 'y':
            return True if i == 0 else not self._cons(i - 1)
        return True

    def _m(self):
        """Return the Porter 'measure' m: the number of consonant-vowel transitions.

        This is the core signal used in many suffix rules. A stem is considered more
        "word-like" when it contains more VC sequences, and the algorithm only strips
        suffixes when the resulting stem satisfies those conditions.
        """
        i = 0
        n = 0
        while True:
            if i > self.j:
                return n
            if not self._cons(i):
                break
            i += 1
        i += 1
        while True:
            while True:
                if i > self.j:
                    return n
                if self._cons(i):
                    break
                i += 1
            i += 1
            n += 1
            while True:
                if i > self.j:
                    return n
                if not self._cons(i):
                    break
                i += 1
            i += 1

    def _vowel_in_stem(self):
        """True if the current stem contains at least one vowel."""
        return any(not self._cons(i) for i in range(self.j + 1))

    def _double_c(self, j):
        """Check whether the current stem ends with a double consonant, like 'll' or 'ss'."""
        if j < 1:
            return False
        if self.b[j] != self.b[j - 1]:
            return False
        return self._cons(j)

    def _cvc(self, i):
        """Check for the classic consonant-vowel-consonant pattern used in step 1 and step 5.

        This helps guard against over-aggressive stripping in words like "hop" versus
        "hopping", where the algorithm should not normalize too aggressively.
        """
        if i < 2 or not self._cons(i) or self._cons(i - 1) or not self._cons(i - 2):
            return False
        ch = self.b[i]
        return ch not in "wxy"

    def _ends(self, s):
        """Check whether the current word ends with the suffix `s`.

        If so, we set `j` to the index immediately before the suffix so later rules can
        rewrite the remaining word stem cleanly.
        """
        length = len(s)
        if s[-1] != self.b[self.k]:
            return False
        if length > (self.k + 1):
            return False
        if self.b[self.k - length + 1:self.k + 1] != s:
            return False
        self.j = self.k - length
        return True

    def _setto(self, s):
        """Replace the suffix ending in `self.j` with a new suffix `s`."""
        length = len(s)
        self.b = self.b[:self.j + 1] + s
        self.k = self.j + length

    def _r(self, s):
        """Apply a replacement only if the measure condition passes."""
        if self._m() > 0:
            self._setto(s)

    def _step1ab(self):
        """Step 1a and 1b: handle common plural and past-tense suffixes.

        This stage removes or rewrites endings like 'sses', 'ies', 'ed', and 'ing' while
        checking whether the root still contains a vowel and whether the word shape is
        appropriate for the replacement.
        """
        if self.b[self.k] == 's':
            if self._ends("sses"):
                self.k -= 2
            elif self._ends("ies"):
                self._setto("i")
            elif self.b[self.k - 1] != 's':
                self.k -= 1
        if self._ends("eed"):
            if self._m() > 0:
                self.k -= 1
        elif (self._ends("ed") or self._ends("ing")) and self._vowel_in_stem():
            self.k = self.j
            if self._ends("at"):
                self._setto("ate")
            elif self._ends("bl"):
                self._setto("ble")
            elif self._ends("iz"):
                self._setto("ize")
            elif self._double_c(self.k):
                if self.b[self.k - 1] not in "lsz":
                    self.k -= 1
            elif self._m() == 1 and self._cvc(self.k):
                self._setto("e")

    def _step1c(self):
        """Step 1c: normalize words ending in y to i when the stem contains a vowel."""
        if self._ends("y") and self._vowel_in_stem():
            self.b = self.b[:self.k] + "i"

    def _step2(self):
        """Step 2: remove common derivational suffixes such as -tional, -ization, -alism."""
        table = {
            "ational": "ate", "tional": "tion", "enci": "ence", "anci": "ance",
            "izer": "ize", "abli": "able", "alli": "al", "entli": "ent",
            "eli": "e", "ousli": "ous", "ization": "ize", "ation": "ate",
            "ator": "ate", "alism": "al", "iveness": "ive", "fulness": "ful",
            "ousness": "ous", "aliti": "al", "iviti": "ive", "biliti": "ble",
        }
        for suf, rep in table.items():
            if self._ends(suf):
                self._r(rep)
                break

    def _step3(self):
        """Step 3: strip suffixes like -icate and -ness when they are safe to remove."""
        table = {
            "icate": "ic", "ative": "", "alize": "al", "iciti": "ic",
            "ical": "ic", "ful": "", "ness": "",
        }
        for suf, rep in table.items():
            if self._ends(suf):
                self._r(rep)
                break

    def _step4(self):
        """Step 4: remove many remaining derivational suffixes if the word is long enough."""
        suffixes = ["al", "ance", "ence", "er", "ic", "able", "ible", "ant",
                    "ement", "ment", "ent", "ou", "ism", "ate", "iti", "ous",
                    "ive", "ize"]
        for suf in suffixes:
            if self._ends(suf):
                if self._m() > 1:
                    self.k = self.j
                return
        if self._ends("ion"):
            if self._m() > 1 and self.j >= 0 and self.b[self.j] in "st":
                self.k = self.j

    def _step5(self):
        """Step 5: clean up trailing e and double-l cases at the end of the word."""
        self.j = self.k
        if self.b[self.k] == 'e':
            a = self._m()
            if a > 1 or (a == 1 and not self._cvc(self.k - 1)):
                self.k -= 1
        if self.b[self.k] == 'l' and self._double_c(self.k) and self._m() > 1:
            self.k -= 1

    def stem(self, word):
        """Stem a single word using the full Porter suffix-stripping pipeline."""
        word = word.lower()
        self.b = word
        self.k = len(word) - 1
        if self.k <= 1:
            return self.b

        # Apply the classical five Porter steps in order.
        self._step1ab()
        if self.k >= 0:
            self._step1c()
            self._step2()
            self._step3()
            self._step4()
            self._step5()
        return self.b[:self.k + 1]


_stemmer = PorterStemmer()


def porter_stem(word):
    """Convenience wrapper for stemming a single token."""
    return _stemmer.stem(word)
