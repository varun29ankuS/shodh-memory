#!/usr/bin/env python3
"""Build `src/taxonomy/hypernyms.tsv` from WordNet.

The graph can only relate entities the corpus mentions together. It cannot know
that a turtle is an animal, because nobody in the corpus ever says so — that is
world knowledge, the same gap `src/kb` fills for identity. This script fills it
for *category*.

Output format, one line per noun lemma:

    lemma<TAB>ancestor1|ancestor2|...

Ancestors are ordered nearest-first, so a consumer can weight by distance.

Sense selection
---------------
Word-sense ambiguity is the whole difficulty. Two obvious rules both fail:

* **All senses.** Produces junk edges — `share` reaches `way` through the
  "portion" sense, `work` reaches `check`. Measured on the LoCoMo gate, a naive
  all-senses walk produced 2 bridges of which 1 was spurious.
* **First sense.** WordNet orders `turtle` with *turtleneck sweater* first, so
  the single case this layer exists to serve resolves to `garment` and misses
  `animal` entirely.

The rule used here: **prefer the synset whose own name is the lemma.**
`turtleneck.n.01` merely lists "turtle" as an alias, while `turtle.n.02` is
*named* turtle — the latter is the sense a speaker means when they say the bare
word. Same for `dawn.n.01` vs `sunrise.n.02` and `puppy.n.02` vs `pup.n.01`.
Where no synset is so named, fall back to sense 1. This is deterministic, needs
no model, and is explainable to anyone reading an edge.

Very abstract roots are dropped: an `IsA` edge to `entity` or `abstraction`
relates everything to everything and is pure fan-out.

Usage:
    pip install nltk && python -c "import nltk; nltk.download('wordnet')"
    python scripts/build_taxonomy.py
"""

import io
import os
import sys

try:
    from nltk.corpus import wordnet as wn
except ImportError:  # pragma: no cover - developer tooling
    sys.exit("nltk is required: pip install nltk && python -m nltk.downloader wordnet")

# Ancestors so general that an edge to them carries no discriminative signal.
# `turtle IsA entity` is true and useless; it links the turtle to every noun.
ABSTRACT_ROOTS = {
    "entity",
    "abstraction",
    "physical_entity",
    "object",
    "whole",
    "psychological_feature",
    "attribute",
    "relation",
    "group",
    "grouping",
    "measure",
    "matter",
    "part",
    "unit",
    "causal_agent",
    "thing",
}

# Hypernym chains are shallow in practice; this only bounds pathological cases.
MAX_DEPTH = 8

# Multi-word ancestors ("young_mammal") are kept — a consumer that only wants
# single tokens can filter — but multi-word *lemmas* are dropped, since the
# entity surfaces we look up are single tokens.
OUT = os.path.join(os.path.dirname(__file__), "..", "src", "taxonomy", "hypernyms.tsv")


def preferred_synset(lemma):
    """The sense a speaker means by the bare word. See module docstring."""
    senses = wn.synsets(lemma, pos=wn.NOUN)
    if not senses:
        return None
    named = [s for s in senses if s.name().split(".")[0] == lemma]
    return (named or senses)[0]


def ancestors(lemma):
    """Hypernym closure of the preferred sense, nearest first, roots dropped."""
    synset = preferred_synset(lemma)
    if synset is None:
        return []
    out, seen, frontier = [], set(), [synset]
    for _ in range(MAX_DEPTH):
        nxt = []
        for node in frontier:
            for hypernym in node.hypernyms():
                name = hypernym.name().split(".")[0]
                if name in seen:
                    continue
                seen.add(name)
                nxt.append(hypernym)
                if name not in ABSTRACT_ROOTS:
                    out.append(name)
        if not nxt:
            break
        frontier = nxt
    return out


def main():
    lemmas = set()
    for synset in wn.all_synsets(pos=wn.NOUN):
        for lemma in synset.lemmas():
            name = lemma.name().lower()
            if "_" not in name and name.isalpha() and len(name) > 2:
                lemmas.add(name)

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    written = 0
    with io.open(OUT, "w", encoding="utf-8", newline="\n") as fh:
        for lemma in sorted(lemmas):
            anc = ancestors(lemma)
            if not anc:
                continue
            fh.write("{}\t{}\n".format(lemma, "|".join(anc)))
            written += 1

    size = os.path.getsize(OUT)
    print("lemmas considered : {}".format(len(lemmas)))
    print("rows written      : {}".format(written))
    print("bytes             : {} ({:.1f} MB)".format(size, size / 1e6))


if __name__ == "__main__":
    main()
