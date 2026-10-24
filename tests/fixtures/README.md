# Synthetic development fixtures

Every document, query and label in this directory is invented solely for software tests.
These are not research data, agricultural recommendations, real citations or evidence
of model quality. The `example.invalid` source URL is a deliberate placeholder.

`make_fixtures.py` defines gold evidence using exact, deliberately authored sentences,
independently of retrieval rankings and chunk IDs. The Hindi query tests Unicode input;
English BM25 does not translate it. Any retrieval of its evidence can be a lexical tie.

Regenerate from the repository root: `python scripts/make_fixtures.py`.
The PDF uses invariant metadata; extraction fingerprints still depend on pypdf versions.
