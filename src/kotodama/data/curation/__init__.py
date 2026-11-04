"""Pretraining-corpus curation: from raw sources to a shuffled uint16 token stream.

The stages run in order, each a module with its own command line, and each
reads the previous stage's directory:

1. ``ingest``: read a source through an adapter (``sources``), clean
   (``clean``) and filter (``filters``) each document, and write chunked JSONL
   (``chunks``) with per-reason accounting.
2. ``dedup``: exact or MinHash LSH near-duplicate removal across all sources,
   keeping the copy from the highest-priority source.
3. ``tokenize``: one ``<source>.bin`` per source, each document followed by EOS.
4. ``assemble``: one shuffled stream at ``seq_len`` granularity, with per-source
   drop / weight / cap and a provenance file that labels every chunk with its
   source (``kotodama.data.provenance``).

``audit`` samples a source and reports what the filters would do to it.
``kotodama.data.stream`` inspects, samples and remixes an assembled stream.
"""
