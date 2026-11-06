# kotodama-tsumugi 1.4B: data recipe

This page describes the pretraining data of [kotodama-tsumugi](https://huggingface.co/LuxiaSL/kotodama-tsumugi-1b): which sources the data came from, how much of each source the model saw, and how the text was processed. It describes the mix only and redistributes none of it. The code that implements each step is in the public [kotodama](https://github.com/anima-research/kotodama) repository.

## 1. Summary

kotodama-tsumugi saw **110.00B tokens** (109,997,719,552). Of these, **109.26B** came from a 384.35B-token corpus of 32 sources (called `train.bin`). The other **0.737B** came from a dialogue supplement (called `rdlg2`), mixed in at 0.67% of sequences. The text was tokenized with the SmolLM2 tokenizer, and each document ends with an end-of-text (EOS) token. The token stream was cut into 4,096-token sequences. The model read a seeded, shuffled 28.4% of the corpus and about 28% of the supplement, and it never read the same sequence twice.

The project excluded known synthetic datasets and assistant chats when it chose the sources, and no model was run over the text to filter or rewrite it. The excluded datasets are OpenCodeReasoning, SmolTalk, SODA, WildChat and OASST2. Two kinds of model-written text are still present:

- **One source turned out to be LLM-written.** `fourchan` (0.06% of the corpus) was rewritten in full by Hermes Llama 3.1 8B before the project downloaded it.
- **Web and forum text contains some model output.** Audits of samples estimate LLM-written content at about 2–4% of finefineweb, 1.5–3% of finefineweb_backfill, 2–4% of finemath, about 1% of doab, 0.5–2% of reddit and 0.5–2% of fandom. Together with fourchan, that is roughly 0.5–1% of all tokens seen. The audits used small samples, and their sizes are not recorded, so the range is wide.

The main corpus has no date cutoff. Most of its volume comes from upstream snapshots frozen before 2025. Eight sources document content from 2025 or later; together they make up about 18.9B of the tokens seen (about 17%). The supplement's Reddit pool is cut at 2022-12-01.

## 2. Sources

### 2.1 Main corpus (`train.bin`)

The corpus holds 384,345,616,384 tokens in 93,834,379 sequences. It was assembled on 2026-05-17, from sources downloaded between 2026-04-20 and 2026-05-17.

Each count covers whole 4,096-token sequences, so it includes the EOS separators. The last column is the *expected* exposure: the model read a random prefix of the shuffled corpus, so each source's realized share matches its corpus share up to sampling noise. That noise is negligible for the large sources and a few percent relative for the smallest.

Revisions: the code loaded each dataset at `main`, with no pinned revision. "logged" means the commit appears in the download logs. "inferred" means the repository had exactly one commit at or before the download date, so the download must have used it.

| # | source | category | upstream @ revision | content dates | tokens in corpus | % of corpus | tsumugi tokens (expected) | notes |
|---|---|---|---|---|---|---|---|---|
| 1 | stack_v1 | code | [bigcode/the-stack](https://huggingface.co/datasets/bigcode/the-stack) @ 349a7135 (logged; the upstream history was later squashed into 153f630) | repositories created 2015-01-01 to 2022-03-31; files fetched Nov 2021 to Jun 2022 | 71,015,059,456 | 18.48% | 20.19B | Permissive licenses only. Auto-generated files, minified files, binary and lock-file extensions, and files that match patterns for secret keys dropped. 100–100,000 characters. At most 1M files per language (seeded random sample). 26.9% removed as duplicates. |
| 2 | finefineweb | web | [m-a-p/FineFineWeb](https://huggingface.co/datasets/m-a-p/FineFineWeb) @ 7fd92dc8 (inferred) | Common Crawl via FineWeb; repository frozen 2024-12-19 | 65,904,111,616 | 17.15% | 18.74B | 936 files across 64 topics. At least 200 characters. Audit: about 2–4% pages from LLM content mills. |
| 3 | pesa2o | science | [allenai/peS2o](https://huggingface.co/datasets/allenai/peS2o) v2 @ 636a503e (logged) | knowledge cutoff 2023-01-03 | 60,719,550,464 | 15.80% | 17.26B | Academic papers. Not deduplicated (a dry run found no overlap with other sources). |
| 4 | us_patents | patents | [allenai/us-patents](https://huggingface.co/datasets/allenai/us-patents) @ 19aea30f (inferred) | patent grants and applications, 1976–2025 | 36,728,127,488 | 9.56% | 10.44B | Title, abstract and description joined into one document. At least 200 characters. |
| 5 | pile_of_law | law | [pile-of-law/pile-of-law](https://huggingface.co/datasets/pile-of-law/pile-of-law) @ 0dc9f2c2 (inferred) | subsets run to 2022 | 17,943,523,328 | 4.67% | 5.10B | Court opinions and congressional hearings subsets only. At least 300 characters. |
| 6 | pre_1929_books | books | [common-pile/pre_1929_books](https://huggingface.co/datasets/common-pile/pre_1929_books) @ a158135b (inferred) | US books published before 1929 | 15,749,390,336 | 4.10% | 4.48B | At least 500 characters. OCR filter: a document is dropped when more than 5% of its characters are replacement or control characters. |
| 7 | stackexchange | forums (Q&A) | [HuggingFaceTB/stackexchange_2025_md](https://huggingface.co/datasets/HuggingFaceTB/stackexchange_2025_md) @ fe382c48 (inferred) | March 2025 dump | 15,612,219,392 | 4.06% | 4.44B | 22 sites; whole threads with score at least 1. At least 300 characters. |
| 8 | openwebmath | math | [open-web-math/open-web-math](https://huggingface.co/datasets/open-web-math/open-web-math) @ fde8ef8d (inferred) | Common Crawl pages; repository frozen 2023-10-17 | 13,439,942,656 | 3.50% | 3.82B | |
| 9 | library_of_congress | books | [common-pile/library_of_congress](https://huggingface.co/datasets/common-pile/library_of_congress) @ efa378ea (inferred) | historical digitized books (window not stated) | 13,317,722,112 | 3.47% | 3.79B | At least 500 characters; same OCR filter as row 6. |
| 10 | reddit | forums | [mgrtsv/reddit_dataset_46](https://huggingface.co/datasets/mgrtsv/reddit_dataset_46) @ 3a5f532e (logged) | 2008-11-10 to 2024-12-29; about 86% from 2024 | 9,119,948,800 | 2.37% | 2.59B | At least 50 characters. 34.7% removed as duplicates. Audit: about 0.5–2% chatbot answers. |
| 11 | finefineweb_backfill | web | [m-a-p/FineFineWeb](https://huggingface.co/datasets/m-a-p/FineFineWeb) @ 7fd92dc8 (inferred) | as row 2 | 8,387,244,032 | 2.18% | 2.38B | 120 more files across 14 topics, added to fill topic gaps. Audit: about 1.5–3% LLM content. |
| 12 | usgpo | government | [common-pile/usgpo](https://huggingface.co/datasets/common-pile/usgpo) @ 0b314ed3 (logged) | not recorded (repository's last commit 2025-06-06) | 8,212,934,656 | 2.14% | 2.33B | 38.6% removed as duplicates (document revisions). |
| 13 | pg19 | books | [common-pile/project_gutenberg](https://huggingface.co/datasets/common-pile/project_gutenberg) @ 01dc90a5 (inferred) | public-domain books | 7,270,502,400 | 1.89% | 2.07B | Named for PG-19, but taken from the larger Common Pile Gutenberg set. Gutenberg headers, footers and license blocks removed. At least 500 characters. |
| 14 | finemath | math | [HuggingFaceTB/finemath](https://huggingface.co/datasets/HuggingFaceTB/finemath) finemath-4plus @ e92b25a6 (inferred) | Common Crawl; repository frozen 2025-02-06 | 6,856,441,856 | 1.78% | 1.95B | 36.4% removed as duplicates (overlap with openwebmath). Audit: about 2–4% LLM content. |
| 15 | nntp | forums (Usenet) | [nyuuzyou/nntp-text-387m](https://huggingface.co/datasets/nyuuzyou/nntp-text-387m) @ 406439e9 (logged) | 2002 to about January 2026 | 5,862,473,728 | 1.53% | 1.67B | Non-English newsgroup hierarchies dropped. A message is dropped when more than 60% of its lines are quotes, when URLs make up more than 15% of its characters, or when it contains a long run of one repeated character. 150–50,000 characters. |
| 16 | wikipedia | reference | [wikimedia/wikipedia](https://huggingface.co/datasets/wikimedia/wikipedia) 20231101.en @ b04c8d1c (inferred) | dump of 2023-11-01 | 5,006,184,448 | 1.30% | 1.42B | |
| 17 | youtube | transcripts | [common-pile/youtube_filtered](https://huggingface.co/datasets/common-pile/youtube_filtered) @ dff8c8a5 (inferred) | not recorded (repository created 2025-05-23) | 4,125,872,128 | 1.07% | 1.17B | Whisper speech-recognition transcripts of CC-BY channels. Audit: 3–5% machine translation. |
| 18 | fandom | reference (fan wikis) | [nyuuzyou/fandom](https://huggingface.co/datasets/nyuuzyou/fandom) @ dec69c62 (logged) | pages as of 2025-02-18 | 3,548,844,032 | 0.92% | 1.01B | At least 200 characters. Audit: about 0.5–2% model-written pages. |
| 19 | doab | books | [common-pile/doab](https://huggingface.co/datasets/common-pile/doab) @ 89e7a359 (inferred) | not recorded (an audit saw 2023–24 prefaces) | 3,046,154,240 | 0.79% | 0.87B | Open-access books. At least 500 characters; same OCR filter as row 6. Audit: about 1% LLM content. |
| 20 | hacker_news | forums | [jkeisling/hacker-news-corpus-2007-2022](https://huggingface.co/datasets/jkeisling/hacker-news-corpus-2007-2022) @ fbdad574 (inferred) | 2007 to 2022-11-16 | 3,043,090,432 | 0.79% | 0.87B | At least 50 characters. |
| 21 | opensubtitles | dialogue (subtitles) | [OPUS OpenSubtitles v2018](https://object.pouta.csc.fi/OPUS-OpenSubtitles/v2018/mono/en.txt.gz), English monolingual (not on Hugging Face) | v2018 release | 2,379,501,568 | 0.62% | 0.68B | Markup, music symbols, bracketed annotations and credit lines removed. The upstream text has no film boundaries, so it was cut into 200-line blocks of at least 500 characters. 38.5% removed as duplicates. |
| 22 | uk_hansard | government (parliament) | [common-pile/uk_hansard](https://huggingface.co/datasets/common-pile/uk_hansard) @ 05eeb43c (inferred) | Commons from 1918 and Lords from 1999, to about 2025 | 2,223,984,640 | 0.58% | 0.63B | |
| 23 | fimfiction | fiction | [trentmkelly/fimfiction-scrape](https://huggingface.co/datasets/trentmkelly/fimfiction-scrape) @ bd06050b (inferred) | scraped June 2025; no per-story dates | 2,122,567,680 | 0.55% | 0.60B | 90,451 stories. An audit rated it clean, but it may hold some AI-written stories from 2023–25; none were counted. |
| 24 | regulations | government (regulation) | [common-pile/regulations](https://huggingface.co/datasets/common-pile/regulations) @ d1a63eef (inferred) | 2000–2023 (per the source label; the upstream card is silent) | 1,522,905,088 | 0.40% | 0.43B | 34.8% removed as duplicates (boilerplate). |
| 25 | discord | dialogue (chat) | [mookiezi/Discord-Dialogues](https://huggingface.co/datasets/mookiezi/Discord-Dialogues) @ a8b2294b (inferred) | anonymized Discord conversations, late spring to early fall 2025 | 323,821,568 | 0.08% | 0.09B | Apache-2.0. At least 30 characters. Its upstream text carries chat-template markup: 10.4% of its tokens are the template's special tokens. |
| 26 | fourchan | forums (imageboard) | [adamo1139/4chan_archive_ShareGPT_fixed_newlines_unfiltered](https://huggingface.co/datasets/adamo1139/4chan_archive_ShareGPT_fixed_newlines_unfiltered) @ c936abde (inferred) | scraped 2024 | 226,299,904 | 0.06% | 0.06B | **Rewritten in full by Hermes Llama 3.1 8B upstream.** Two or more turns; at least 100 characters. |
| 27 | *private source (social media)* | private conversational | not released (see §4) | not recorded | 224,518,144 | 0.06% | 0.06B | Added after deduplication across sources. |
| 28 | writingprompts | fiction | [euclaise/writingprompts](https://huggingface.co/datasets/euclaise/writingprompts) @ 35f0aa35 (inferred) | r/WritingPrompts | 207,433,728 | 0.05% | 0.06B | Formatted as `[WP] prompt`, a blank line, then the story; stories of at least 200 characters. |
| 29 | enron | email | [corbt/enron-emails](https://huggingface.co/datasets/corbt/enron-emails) @ cfc06c75 (inferred) | Enron corpus, about 1998–2002 (the card is silent) | 97,538,048 | 0.03% | 0.03B | Message bodies. 63.2% removed as duplicates (quoted threads). |
| 30 | screenplays | dialogue (scripts) | [IsmaelMousa/movies](https://huggingface.co/datasets/IsmaelMousa/movies) @ 67654dcf (inferred) | not recorded per script | 45,477,888 | 0.01% | 0.01B | About 1.2K film scripts. |
| 31 | gutenberg_poetry | books (poetry) | [biglam/gutenberg-poetry-corpus](https://huggingface.co/datasets/biglam/gutenberg-poetry-corpus) @ fcd42e24 (inferred) | public-domain poetry | 33,579,008 | 0.01% | 0.01B | Lines grouped back into one document per book. At least 200 characters. |
| 32 | philosophy | reference | [davidheineman/plato](https://huggingface.co/datasets/davidheineman/plato) @ 84500b01 (inferred) | Stanford Encyclopedia of Philosophy, pulled August 2025 | 28,651,520 | 0.01% | 0.01B | At least 500 characters. |
| | **total** | | | | **384,345,616,384** | **100%** | **109.26B** | 99.33% of the tokens seen |

Categories, as shares of the corpus: web 19.33%; code 18.48%; science 15.80%; books 10.26%; patents 9.56%; forums 8.81%; law and government 7.78%; math 5.28%; reference 2.23%; dialogue and transcripts 1.78%; fiction 0.61%; private conversational 0.06%; email 0.03%. Multi-turn dialogue between a few speakers (subtitles, transcripts, scripts and chat) makes up about 1.8%. The forums (Reddit, Hacker News, Usenet, Stack Exchange, 4chan) are threaded discussion.

### 2.2 Dialogue supplement (`rdlg2`)

The supplement holds about 2.600B tokens of conversational text, built on 2026-10-19 from five pools. It was mixed into the whole run at 0.67% of sequences, and the model read about 28% of it. It is a separate token stream and is not part of `train.bin`. Its pool shares come from its build record; the Reddit and 4chan shares are approximate, and there are no exact per-pool token counts.

| pool | category | upstream | content dates | share of supplement | tsumugi tokens (expected) | notes |
|---|---|---|---|---|---|---|
| Reddit | forums | Pushshift Reddit dumps on archive.org ([pushshift_reddit_200506_to_202212](https://archive.org/details/pushshift_reddit_200506_to_202212)): 416 monthly dumps | 2005-06 to 2022-11 (cut at 2022-12-01 00:00 UTC) | about 70% | about 0.52B | 680 subreddits, selected by an English test and by how often "you" refers to someone in the thread. See §3.6. |
| 4chan | forums (imageboard) | 4chan threads from a 2015 public archive dump (8 work-safe boards) | 2008–2015 | about 10% | about 0.07B | Original human threads, unlike the rewritten `fourchan` source above. |
| Usenet, multi-speaker | forums (Usenet) | subset of the corpus's `nntp` source | 2002–2026, no date cut | 10.2% | about 0.075B | Only documents with at least two speakers. |
| screenplays, multi-speaker | dialogue (scripts) | subset of the corpus's `screenplays` source | as row 30 | 6.95% | about 0.05B | Only documents with at least two speakers; the pool is repeated 4 times. |
| *private pool (social media)* | private conversational | not released (see §4) | through 2026-10 | 2.85% | about 0.021B | Multi-author threads. Documents that mention language models are dropped; the pool is repeated 6 times. |
| **total** | | | | **100%** | **0.737B** | 0.67% of the tokens seen |

## 3. Processing

### 3.1 Source selection

Sources were chosen whole. Known synthetic datasets and assistant chats were left out of the corpus: OpenCodeReasoning, SmolTalk, SODA, WildChat, OASST2 and Stack-Edu (Stack-Edu was metadata only). No document-level LLM detector and no date filter ran on the main corpus.

### 3.2 Per-source formatting and filters

Each source has its own reader, which applies the formatting and filters in the "notes" column of §2.1.

- **Code:** license filter, generated-file and minified-file filters, secret-pattern filter, length bounds, and a per-language cap.
- **Usenet:** language, quote-ratio and spam filters.
- **Gutenberg books:** header, footer and license stripping.
- **Subtitles:** markup removal and fixed-size blocks.
- **Stack Exchange:** site list and score threshold.

The corpus was built with an earlier version of this pipeline. The public [`sources.py`](https://github.com/anima-research/kotodama/blob/main/src/kotodama/data/curation/sources.py) presets reproduce the settings for stackexchange, us_patents, writingprompts, gutenberg_poetry, wikipedia and finemath. The readers for the other sources are not in the public repository.

### 3.3 Universal cleaning

Every document passed through one cleaning function, in this order:

1. NFC Unicode normalization, and removal of control characters other than newline and tab.
2. Whitespace normalization: runs of spaces and tabs become one space, and at most one blank line is kept.
3. A length filter, with a default of at least 100 characters. Sources with a different floor list it in §2.1.
4. PII scrubbing. E-mail addresses, US phone numbers, US Social Security numbers and IPv4 addresses were replaced with the placeholders `<EMAIL>`, `<PHONE>`, `<SSN>` and `<IP>`. Code, Gutenberg books and subtitles were not scrubbed.

Implementation: [`clean.py`](https://github.com/anima-research/kotodama/blob/main/src/kotodama/data/curation/clean.py). The public version applies the same steps but writes lowercase placeholders (`[email]` and so on).

The public [`filters.py`](https://github.com/anima-research/kotodama/blob/main/src/kotodama/data/curation/filters.py) holds a battery of quality filters, which was **not** applied to this corpus. The battery checks markup, repetition, script and language. It was written after this corpus was built.

### 3.4 Deduplication

Near-duplicates were removed with MinHash locality-sensitive hashing across all sources at once:

- **Signatures.** Each document was lower-cased and split into word 5-gram shingles, which were hashed with xxh32. The signature keeps the minimum of each of 112 seeded hash functions (seed 42).
- **Matching.** The signature was cut into 14 bands of 8 rows. Two documents were joined when any band matched, which catches pairs above about 0.7 Jaccard similarity with high probability. Joined documents formed clusters.
- **Survivors.** In each cluster, the copy from the highest-priority source survived:

| priority | sources |
|---|---|
| 10 | reference and books: wikipedia, pesa2o, openwebmath, pg19, pre_1929_books, library_of_congress, doab, pile_of_law, enron, philosophy, gutenberg_poetry, screenplays |
| 9 | finemath, usgpo, regulations, uk_hansard, us_patents |
| 8 | stack_v1, stackexchange, youtube, opensubtitles |
| 7 | fandom, fimfiction, writingprompts |
| 6 | reddit, hacker_news, nntp |
| 5 | discord, fourchan |
| 3 | finefineweb |

The pass removed 19% of documents: 78.4M of 412M. peS2o was not deduplicated. The private social-media source was added after the pass. Implementation: [`dedup.py`](https://github.com/anima-research/kotodama/blob/main/src/kotodama/data/curation/dedup.py), with the same shingle size, band layout and seed; the public version takes the priorities as `--priority` flags. No decontamination against benchmarks was applied.

### 3.5 Tokenization, packing and assembly

1. **Tokenization.** Each source was tokenized with the SmolLM2 tokenizer ([`HuggingFaceTB/SmolLM2-135M`](https://huggingface.co/HuggingFaceTB/SmolLM2-135M), vocabulary 49,152; [`presets.py`](https://github.com/anima-research/kotodama/blob/main/src/kotodama/presets.py)), and each document was followed by EOS (id 0). No other special tokens were added. In the main corpus, any literal special-token text in a document was encoded as the matching special token. Implementation: [`tokenize.py`](https://github.com/anima-research/kotodama/blob/main/src/kotodama/data/curation/tokenize.py).
2. **Packing.** Each source's token stream was cut into consecutive 4,096-token sequences, and the last partial sequence was dropped. Documents are therefore packed back to back, and each sequence comes from a single source. A document longer than 4,096 tokens spans several sequences.
3. **Shuffle.** All sequences, with sources in name order, were permuted with `numpy.random.RandomState(42).permutation` and written as one file. A provenance file records each source's sequence count and the seed, which is enough to recover the source of every sequence. Implementation: [`assemble.py`](https://github.com/anima-research/kotodama/blob/main/src/kotodama/data/curation/assemble.py) and [`provenance.py`](https://github.com/anima-research/kotodama/blob/main/src/kotodama/data/provenance.py).

### 3.6 The supplement

The supplement was built by separate scripts, which are not in the public repository. Its rules:

- **Reddit.** Subreddits were kept when they passed an English test (at least 97% ASCII letters and at least 180 stopwords per 1,000 words). They also needed at least 8 instances of "you" per 1,000 tokens whose referent can be resolved in the thread. A short list of hate and harassment communities was excluded. No subreddit could exceed 1.5% of the pool's tokens; a larger one kept a fraction of its threads, chosen by a deterministic hash of the thread id. Posts and comments from 2022-12-01 00:00 UTC onward were dropped. Comments by bot accounts were dropped: accounts on a fixed list, or with names ending in "bot" or "_bot". One kept subreddit is built around GPT-2 bot accounts. The bot rule probably removes their turns, but this is unverified; the exposure is about 0.4M tokens at most.
- **4chan.** One document per thread, rendered in the imageboard's own format. Threads with fewer than 3 posts were dropped, and so were deleted posts. Poster IP addresses, e-mail fields and delete passwords were never read into the output.
- **Multi-speaker pools.** A document was kept only if at least two distinct speakers each label at least two lines.
- **Private pool.** Any document that mentions a language model, an AI assistant or a model vendor by name was dropped.
- **Tokenization.** SmolLM2, with each document followed by EOS. Literal special-token text was encoded as plain text, never as a special token.

### 3.7 Sampling during training

- **Partitions.** The corpus was split into 8 equal, contiguous partitions, one per GPU (data-parallel rank).
- **Order.** Each rank shuffled its own partition with a seeded generator (data seed 42). The generator is salted, so its order differs from the assembly shuffle.
- **Coverage.** The model read the first 28.4% of each rank's shuffled order. The 100B-token trunk and the 10B-token anneal read one continuous stream, so no sequence was read twice.
- **Mixing.** The supplement was interleaved by a rule that tracks its share: before each sequence, a rank takes the supplement's next sequence whenever the supplement's share so far is below 0.67%. The mix holds that fraction exactly, to within one sequence, over the whole run.
- **Masking.** Attention was masked at document boundaries, and position ids restart at each EOS, so a document never attends to its neighbours in the same sequence.

Two caveats apply to "never read twice". The supplement repeats two small pools (screenplays 4 times, the private pool 6 times). Its Usenet and screenplays pools are also subsets of main-corpus sources. Some of those documents can therefore recur.

Implementation: `TokenizedDataset` and `MixedTokenDataset` in [`dataset.py`](https://github.com/anima-research/kotodama/blob/main/src/kotodama/data/dataset.py); `--mix_fraction` and `--data_seed` in [`train.py`](https://github.com/anima-research/kotodama/blob/main/src/kotodama/training/train.py); the run's settings in [`configs/tsumugi-1b.yaml`](https://github.com/anima-research/kotodama/blob/main/configs/tsumugi-1b.yaml), which extends [`configs/ladder-a0.yaml`](https://github.com/anima-research/kotodama/blob/main/configs/ladder-a0.yaml) (`doc_masking: true`, `sequence_length: 4096`).

## 4. Not released; described by category only

The training token streams are not published. Two components are described here only by category, because they hold conversations from private communities. They are not released, and their upstream names are withheld.

| component | category | where | tokens | share | tsumugi tokens (expected) |
|---|---|---|---|---|---|
| private source (social media) | private conversational (social media) | main corpus, row 27 | 224,518,144 | 0.06% of the corpus | 0.06B |
| private pool (social media) | private conversational (social media) | supplement | not recorded (2.85% of the supplement) | 2.85% of the supplement | about 0.021B |

Together they make up about 0.085B of the 110B tokens seen, under 0.1%. The private main-corpus source is excluded from new mixes, and so is `discord`, for its chat-template markup.

The cleaned, deduplicated text of 19 of the 32 main-corpus sources is published as [aethera-gp/kotodama-3b-corpus-open](https://huggingface.co/datasets/aethera-gp/kotodama-3b-corpus-open).
