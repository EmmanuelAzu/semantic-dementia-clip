# Simulating Semantic Dementia Through Iterative Pruning of the CLIP Embedding Space

This repository contains the open-source clinical simulation framework developed for Emmanuel Azubuike's BSc Honours research at the University of the Witwatersrand.

## Abstract
Traditional machine learning treats cognitive decline as discrete, categorical steps. This project bridges neuropsychology and computational clinical modeling by simulating continuous artificial neuro-degeneration. Using the OpenAI CLIP (ViT-B/32) architecture coupled with an exact k-NN retrieval framework, we model progressive cortical atrophy through iterative, layer-targeted weight pruning to evaluate the rate and trajectory of conceptual erosion against a strict 4-tier clinical taxonomy.

## System Architecture
1. **Phase A (Offline Indexing Phase):** Curating taxonomy matrices, processing visual tokens from the THINGS/BOSS subsets, and caching unit-normalized multi-modal embedding matrices.
2. **Atrophy Engine:** Applying global and localized weight pruning masks directly to Transformer attention layers to simulate semantic hub degradation.
3. **Phase B (Online Evaluation):** Generating clinical text/image queries, performing distance metrics, and categorizing retrieval outcomes into clinical error patterns (Coordinate, Superordinate, and Domain Collapse).
