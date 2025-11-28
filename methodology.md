# Modeling Second-Language Learnability Through Sequential Pre-training

## Motivation
Humans acquire languages under differing native language (L1) backgrounds, and evidence shows that typological distance between L1 and L2 (second language) strongly predicts second-language acquisition difficulty for adults. This project will use small-scale sequential pre-training experiments to model how L1 influences L2 learnability in multilingual settings.

## Research Questions
1. RQ1: How does pre-training on a given L1 affect a model’s efficiency and accuracy in acquiring an L2?
2. RQ2: Do typologically closer L1–L2 pairs (e.g., English–German) yield faster adaptation than distant pairs (e.g., Turkish–German)?
3. RQ3: Do model learnability patterns correlate with known human second-language acquisition difficulties?
4. RQ4: How does sequential vs. simultaneous pre-training impact on the model’s performances in acquiring L1 and L2?

## Hypothesis
Pre-training on a morphologically rich or typologically similar L1 will provide inductive biases that accelerate L2 acquisition, whereas structurally distant L1s will slow adaptation.

## Methodology
* Corpora: ~10M tokens per language from open sources (eg. BabyBabelLM subsets).
* Tier 1 languages from BabyBabelLM has 100M-token datasets, so we will samle 10% of it.
* Tier 2 languages already have 10M-token datasets.

### 1 Prioritized Languages (Tier 1 + 2):
1. English (Germanic, SVO) - baseline for distance
2. German (Germanic, Inflectional, V2, Case)
3. Dutch (Germanic, Tier 1)
4. Basque (Tier 2): Latin script, but not in Indo-European family
5. Indonesian (Tier 1): Latin script, but not in Indo-European family

### 2 Pretraining
I will implement a pipeline that can:
* load BabyLM datasets for the given language, and sample 10M-token if needed. 
* pre-train models from scratch, first on L1, then on L2, for a given (L1, L2) pair.
* save the model checkpoint locally, naming the model l1_to_l2
* pre-train monolingual models for a given language L1, and save the model checkpoint, named mono_l1.
* evaluate a given model (checkpoint) on a given language using the MultiBlimp minimal pair evaluation suite. 

### 3 Evaluation Suite
To assess the Formal linguistic competence of the models, I will make use of the MultiBlimP, a Multilingual Benchmark of Linguistic Minimal Pairs. An example script on how to use the MultiBlimp to evaluate one's models is available: [notebook](https://github.com/jumelet/multiblimp/blob/main/lm_eval_example.ipynb).