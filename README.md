# ai-engineer-roadmap

A path through becoming an AI engineer: programming and math first, then classical ML, deep learning, and the current LLM stack, agents, and RAG. Pick a phase below and start reading.

## Contents

- [Phase 1: Prerequisites](#phase-1-prerequisites)
- [Phase 2: Classical ML](#phase-2-classical-ml)
- [Phase 3: Deep Learning](#phase-3-deep-learning)
- [Phase 4: NLP](#phase-4-nlp)
- [Phase 5: Computer Vision](#phase-5-computer-vision)
- [Phase 6: LLMs](#phase-6-llms)
- [Phase 7: Agents](#phase-7-agents)
- [Phase 8: RAG](#phase-8-rag)
- [Phase 9: Fine-tuning](#phase-9-fine-tuning)
- [Phase 10: MLOps & Production](#phase-10-mlops--production)
- [Phase 11: Staying current](#phase-11-staying-current)
- [Tracks](#tracks)
- [Contributing](#contributing)
- [License](#license)

---

## Phase 1: Prerequisites

Install packages, use git, do linear algebra, run bash. If you already program, move quickly through this phase; otherwise these are the basics everyone needs.

### Python

The one language modern AI runs on. Get to the point where you can read someone else's PyTorch code and not get lost.

- **[Automate the Boring Stuff with Python](https://automatetheboringstuff.com/)** (Sweigart): free, book. Takes people who have never programmed to a productive level in Python.
- **[Real Python: Python Basics learning path](https://realpython.com/learning-paths/python-basics/)**: free, articles + exercises. Good if you learn better in small pieces than a whole book.
- **[Fluent Python, 2nd ed.](https://www.oreilly.com/library/view/fluent-python-2nd/9781492056348/)** (Ramalho): $, book, 1000 pages. `[underrated]` The one book that will change how you write Python. Come back to it after a year of Python under your belt; skip on the first pass.

Prereqs: none.

### Math

Enough linear algebra, calculus, and probability to read papers and understand what a training loop is doing.

- **[Mathematics for Machine Learning](https://mml-book.github.io/)** (Deisenroth, Faisal, Ong): free, book. The canonical "math you need for ML, without going full pure-math," free PDF from the authors.
- **[Essence of Linear Algebra](https://www.3blue1brown.com/topics/linear-algebra)** (3Blue1Brown): free, YouTube series, ~5 hours. `[underrated as a first pass]` Builds intuition for what vectors, matrices, and eigenvectors actually are before you drown in notation.
- **[Introduction to Probability](https://www.edx.org/learn/probability/harvard-university-introduction-to-probability)** (Blitzstein, Harvard, Stat 110): free to audit, course. `[underrated]` The probability foundation almost nobody has and everyone should. Come back after ML basics if the full course is too much upfront.

Prereqs: [Python](#python) helps for the ML-flavored examples.

### Git

Version control. Non-negotiable.

- **[Pro Git](https://git-scm.com/book/en/v2)** (Chacon, Straub): free, book. The canonical reference. Read chapters 1-3; treat the rest as a reference to return to.
- **[Oh Shit, Git!?!](https://ohshitgit.com/)** (Julia Evans-flavored): free, cheat sheet. `[underrated]` The one page you reach for when you've mangled a rebase and don't know what to do.

Prereqs: [CLI / Linux](#cli--linux).

### CLI / Linux

Command line + basic shell. If `cd`, `ls`, `grep`, `chmod`, `|`, `>` aren't second nature, start here.

- **[The Missing Semester of Your CS Education](https://missing.csail.mit.edu/)** (MIT): free, lecture series, ~12 hours. `[underrated as a starting point]` Everything CS programs assume you already know: shell, git, editors, debugging, remote machines. Watch lectures 1-4 minimum.
- **[The Linux Command Line](https://linuxcommand.org/tlcl.php)** (Shotts): free, book. Systematic reference if you want a real book.

Prereqs: none.

---

## Phase 2: Classical ML

Build + evaluate a scikit-learn classifier end to end. Modern LLM work still rests on the concepts here (loss functions, regularization, evaluation metrics). Don't skip.

### Supervised learning

Regression + classification: predicting labels from features. Learn the vocabulary here, it applies everywhere else.

- **[Machine Learning Specialization](https://www.coursera.org/specializations/machine-learning-introduction)** (Andrew Ng, Coursera): free to audit, course. The canonical entry point for supervised learning.
- **[Hands-On Machine Learning with Scikit-Learn, Keras & TensorFlow, 3rd ed.](https://www.oreilly.com/library/view/hands-on-machine-learning/9781098125967/)** (Géron): $, book. Best practitioner text; chapters 1-6 cover supervised learning end to end with real code.
- **[StatQuest with Josh Starmer](https://www.youtube.com/@statquest)**: free, YouTube channel. `[underrated]` Visual explanations of every classical ML concept. One of the clearest resources for the "why" behind linear regression, logistic regression, decision trees.

Prereqs: [Math](#math), [Python](#python).

### Unsupervised learning

Finding structure without labels: clustering, dimensionality reduction.

- **[An Introduction to Statistical Learning, 2nd ed.](https://www.statlearning.com/)** (James, Witten, Hastie, Tibshirani): free, book, Python and R editions available. The definitive ML book grounded in statistics. Chapters 10 and 12 for unsupervised.
- **[StatQuest: Clustering and PCA playlists](https://www.youtube.com/@statquest/playlists)**: free, YouTube. `[underrated]` If ISLR feels too dense, watch these first, then read.
- **[scikit-learn: clustering user guide](https://scikit-learn.org/stable/modules/clustering.html)**: free, official docs. Reach for this when you're about to `pip install` a clustering algorithm and want to know which one to pick.

Prereqs: [Supervised learning](#supervised-learning).

### Model evaluation

Metrics, cross-validation, train/val/test splits. The most-skipped step and the one that separates working models from broken ones.

- **[scikit-learn: model evaluation user guide](https://scikit-learn.org/stable/modules/model_evaluation.html)**: free, official docs. Canonical reference for every metric name and when to use it.
- **[Google's Machine Learning Crash Course: Classification](https://developers.google.com/machine-learning/crash-course/classification/video-lecture)**: free, course section. Best intro to precision/recall/AUC and why accuracy alone lies.
- **[Designing Machine Learning Systems](https://www.oreilly.com/library/view/designing-machine-learning/9781098107956/)** (Huyen) chapter 6: $, book. `[underrated]` The one book that treats evaluation as a production problem, not a homework problem.

Prereqs: [Supervised learning](#supervised-learning).

### Feature engineering

Turning raw data into what the model actually needs. Modern deep-learning talks less about this, but real production pipelines are half feature engineering.

- **[Feature Engineering for Machine Learning](https://www.oreilly.com/library/view/feature-engineering-for/9781491953235/)** (Zheng, Casari): $, book. Canonical practitioner text.
- **[Kaggle Learn: Feature Engineering](https://www.kaggle.com/learn/feature-engineering)**: free, short course. Fastest way to see the techniques applied to real tabular data.
- **[Applied Predictive Modeling](http://appliedpredictivemodeling.com/)** (Kuhn, Johnson): $, book. `[underrated by modern ML crowd]` Older and written in R, but the reasoning about feature construction still beats most newer books.

Prereqs: [Supervised learning](#supervised-learning).

### scikit-learn

The library everything else in classical ML uses. Get comfortable with the API; you'll use it in fine-tuning data prep, evaluation, and prototyping.

- **[scikit-learn: user guide](https://scikit-learn.org/stable/user_guide.html)**: free, official docs. Better than most tutorials. Start with "Getting started".
- **[Hands-On ML](https://www.oreilly.com/library/view/hands-on-machine-learning/9781098125967/)** (Géron) chapters 3-8: $, book. Best practical introduction to sklearn, written so you can run the code as you read.
- **[Machine Learning in Python with scikit-learn](https://www.fun-mooc.fr/en/courses/machine-learning-python-scikit-learn/)** (Inria, scikit-learn core team): free, MOOC. `[underrated]` Taught by the people who wrote the library.

Prereqs: [Supervised learning](#supervised-learning).

---

## Phase 3: Deep Learning

Train a CNN from scratch on a small dataset. This is where PyTorch enters and the rest of the roadmap depends on it.

### Neural network basics

Backpropagation, activation functions, loss functions. Understand these before you touch PyTorch, or you'll be copying code without understanding what it does.

- **[Deep Learning Specialization](https://www.coursera.org/specializations/deep-learning)** (Andrew Ng, DeepLearning.AI): free to audit, course. Canonical intro to DL; courses 1 and 2 for the fundamentals.
- **[3Blue1Brown: Neural Networks playlist](https://www.3blue1brown.com/topics/neural-networks)**: free, YouTube, ~1 hour. Best intuition for what backprop is actually doing before you write a line of code.
- **[Neural Networks: Zero to Hero](https://karpathy.ai/zero-to-hero.html)** (Karpathy): free, YouTube series, ~10 hours. `[underrated the whole series]` Builds autograd from scratch, then a transformer. If you only watch one thing in this phase, watch the first video.

Prereqs: [Math](#math), [Python](#python).

### PyTorch

The deep-learning framework most modern research and a lot of production runs on. Learn it after neural-network basics; you'll use it through the rest of this roadmap.

- **[Deep Learning with PyTorch, 2nd ed.](https://www.manning.com/books/deep-learning-with-pytorch-second-edition)** (Antiga, Stevens, Huang, Viehmann): $, book. Canonical textbook, written by PyTorch core devs and updated for transformers and generative models.
- **[PyTorch: 60 Minute Blitz](https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html)**: free, tutorial, ~2 hours. Fastest way to get a real training loop running.
- **[Zero to Hero: makemore series](https://karpathy.ai/zero-to-hero.html)** (Karpathy): free, YouTube. `[underrated]` Continuation of the neural-network-basics pick above, now in PyTorch. Best "click" moment for autograd.
- **[fast.ai Practical Deep Learning](https://course.fast.ai/)**: free, course, ~22 hours. Alternative starting point if you'd rather start from working projects than from theory.
- **[PyTorch Discuss](https://discuss.pytorch.org/)**: free, forum. Searchable, core devs answer, more useful than Stack Overflow for PyTorch-specific questions.

Prereqs: [Neural network basics](#neural-network-basics), [Python](#python).

### Training loops

The `for epoch in range(...)` machinery: batches, optimizers, learning-rate schedules, checkpointing.

- **[PyTorch: Training a Classifier tutorial](https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html)**: free, tutorial. The one you'll copy-paste the first ten times you write a training loop.
- **[fast.ai: lessons 1-4](https://course.fast.ai/)**: free, video lessons. Cover the training loop from a practitioner angle.
- **[A Recipe for Training Neural Networks](https://karpathy.github.io/2019/04/25/recipe/)** (Karpathy): free, blog post. `[underrated]` The one page every practitioner rediscovers eventually. Read it before your first "why isn't this model learning" bug.

Prereqs: [PyTorch](#pytorch).

### Regularization

Dropout, weight decay, batch normalization, early stopping. Why your model works on training data and fails on validation.

- **[Deep Learning](https://www.deeplearningbook.org/)** (Goodfellow, Bengio, Courville) chapter 7: free, book. The academic reference.
- **[Sebastian Raschka's blog: regularization posts](https://sebastianraschka.com/blog/)**: free, blog. `[underrated]` Concrete visual walkthroughs of dropout, batch norm, layer norm, better than most textbook treatments.
- **[Deep Learning Specialization, course 2](https://www.coursera.org/learn/deep-neural-network)** (Ng): free to audit. Section on regularization is clear and focused.

Prereqs: [Training loops](#training-loops).

### Transfer learning

Reusing pretrained models. This is how 90% of modern applied deep learning actually starts.

- **[PyTorch: Transfer Learning Tutorial](https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html)**: free, tutorial. Best minimal example: swap the last layer of a ResNet and fine-tune.
- **[fast.ai: chapter on transfer learning](https://github.com/fastai/fastbook)**: free, book chapter (Notebook 5). Best explanation that starts from a working example, with code that runs.
- **[HuggingFace: Transfer Learning](https://huggingface.co/learn/nlp-course/chapter1/4)**: free, course chapter. `[underrated as intro]` Transfer learning in the NLP context you'll actually use it in later.

Prereqs: [Training loops](#training-loops).

---

## Phase 4: NLP

Explain attention and build a text classifier. Older techniques (tokenization, embeddings) are still needed for LLM work; newer parts (transformer architecture) are the whole modern game.

### Tokenization

How text becomes the integers a model actually operates on. Every downstream LLM concept (context window, cost, prompt injection via odd tokens) traces back to this step.

- **[Hugging Face LLM Course: tokenizers](https://huggingface.co/learn/llm-course/chapter6/1)**: free, course chapter. Covers BPE, WordPiece, and Unigram, and has you build a tokenizer block by block.
- **[Let's build the GPT Tokenizer](https://www.youtube.com/watch?v=zduSFxRajkE)** (Karpathy): free, YouTube, ~2 hours. `[underrated depth]` Builds a byte-pair-encoding tokenizer from scratch in code; the one video that removes all the mystery around why LLMs are bad at spelling and arithmetic.
- **[tiktoken](https://github.com/openai/tiktoken)** (OpenAI): free, repo. The tokenizer library used by OpenAI's models; the README and `playground` examples are enough to see tokenization happen on real text.

Prereqs: [Python](#python).

### Embeddings

Turning words, sentences, or documents into vectors so that similarity becomes a distance calculation. Word embeddings (word2vec era) and the sentence and document embeddings used in modern RAG systems are the same core idea applied at different granularity.

- **[The Illustrated Word2vec](https://jalammar.github.io/illustrated-word2vec/)** (Alammar): free, blog. The standard visual introduction to how word embeddings are learned and what the resulting vector space actually encodes.
- **[On word embeddings, Part 1](https://www.ruder.io/word-embeddings-1/)** (Ruder): free, blog. `[underrated]` Goes deeper than most tutorials into why the different word2vec and GloVe objectives produce different embeddings.
- **[OpenAI: Vector embeddings guide](https://platform.openai.com/docs/guides/embeddings)**: free, official docs. The practical modern version: how to call an embeddings API and what to do with the vectors it returns.

Prereqs: [Python](#python), [Math](#math).

### Transformer architecture

Attention, self-attention, and the encoder-decoder (or decoder-only) stack that replaced RNNs for almost everything. This is the architecture every model in Phase 6 is built on.

- **[Attention Is All You Need](https://arxiv.org/abs/1706.03762)** (Vaswani et al.): free, paper. The original. Worth reading once you already have an intuition for attention from elsewhere; dense on a first pass.
- **[The Annotated Transformer](https://nlp.seas.harvard.edu/annotated-transformer/)** (Harvard NLP): free, code walkthrough. Puts the paper's equations next to a working PyTorch implementation, line by line.
- **[Formal Algorithms for Transformers](https://arxiv.org/abs/2207.09238)** (Phuong, Hutter, DeepMind): free, paper. `[underrated]` A complete, precise pseudocode reference for every transformer variant in about ten pages; useful once prose explanations start feeling hand-wavy.

Prereqs: [Neural network basics](#neural-network-basics), [Embeddings](#embeddings).

### Classical NLP tasks

Named entity recognition, part-of-speech tagging, sentiment analysis, text classification: the tasks NLP meant before LLMs made "just prompt it" an option. Still the faster and cheaper choice when a task is narrow and well defined.

- **[Advanced NLP with spaCy](https://course.spacy.io/en/)**: free, interactive course. Builds rule-based and statistical pipelines for these exact tasks, in the browser, no setup.
- **[Speech and Language Processing, 3rd ed. draft](https://web.stanford.edu/~jurafsky/slp3/)** (Jurafsky, Martin): free, book draft. `[underrated]` The standard academic text for the field, free and continuously updated; most people jump straight to a course and never discover it covers the same ground more rigorously.
- **[spaCy 101](https://spacy.io/usage/spacy-101)**: free, official docs. Reach for this once you know what task you want and just need the API.

Prereqs: [Python](#python), [Embeddings](#embeddings).

---

## Phase 5: Computer Vision

Fine-tune a vision model and run inference. Skippable if your target is pure LLM/agent work; keep for breadth.

### Image basics

_TBD, targeted for v1.2_

### Convnets

_TBD, targeted for v1.2_

### Object detection

_TBD, targeted for v1.3_

### Vision transformers

_TBD, targeted for v1.3_

---

## Phase 6: LLMs

Build a production LLM app with structured output. This is where the modern AI engineer stack starts.

### How LLMs work

Attention, next-token prediction, sampling. Enough to reason about what an LLM will and won't do, without needing to train one from scratch.

- **[Intro to Large Language Models](https://www.youtube.com/watch?v=zjkBMFhNj_g)** (Karpathy): free, YouTube, 1 hour. The introduction every AI engineer should watch, about an hour long. Explains what LLMs are, how they're trained, what they can't do.
- **[Let's build GPT: from scratch, in code, spelled out](https://www.youtube.com/watch?v=kCc8FmEb1nY)** (Karpathy): free, YouTube, ~2 hours. `[underrated depth]` Best "click" for how transformers actually work. Follow along in code.
- **[Anthropic: What are LLMs?](https://docs.claude.com/en/docs/intro-to-claude)**: free, official docs. Good short primer if the Karpathy videos are too much time upfront.
- **[The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)** (Alammar): free, blog. `[underrated]` The one blog post full of diagrams that made attention click for a generation of engineers.

Prereqs: [Transformer architecture](#transformer-architecture).

### Prompt engineering

The craft of getting the model to do what you want. Less mystical than the internet makes it sound; a few real principles you can learn in a weekend.

- **[Anthropic: Prompt Engineering Overview](https://docs.claude.com/en/docs/build-with-claude/prompt-engineering/overview)**: free, official docs. Best structured intro from a model builder.
- **[OpenAI: Prompt engineering](https://platform.openai.com/docs/guides/prompt-engineering)**: free, official docs. Best complement to Anthropic's guide.
- **[Learn Prompting](https://learnprompting.org/)**: free, interactive course. Best full-course treatment if you want structured lessons.
- **[Simon Willison's blog: llms tag](https://simonwillison.net/tags/llms/)**: free, blog. `[underrated]` The most consistently useful practitioner blog on prompting; read the highlights.

Prereqs: [How LLMs work](#how-llms-work).

### Structured output

Getting JSON out of an LLM reliably. This is the technique underneath every real LLM app.

- **[Instructor library docs](https://python.useinstructor.com/)**: free, docs. The canonical library for Pydantic-schema-driven LLM extraction. Multi-provider.
- **[OpenAI: Structured Outputs](https://platform.openai.com/docs/guides/structured-outputs)**: free, official docs. Best treatment of provider-native structured output (versus library-mediated).
- **[Pydantic AI docs](https://ai.pydantic.dev/)**: free, docs. `[underrated as newer]` Type-safe agent framework built by the Pydantic team. Read alongside Instructor to see the two shapes.

Prereqs: [Prompt engineering](#prompt-engineering).

### Evaluation

How you know your LLM app actually works. Different from classical ML evaluation because outputs are open-ended text.

- **[Evaluating LLM-based applications](https://huyenchip.com/2024/07/25/genai-platform.html)** (Chip Huyen): free, blog post. `[underrated as intro]` Best short primer on what to actually measure.
- **[Ragas documentation](https://docs.ragas.io/)**: free, docs. Canonical library for RAG evaluation; the metrics section is the useful part even if you don't use the library.
- **[Eugene Yan: Evals blog series](https://eugeneyan.com/writing/evals/)**: free, blog. `[underrated]` The most practical treatment of building evals for real LLM systems.
- **[OpenAI Evals](https://github.com/openai/evals)**: free, repo. Canonical open-source eval framework; read the examples folder for shapes to copy.

Prereqs: [Prompt engineering](#prompt-engineering).

### Provider SDKs

The Python SDKs you'll actually call: OpenAI, Anthropic, Google.

- **[OpenAI Python SDK docs](https://platform.openai.com/docs/libraries)**: free, docs. Canonical.
- **[Anthropic Python SDK docs](https://docs.claude.com/en/api/client-sdks)**: free, docs. Canonical.
- **[Google Gen AI SDK docs](https://ai.google.dev/gemini-api/docs/quickstart)**: free, docs. Canonical.
- **[LiteLLM docs](https://docs.litellm.ai/)**: free, docs. `[underrated for multi-provider]` One interface across ~100 LLM providers; read the "getting started" and provider-config pages.

Prereqs: [Prompt engineering](#prompt-engineering).

---

## Phase 7: Agents

Build agents in at least two frameworks. See sibling project [real-world-agents](https://github.com/rajeshm71/real-world-agents) for the applied version.

### Tool use

_TBD, targeted for v1.2_

### ReAct pattern

_TBD, targeted for v1.2_

### Multi-agent collaboration

_TBD, targeted for v1.3_

### Framework survey (LangGraph, CrewAI, OpenAI Agents SDK, PydanticAI)

_TBD, targeted for v1.3_

---

## Phase 8: RAG

Build a production RAG pipeline. See sibling project [rag-recipes](https://github.com/rajeshm71/rag-recipes) for the applied version.

### Vector embeddings

The representation a RAG system actually searches over: text turned into vectors so that "relevant to this query" becomes "close in vector space." Builds directly on [Embeddings](#embeddings) from Phase 4, applied to retrieval instead of general similarity.

- **[Pinecone: Learning Center](https://www.pinecone.io/learn/)**: free, guides. The practical, RAG-focused walkthrough of what embeddings are and how they get used in a retrieval pipeline, from a team that builds a vector database for a living.
- **[What are embeddings?](https://vickiboykis.com/what_are_embeddings/)** (Boykis): free, long-form write-up. `[underrated]` Goes further than any vendor guide into the engineering tradeoffs (dimensionality, storage, drift) that show up once a RAG system is in production.
- **[Cohere: Embeddings documentation](https://docs.cohere.com/docs/embeddings)**: free, official docs. A second provider's take on the same API shape as OpenAI's, useful for seeing what's provider-specific versus universal.

Prereqs: [Embeddings](#embeddings).

### Retrieval basics

How a query actually finds its matches: similarity search over the vectors above, and the approximate nearest neighbor algorithms that make it fast at scale.

- **[LlamaIndex: Introduction to RAG](https://developers.llamaindex.ai/python/framework/understanding/rag/)**: free, official docs. The clearest map of the moving parts (loading, indexing, retrieving, querying) from a framework built around exactly this problem.
- **[Vector Search Explained](https://weaviate.io/blog/vector-search-explained)** (Weaviate): free, blog. Explains HNSW and the ANN speed/accuracy tradeoff without requiring a systems background.
- **[Faiss](https://github.com/facebookresearch/faiss)** (Meta): free, repo + wiki. `[underrated]` The library actual similarity search under the hood usually runs on; most people never open it because a managed vector database hides it, but the wiki's tutorial is short and worth the hour.

Prereqs: [Vector embeddings](#vector-embeddings).

### Hybrid search + reranking

Pure vector search misses exact keyword matches; pure keyword search misses paraphrases and synonyms. Hybrid search runs both and combines the scores; reranking then re-scores the combined candidates with a model that reads the query and each candidate together.

- **[Getting Started with Hybrid Search](https://www.pinecone.io/learn/hybrid-search-intro/)** (Pinecone): free, guide. Explains the dense-plus-sparse combination and the tradeoff of weighting one against the other.
- **[Cohere Rerank overview](https://docs.cohere.com/docs/rerank-overview)**: free, official docs. The most commonly reached-for reranking API; read this to see what a second-stage reranker actually buys you over retrieval alone.
- **[Sentence Transformers: Retrieve & Rerank](https://sbert.net/examples/sentence_transformer/applications/retrieve_rerank/README.html)**: free, official docs. `[underrated]` A free, self-hosted cross-encoder reranker; most people reach for a paid API without realizing this ships in a library they likely already have installed.

Prereqs: [Retrieval basics](#retrieval-basics).

### Chunking strategies

How a document gets split before it's embedded. The chunking decision shapes what the retriever can find; too large and irrelevant text dilutes the match, too small and the model loses context it needed.

- **[5 Levels of Text Splitting](https://github.com/FullStackRetrieval-com/RetrievalTutorials/blob/main/tutorials/LevelsOfTextSplitting/5_Levels_Of_Text_Splitting.ipynb)** (Kamradt): free, notebook. The reference nearly everyone in RAG eventually finds; walks character splitting through semantic and agent-driven chunking with runnable code.
- **[LangChain: text splitters](https://docs.langchain.com/oss/python/integrations/splitters)**: free, official docs. Practical reference once you know which strategy you want and just need the class to call.
- **[Is Semantic Chunking Worth the Computational Cost?](https://arxiv.org/abs/2410.13070)**: free, paper. `[underrated]` A rare counter-argument paper: tests whether the trendier semantic-chunking approach actually beats naive fixed-size chunks, and finds the answer is often no.

Prereqs: [Vector embeddings](#vector-embeddings).

### Evaluation and observability

How you know a RAG pipeline is actually retrieving the right context and answering from it honestly, and how you watch it in production once it's live.

- **[Ragas documentation](https://docs.ragas.io/)**: free, docs. The canonical library for RAG-specific metrics (context precision, faithfulness, answer relevance); the concepts page is useful even without adopting the library.
- **[Langfuse documentation](https://langfuse.com/docs)**: free, open source. Tracing and evaluation for a live pipeline: see the actual retrieved chunks and prompts behind any given answer.
- **[Your AI Product Needs Evals](https://hamel.dev/blog/posts/evals/)** (Husain): free, blog post. `[underrated]` A more skeptical, practitioner-first take than most vendor docs on what a useful eval loop actually requires.

Prereqs: [Retrieval basics](#retrieval-basics), [Evaluation](#evaluation).

---

## Phase 9: Fine-tuning

Know when fine-tuning is the right call and can run one.

### When to fine-tune

_TBD, targeted for v1.3_

### LoRA and adapters

_TBD, targeted for v1.4_

### Data preparation

_TBD, targeted for v1.4_

### Open-model serving

_TBD, targeted for v1.4_

---

## Phase 10: MLOps & Production

Deploy + monitor + iterate on a shipped AI system.

### Deployment (HF Spaces, Modal, Replicate)

_TBD, targeted for v1.3_

### Monitoring: cost, latency

_TBD, targeted for v1.4_

### Model versioning

_TBD, targeted for v1.4_

### Observability (Langfuse, Helicone, Phoenix)

_TBD, targeted for v1.4_

### Safety and guardrails

_TBD, targeted for v1.5_

---

## Phase 11: Staying current

Keep up + find work.

### Weekly newsletters

_TBD, targeted for v1.4_

### Communities

_TBD, targeted for v1.4_

### Papers to track

_TBD, targeted for v1.5_

### Landing a job

_TBD, targeted for v1.5_

---

## Tracks

### Default AI engineer path

Phase 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11 in order. Full breadth. ~6-9 months full-time equivalent.

### LLM engineer fast-track

Phase 1, 2 (lighter pass), 3 (lighter pass through PyTorch + training loops), 6, 7, 8, 9, 10, 11. Skips computer vision; goes through classical ML and deep-learning fundamentals at reduced depth. For readers who already program and want the modern LLM/agent/RAG stack fast. ~3-4 months full-time equivalent.

---

## Contributing

Fixes for dead links, proposals for a better pick, missing topics you notice: see [`CONTRIBUTING.md`](CONTRIBUTING.md). Two rules:

1. One pick per resource type per topic. If the current pick is genuinely worse than yours, PR to replace it, don't PR to add a second.
2. Every new pick needs a one-sentence why it beats the current one.

## License

Content licensed [CC BY 4.0](LICENSE). Free to use, share, adapt with attribution.
