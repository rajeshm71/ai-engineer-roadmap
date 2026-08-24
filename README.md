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
- **[Fluent Python, 2nd ed.](https://www.oreilly.com/library/view/fluent-python-2nd/9781492056348/)** (Ramalho): $50, book, 1000 pages. `[underrated]` The one book that will change how you write Python. Come back to it after a year of Python under your belt; skip on the first pass.

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
- **[Hands-On Machine Learning with Scikit-Learn, Keras & TensorFlow, 3rd ed.](https://www.oreilly.com/library/view/hands-on-machine-learning/9781098125967/)** (Géron): $60, book. Best practitioner text; chapters 1-6 cover supervised learning end to end with real code.
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
- **[Designing Machine Learning Systems](https://www.oreilly.com/library/view/designing-machine-learning/9781098107956/)** (Huyen) chapter 6: $50, book. `[underrated]` The one book that treats evaluation as a production problem, not a homework problem.

Prereqs: [Supervised learning](#supervised-learning).

### Feature engineering

Turning raw data into what the model actually needs. Modern deep-learning talks less about this, but real production pipelines are half feature engineering.

- **[Feature Engineering for Machine Learning](https://www.oreilly.com/library/view/feature-engineering-for/9781491953235/)** (Zheng, Casari): $40, book. Canonical practitioner text.
- **[Kaggle Learn: Feature Engineering](https://www.kaggle.com/learn/feature-engineering)**: free, short course. Fastest way to see the techniques applied to real tabular data.
- **[Applied Predictive Modeling](http://appliedpredictivemodeling.com/)** (Kuhn, Johnson): $70, book. `[underrated by modern ML crowd]` Older and written in R, but the reasoning about feature construction still beats most newer books.

Prereqs: [Supervised learning](#supervised-learning).

### scikit-learn

The library everything else in classical ML uses. Get comfortable with the API; you'll use it in fine-tuning data prep, evaluation, and prototyping.

- **[scikit-learn: user guide](https://scikit-learn.org/stable/user_guide.html)**: free, official docs. Better than most tutorials. Start with "Getting started".
- **[Hands-On ML](https://www.oreilly.com/library/view/hands-on-machine-learning/9781098125967/)** (Géron) chapters 3-8: $60, book. Best practical introduction to sklearn, written so you can run the code as you read.
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

- **[Deep Learning with PyTorch](https://www.manning.com/books/deep-learning-with-pytorch)** (Stevens, Antiga, Viehmann): free, book, 500 pages, official PDF. Canonical textbook. Written by PyTorch core devs.
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

_TBD, targeted for v1.1_

### Embeddings

_TBD, targeted for v1.1_

### Transformer architecture

_TBD, targeted for v1.1_

### Classical NLP tasks

_TBD, targeted for v1.2_

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

_TBD, targeted for v1.2_

### Retrieval basics

_TBD, targeted for v1.2_

### Hybrid search + reranking

_TBD, targeted for v1.3_

### Chunking strategies

_TBD, targeted for v1.3_

### Evaluation and observability

_TBD, targeted for v1.3_

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
