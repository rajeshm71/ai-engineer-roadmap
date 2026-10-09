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
- [Resource index](#resource-index)
- [Contributing](#contributing)
- [License](#license)

---

## Phase 1: Prerequisites

Install packages, use git, do linear algebra, run bash. If you already program, move quickly through this phase; otherwise these are the basics everyone needs.

### Python

The one language modern AI runs on. Get to the point where you can read someone else's PyTorch code and not get lost.

- **[Automate the Boring Stuff with Python](https://automatetheboringstuff.com/)** (Sweigart): free, book. Takes people who have never programmed to a productive level in Python.
- **[Real Python: Python Basics learning path](https://realpython.com/learning-paths/python-basics/)**: free, articles + exercises. Good if you learn better in small pieces than a whole book.
- **[Fluent Python, 2nd ed.](https://www.oreilly.com/library/view/fluent-python-2nd/9781492056348/)** (Ramalho): paid, book, 1000 pages. The one book that will change how you write Python. Come back to it after a year of Python under your belt; skip on the first pass.
- **[Learn Python - Full Course for Beginners](https://www.youtube.com/watch?v=rfscVS0vtbw)** (freeCodeCamp.org): free, YouTube, ~4.5 hours. Best pick if you'd rather watch someone type than read a book cover to cover.

Prereqs: none.

### Math

Enough linear algebra, calculus, and probability to read papers and understand what a training loop is doing.

- **[Mathematics for Machine Learning](https://mml-book.github.io/)** (Deisenroth, Faisal, Ong): free, book. The canonical "math you need for ML, without going full pure-math," free PDF from the authors.
- **[Essence of Linear Algebra](https://www.3blue1brown.com/topics/linear-algebra)** (3Blue1Brown): free, YouTube series, ~5 hours. Builds intuition for what vectors, matrices, and eigenvectors actually are before you drown in notation.
- **[Introduction to Probability](https://www.edx.org/learn/probability/harvard-university-introduction-to-probability)** (Blitzstein, Harvard, Stat 110): free to audit, course. The probability foundation almost nobody has and everyone should. Come back after ML basics if the full course is too much upfront.

Prereqs: [Python](#python) helps for the ML-flavored examples.

### Git

Version control. Non-negotiable.

- **[Pro Git](https://git-scm.com/book/en/v2)** (Chacon, Straub): free, book. The canonical reference. Read chapters 1-3; treat the rest as a reference to return to.
- **[Oh Shit, Git!?!](https://ohshitgit.com/)** (Sylor-Miller): free, cheat sheet. The one page you reach for when you've mangled a rebase and don't know what to do.
- **[Git and GitHub for Beginners - Crash Course](https://www.youtube.com/watch?v=RGOj5yH7evk)** (freeCodeCamp.org): free, YouTube, ~1 hour. Fastest way to see commits, branching, and a GitHub PR workflow actually happen before you touch the CLI yourself.

Prereqs: [CLI / Linux](#cli--linux).

### CLI / Linux

Command line + basic shell. If `cd`, `ls`, `grep`, `chmod`, `|`, `>` aren't second nature, start here.

- **[The Missing Semester of Your CS Education](https://missing.csail.mit.edu/)** (MIT): free, lecture series, ~12 hours. Everything CS programs assume you already know: shell, git, editors, debugging, remote machines. Watch lectures 1-4 minimum.
- **[The Linux Command Line](https://linuxcommand.org/tlcl.php)** (Shotts): free, book. Systematic reference if you want a real book.

Prereqs: none.

---

## Phase 2: Classical ML

Build + evaluate a scikit-learn classifier end to end. Modern LLM work still rests on the concepts here (loss functions, regularization, evaluation metrics). Don't skip.

### Supervised learning

Regression + classification: predicting labels from features. Learn the vocabulary here, it applies everywhere else.

- **[Machine Learning Specialization](https://www.coursera.org/specializations/machine-learning-introduction)** (Andrew Ng, Coursera): free to audit, course. The canonical entry point for supervised learning.
- **[Hands-On Machine Learning with Scikit-Learn, Keras & TensorFlow, 3rd ed.](https://www.oreilly.com/library/view/hands-on-machine-learning/9781098125967/)** (Géron): paid, book. Best practitioner text; chapters 1-6 cover supervised learning end to end with real code.
- **[StatQuest with Josh Starmer](https://www.youtube.com/@statquest)**: free, YouTube channel. Visual explanations of every classical ML concept. One of the clearest resources for the "why" behind linear regression, logistic regression, decision trees.

Prereqs: [Math](#math), [Python](#python).

### Unsupervised learning

Finding structure without labels: clustering, dimensionality reduction.

- **[An Introduction to Statistical Learning, 2nd ed.](https://www.statlearning.com/)** (James, Witten, Hastie, Tibshirani): free, book, Python and R editions available. The definitive ML book grounded in statistics. Chapter 12 for unsupervised.
- **[StatQuest: Clustering and PCA playlists](https://www.youtube.com/@statquest/playlists)**: free, YouTube. If ISLR feels too dense, watch these first, then read.
- **[scikit-learn: clustering user guide](https://scikit-learn.org/stable/modules/clustering.html)**: free, official docs. Reach for this when you're about to `pip install` a clustering algorithm and want to know which one to pick.

Prereqs: [Supervised learning](#supervised-learning).

### Model evaluation

Metrics, cross-validation, train/val/test splits. The most-skipped step and the one that separates working models from broken ones.

- **[scikit-learn: model evaluation user guide](https://scikit-learn.org/stable/modules/model_evaluation.html)**: free, official docs. Canonical reference for every metric name and when to use it.
- **[Google's Machine Learning Crash Course: Classification](https://developers.google.com/machine-learning/crash-course/classification/video-lecture)**: free, course section. Best intro to precision/recall/AUC and why accuracy alone lies.
- **[Designing Machine Learning Systems](https://www.oreilly.com/library/view/designing-machine-learning/9781098107956/)** (Huyen) chapter 6: paid, book. The one book that treats evaluation as a production problem, not a homework problem.

Prereqs: [Supervised learning](#supervised-learning).

### Feature engineering

Turning raw data into what the model actually needs. Modern deep-learning talks less about this, but real production pipelines are half feature engineering.

- **[Feature Engineering for Machine Learning](https://www.oreilly.com/library/view/feature-engineering-for/9781491953235/)** (Zheng, Casari): paid, book. Canonical practitioner text.
- **[Kaggle Learn: Feature Engineering](https://www.kaggle.com/learn/feature-engineering)**: free, short course. Fastest way to see the techniques applied to real tabular data.
- **[Applied Predictive Modeling](http://appliedpredictivemodeling.com/)** (Kuhn, Johnson): paid, book. Older and written in R, but the reasoning about feature construction still beats most newer books.
- **[100 Days of Machine Learning](https://www.youtube.com/playlist?list=PLKnIA16_Rmvbr7zKYQuBfsVkjoLcJgxHH)** (CampusX): free, YouTube playlist. Feature scaling, encoding categorical data, and outlier handling each get their own dedicated episode, worked through on real datasets.

Prereqs: [Supervised learning](#supervised-learning).

### scikit-learn

The library everything else in classical ML uses. Get comfortable with the API; you'll use it in fine-tuning data prep, evaluation, and prototyping.

- **[scikit-learn: user guide](https://scikit-learn.org/stable/user_guide.html)**: free, official docs. Better than most tutorials. Start with "Getting started".
- **[Hands-On ML](https://www.oreilly.com/library/view/hands-on-machine-learning/9781098125967/)** (Géron) chapters 3-8: paid, book. Best practical introduction to sklearn, written so you can run the code as you read.
- **[Machine Learning in Python with scikit-learn](https://www.fun-mooc.fr/en/courses/machine-learning-python-scikit-learn/)** (Inria, scikit-learn core team): free, MOOC. Taught by the people who wrote the library.

Prereqs: [Supervised learning](#supervised-learning).

---

## Phase 3: Deep Learning

Train a CNN from scratch on a small dataset. This is where PyTorch enters and the rest of the roadmap depends on it.

### Neural network basics

Backpropagation, activation functions, loss functions. Understand these before you touch PyTorch, or you'll be copying code without understanding what it does.

- **[Deep Learning Specialization](https://www.coursera.org/specializations/deep-learning)** (Andrew Ng, DeepLearning.AI): free to audit, course. Canonical intro to DL; courses 1 and 2 for the fundamentals.
- **[3Blue1Brown: Neural Networks playlist](https://www.3blue1brown.com/topics/neural-networks)**: free, YouTube, ~1 hour. Best intuition for what backprop is actually doing before you write a line of code.
- **[Neural Networks: Zero to Hero](https://karpathy.ai/zero-to-hero.html)** (Karpathy): free, YouTube series, ~10 hours. Builds autograd from scratch, then a transformer. If you only watch one thing in this phase, watch the first video.

Prereqs: [Math](#math), [Python](#python).

### PyTorch

The deep-learning framework most modern research and a lot of production runs on. Learn it after neural-network basics; you'll use it through the rest of this roadmap.

- **[Deep Learning with PyTorch, 2nd ed.](https://www.manning.com/books/deep-learning-with-pytorch-second-edition)** (Antiga, Stevens, Huang, Viehmann): paid, book. Canonical textbook, written by PyTorch core devs and updated for transformers and generative models.
- **[PyTorch: 60 Minute Blitz](https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html)**: free, tutorial, ~2 hours. Fastest way to get a real training loop running.
- **[Zero to Hero: makemore series](https://karpathy.ai/zero-to-hero.html)** (Karpathy): free, YouTube. Continuation of the neural-network-basics pick above, now in PyTorch. Best "click" moment for autograd.
- **[fast.ai Practical Deep Learning](https://course.fast.ai/)**: free, course, ~22 hours. Alternative starting point if you'd rather start from working projects than from theory.
- **[PyTorch Discuss](https://discuss.pytorch.org/)**: free, forum. Searchable, core devs answer, more useful than Stack Overflow for PyTorch-specific questions.

Prereqs: [Neural network basics](#neural-network-basics), [Python](#python).

### Training loops

The `for epoch in range(...)` machinery: batches, optimizers, learning-rate schedules, checkpointing.

- **[PyTorch: Training a Classifier tutorial](https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html)**: free, tutorial. The one you'll copy-paste the first ten times you write a training loop.
- **[fast.ai: lessons 1-4](https://course.fast.ai/)**: free, video lessons. Cover the training loop from a practitioner angle.
- **[A Recipe for Training Neural Networks](https://karpathy.github.io/2019/04/25/recipe/)** (Karpathy): free, blog post. The one page every practitioner rediscovers eventually. Read it before your first "why isn't this model learning" bug.

Prereqs: [PyTorch](#pytorch).

### Regularization

Dropout, weight decay, batch normalization, early stopping. Why your model works on training data and fails on validation.

- **[Deep Learning](https://www.deeplearningbook.org/)** (Goodfellow, Bengio, Courville) chapter 7: free, book. The academic reference.
- **[Sebastian Raschka's blog: regularization posts](https://sebastianraschka.com/blog/)**: free, blog. Concrete visual walkthroughs of dropout, batch norm, layer norm, better than most textbook treatments.
- **[Deep Learning Specialization, course 2](https://www.coursera.org/learn/deep-neural-network)** (Ng): free to audit. Section on regularization is clear and focused.

Prereqs: [Training loops](#training-loops).

### Transfer learning

Reusing pretrained models. This is how 90% of modern applied deep learning actually starts.

- **[PyTorch: Transfer Learning Tutorial](https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html)**: free, tutorial. Best minimal example: swap the last layer of a ResNet and fine-tune.
- **[fast.ai: chapter on transfer learning](https://github.com/fastai/fastbook)**: free, book chapter (Notebook 5). Best explanation that starts from a working example, with code that runs.
- **[HuggingFace: Transfer Learning](https://huggingface.co/learn/nlp-course/chapter1/4)**: free, course chapter. Transfer learning in the NLP context you'll actually use it in later.

Prereqs: [Training loops](#training-loops).

---

## Phase 4: NLP

Explain attention and build a text classifier. Older techniques (tokenization, embeddings) are still needed for LLM work; newer parts (transformer architecture) are the whole modern game.

### Tokenization

How text becomes the integers a model actually operates on. Every downstream LLM concept (context window, cost, prompt injection via odd tokens) traces back to this step.

- **[Hugging Face LLM Course: tokenizers](https://huggingface.co/learn/llm-course/chapter6/1)**: free, course chapter. Covers BPE, WordPiece, and Unigram, and has you build a tokenizer block by block.
- **[Let's build the GPT Tokenizer](https://www.youtube.com/watch?v=zduSFxRajkE)** (Karpathy): free, YouTube, ~2 hours. Builds a byte-pair-encoding tokenizer from scratch in code; the one video that removes all the mystery around why LLMs are bad at spelling and arithmetic.
- **[tiktoken](https://github.com/openai/tiktoken)** (OpenAI): free, repo. The tokenizer library used by OpenAI's models; the README and `playground` examples are enough to see tokenization happen on real text.

Prereqs: [Python](#python).

### Embeddings

Turning words, sentences, or documents into vectors so that similarity becomes a distance calculation. Word embeddings (word2vec era) and the sentence and document embeddings used in modern RAG systems are the same core idea applied at different granularity.

- **[The Illustrated Word2vec](https://jalammar.github.io/illustrated-word2vec/)** (Alammar): free, blog. The standard visual introduction to how word embeddings are learned and what the resulting vector space actually encodes.
- **[On word embeddings, Part 1](https://www.ruder.io/word-embeddings-1/)** (Ruder): free, blog. Goes deeper than most tutorials into why the different word2vec and GloVe objectives produce different embeddings.
- **[OpenAI: Vector embeddings guide](https://platform.openai.com/docs/guides/embeddings)**: free, official docs. The practical modern version: how to call an embeddings API and what to do with the vectors it returns.
- **[Word Embedding and Word2Vec, Clearly Explained!!!](https://www.youtube.com/watch?v=viZrOnJclY0)** (StatQuest with Josh Starmer): free, YouTube. Same clear-explanation style as the classical-ML picks above, applied to how word2vec actually learns a vector space.

Prereqs: [Python](#python), [Math](#math).

### Transformer architecture

Attention, self-attention, and the encoder-decoder (or decoder-only) stack that replaced RNNs for almost everything. This is the architecture every model in Phase 6 is built on.

- **[Attention Is All You Need](https://arxiv.org/abs/1706.03762)** (Vaswani et al.): free, paper. The original. Worth reading once you already have an intuition for attention from elsewhere; dense on a first pass.
- **[The Annotated Transformer](https://nlp.seas.harvard.edu/annotated-transformer/)** (Harvard NLP): free, code walkthrough. Puts the paper's equations next to a working PyTorch implementation, line by line.
- **[Formal Algorithms for Transformers](https://arxiv.org/abs/2207.09238)** (Phuong, Hutter, DeepMind): free, paper. A complete, precise pseudocode reference for every transformer variant in about ten pages; useful once prose explanations start feeling hand-wavy.
- **[Attention in transformers, step-by-step](https://www.youtube.com/watch?v=eMlx5fFNoYc)** (3Blue1Brown): free, YouTube, ~26 minutes. The clearest visual walkthrough of query-key-value attention; watch before the paper, not instead of it.

Prereqs: [Neural network basics](#neural-network-basics), [Embeddings](#embeddings).

### Classical NLP tasks

Named entity recognition, part-of-speech tagging, sentiment analysis, text classification: the tasks NLP meant before LLMs made "just prompt it" an option. Still the faster and cheaper choice when a task is narrow and well defined.

- **[Advanced NLP with spaCy](https://course.spacy.io/en/)**: free, interactive course. Builds rule-based and statistical pipelines for these exact tasks, in the browser, no setup.
- **[Speech and Language Processing, 3rd ed. draft](https://web.stanford.edu/~jurafsky/slp3/)** (Jurafsky, Martin): free, book draft. The standard academic text for the field, free and continuously updated; most people jump straight to a course and never discover it covers the same ground more rigorously.
- **[spaCy 101](https://spacy.io/usage/spacy-101)**: free, official docs. Reach for this once you know what task you want and just need the API.
- **[Natural Language Processing with spaCy & Python: Course for Beginners](https://www.youtube.com/watch?v=dIUTsFT2MeQ)** (freeCodeCamp.org): free, YouTube, ~3 hours. Walks through NER, POS tagging, and text classification pipelines in spaCy end to end, code alongside explanation.

Prereqs: [Python](#python), [Embeddings](#embeddings).

---

## Phase 5: Computer Vision

Fine-tune a vision model and run inference. Skippable if your target is pure LLM/agent work; keep for breadth.

### Image basics

How an image becomes a tensor a model can read: pixels, channels, resizing, normalization, and the augmentations that make a small dataset behave like a larger one.

- **[Practical Deep Learning for Coders](https://course.fast.ai/)**: free, course. Trains a real image classifier in lesson 1 before explaining any of the underlying theory.
- **[Torchvision: transforms](https://docs.pytorch.org/vision/stable/transforms.html)**: free, official docs. The practical reference for resizing, cropping, and normalizing images before they reach a model.
- **[But what is a convolution?](https://www.3blue1brown.com/lessons/convolutions/)** (3Blue1Brown): free, YouTube, ~23 minutes. Builds the intuition for what a convolution is doing before you meet it as a neural network layer.

Prereqs: [PyTorch](#pytorch).

### Convnets

Convolutional layers exploit the fact that nearby pixels are related; stacking them builds up from edges to textures to whole objects. The dominant architecture for vision before transformers arrived.

- **[CS231n: Convolutional Neural Networks](https://cs231n.github.io/convolutional-networks/)** (Stanford): free, lecture notes. The canonical explanation of convolutional layers, pooling, and classic architectures.
- **[A Guide to Convolution Arithmetic for Deep Learning](https://arxiv.org/abs/1603.07285)** (Dumoulin, Visin): free, paper. Answers the "what output size do I actually get" question with diagrams, for every combination of padding, stride, and dilation.
- **[CS231n Winter 2016: Lecture 7: Convolutional Neural Networks](https://www.youtube.com/watch?v=LxfUGhug-iQ)** (Stanford, Karpathy & Fei-Fei Li): free, YouTube. The lecture the notes above are drawn from, taught by the people who wrote them.

Prereqs: [Image basics](#image-basics), [Training loops](#training-loops).

### Object detection

Locating objects in an image, not just classifying the whole picture: bounding boxes, confidence scores, and the metrics (IoU, mAP) used to judge them.

- **[Ultralytics YOLO: Object Detection](https://docs.ultralytics.com/tasks/detect)**: free, official docs. The most widely deployed real-time detector family; the docs double as a practical primer on the task itself.
- **[TorchVision Object Detection Finetuning Tutorial](https://docs.pytorch.org/tutorials/intermediate/torchvision_tutorial.html)**: free, official tutorial. Fine-tunes a pretrained Mask R-CNN on a small custom dataset end to end.
- **[mAP (mean Average Precision) for Object Detection](https://naokishibuya.github.io/blog/2022-05-27-object-detection-mean-average-precision/)** (Shibuya): free, blog. Works through precision, recall, and IoU by hand before computing mAP, rather than treating it as a black-box metric.

Prereqs: [Convnets](#convnets).

### Vision transformers

Applying the transformer architecture from Phase 4 to images by splitting them into patches and treating each patch as a token. Increasingly the default choice over convnets at large data scale.

- **[An Image is Worth 16x16 Words](https://arxiv.org/abs/2010.11929)** (Dosovitskiy et al.): free, paper. The original ViT paper; shows a plain transformer competing with convnets once pretraining data is large enough.
- **[Hugging Face: Vision Transformer (ViT)](https://huggingface.co/docs/transformers/model_doc/vit)**: free, official docs. The practical route: load a pretrained ViT and fine-tune it in a few lines.
- **[Hugging Face Computer Vision Course: ViT for image classification](https://huggingface.co/learn/computer-vision-course/unit3/vision-transformers/vision-transformers-for-image-classification)**: free, course chapter. A newer, less-discovered course that walks the patch-embedding-to-classification pipeline step by step.
- **[ViT (Vision Transformer) - An Image Is Worth 16x16 Words (Paper Explained)](https://www.youtube.com/watch?v=8phM16htKbU)** (Yannic Kilcher): free, YouTube, ~30 minutes. A thorough walkthrough of the original paper for anyone who wants the reasoning behind ViT, not just the API to run one.

Prereqs: [Transformer architecture](#transformer-architecture), [Convnets](#convnets).

---

## Phase 6: LLMs

Build a production LLM app with structured output. This is where the modern AI engineer stack starts.

### How LLMs work

Attention, next-token prediction, sampling. Enough to reason about what an LLM will and won't do, without needing to train one from scratch.

- **[Intro to Large Language Models](https://www.youtube.com/watch?v=zjkBMFhNj_g)** (Karpathy): free, YouTube, 1 hour. The introduction every AI engineer should watch, about an hour long. Explains what LLMs are, how they're trained, what they can't do.
- **[Let's build GPT: from scratch, in code, spelled out](https://www.youtube.com/watch?v=kCc8FmEb1nY)** (Karpathy): free, YouTube, ~2 hours. Best "click" for how transformers actually work. Follow along in code.
- **[The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)** (Alammar): free, blog. The one blog post full of diagrams that made attention click for a generation of engineers.

Prereqs: [Transformer architecture](#transformer-architecture).

### Prompt engineering

The craft of getting the model to do what you want. Less mystical than the internet makes it sound; a few real principles you can learn in a weekend.

- **[Anthropic: Prompt Engineering Overview](https://docs.claude.com/en/docs/build-with-claude/prompt-engineering/overview)**: free, official docs. Best structured intro from a model builder.
- **[OpenAI: Prompt engineering](https://platform.openai.com/docs/guides/prompt-engineering)**: free, official docs. Best complement to Anthropic's guide.
- **[Learn Prompting](https://learnprompting.org/)**: free, interactive course. Best full-course treatment if you want structured lessons.
- **[Simon Willison's blog: llms tag](https://simonwillison.net/tags/llms/)**: free, blog. The most consistently useful practitioner blog on prompting; read the highlights.
- **[AI prompt engineering: A deep dive](https://www.youtube.com/watch?v=T9aRN5JkmL8)** (Anthropic): free, YouTube. Anthropic's own prompt engineering team walking through their principles directly, rather than a third party's summary of them.

Prereqs: [How LLMs work](#how-llms-work).

### Structured output

Getting JSON out of an LLM reliably. This is the technique underneath every real LLM app.

- **[Instructor library docs](https://python.useinstructor.com/)**: free, docs. The canonical library for Pydantic-schema-driven LLM extraction. Multi-provider.
- **[OpenAI: Structured Outputs](https://platform.openai.com/docs/guides/structured-outputs)**: free, official docs. Best treatment of provider-native structured output (versus library-mediated).
- **[Pydantic AI docs](https://ai.pydantic.dev/)**: free, docs. Type-safe agent framework built by the Pydantic team. Read alongside Instructor to see the two shapes.
- **[OpenAI DevDay 2024: Structured outputs for reliable applications](https://www.youtube.com/watch?v=kE4BkATIl9c)** (OpenAI): free, YouTube. The team that built the feature explaining why provider-native structured output exists and what it fixes over prompt-only JSON.

Prereqs: [Prompt engineering](#prompt-engineering).

### Evaluation

How you know your LLM app actually works. Different from classical ML evaluation because outputs are open-ended text.

- **[Ragas documentation](https://docs.ragas.io/)**: free, docs. Canonical library for RAG evaluation; the metrics section is the useful part even if you don't use the library.
- **[Eugene Yan: Evals blog series](https://eugeneyan.com/writing/evals/)**: free, blog. The most practical treatment of building evals for real LLM systems.
- **[OpenAI Evals](https://github.com/openai/evals)**: free, repo. Canonical open-source eval framework; read the examples folder for shapes to copy.
- **[How to Systematically Setup LLM Evals](https://www.youtube.com/watch?v=a3SMraZWNNs)** (Datalumina): free, YouTube. A practical walkthrough of unit tests, human/model evals, and LLM-as-a-judge in one place, useful for seeing the pieces assembled rather than read about separately.

Prereqs: [Prompt engineering](#prompt-engineering).

### Provider SDKs

The Python SDKs you'll actually call: OpenAI, Anthropic, Google.

- **[OpenAI Python SDK docs](https://platform.openai.com/docs/libraries)**: free, docs. Canonical.
- **[Anthropic Python SDK docs](https://docs.claude.com/en/api/client-sdks)**: free, docs. Canonical.
- **[Google Gen AI SDK docs](https://ai.google.dev/gemini-api/docs/quickstart)**: free, docs. Canonical.
- **[LiteLLM docs](https://docs.litellm.ai/)**: free, docs. One interface across ~100 LLM providers; read the "getting started" and provider-config pages.

Prereqs: [Prompt engineering](#prompt-engineering).

---

## Phase 7: Agents

Build agents in at least two frameworks. See sibling project [real-world-agents](https://github.com/rajeshm71/real-world-agents) for the applied version.

### Tool use

Letting a model call functions you define, so it can look things up or take actions instead of only generating text. The mechanism underneath almost everything else in this phase.

- **[Claude: Tool use overview](https://platform.claude.com/docs/en/agents-and-tools/tool-use/overview)**: free, official docs. Explains client-side versus server-side tools and how Claude decides when to call one.
- **[OpenAI: Function calling guide](https://platform.openai.com/docs/guides/function-calling)**: free, official docs. The equivalent mechanism on the other major provider; read both to see what's provider-specific.
- **[PydanticAI: Tools](https://ai.pydantic.dev/tools/)**: free, official docs. Tool definitions as typed Python functions with automatic schema generation, instead of hand-written JSON schemas.
- **[Function Calling Demo at OpenAI DevDay](https://www.youtube.com/watch?v=D1Oqt7vig9k)** (OpenAI, Nov 2023): free, YouTube. The original demo introducing function calling, the mechanism most tool-use APIs since have converged on.

Prereqs: [Structured output](#structured-output).

### ReAct pattern

Interleaving reasoning ("what should I do next") with acting ("call this tool, observe the result") in a loop, instead of planning everything upfront. The pattern most agent frameworks build their default loop around.

- **[ReAct: Synergizing Reasoning and Acting in Language Models](https://arxiv.org/abs/2210.03629)** (Yao et al.): free, paper. The original; short enough to read directly rather than only through summaries.
- **[LLM Powered Autonomous Agents](https://lilianweng.github.io/posts/2023-06-23-agent/)** (Weng): free, blog post. Places ReAct inside the larger design space (planning, memory, tool use); most people cite this post without having read past the ReAct section.

Prereqs: [Tool use](#tool-use).

### Multi-agent collaboration

Splitting a task across multiple agents, each with a narrower role, instead of one agent doing everything. Worth the added complexity only when a single agent's context or focus becomes the bottleneck.

- **[Building Effective Agents](https://www.anthropic.com/engineering/building-effective-agents)** (Anthropic): free, official blog. Draws the line between a workflow (fixed code paths) and an agent (the model directs its own steps), and covers five common patterns before multi-agent is even necessary.
- **[How we built our multi-agent research system](https://www.anthropic.com/engineering/multi-agent-research-system)** (Anthropic): free, official blog. Concrete, with real numbers (token cost, failure modes) from a production multi-agent system, rather than an architecture diagram alone.
- **[CrewAI documentation](https://docs.crewai.com/)**: free, official docs. The framework built specifically around role-based multi-agent "crews"; see the pattern as a first-class abstraction rather than something you assemble yourself.
- **[Anthropic: How to Build Multi-Agent Systems](https://www.youtube.com/watch?v=0HljLKVdtjo)** (Anthropic): free, YouTube. Drawn from the same production system as the writeup above, from the team that built it.

Prereqs: [ReAct pattern](#react-pattern).

### Framework survey (LangGraph, CrewAI, OpenAI Agents SDK, PydanticAI)

Four different opinions on how much structure an agent framework should impose. Worth knowing what each optimizes for before picking one; see [real-world-agents](https://github.com/rajeshm71/real-world-agents) for all four applied to real tasks.

- **[LangGraph](https://docs.langchain.com/oss/python/langgraph/overview)**: free, official docs. Graph-based orchestration; the most explicit about control flow, at the cost of more boilerplate.
- **[CrewAI documentation](https://docs.crewai.com/)**: free, official docs. Role-based abstraction (agents, crews, tasks); fastest to a working multi-agent prototype.
- **[OpenAI Agents SDK](https://openai.github.io/openai-agents-python/)**: free, official docs. Lightweight, provider-authored, minimal abstraction over the tool-use loop itself.
- **[PydanticAI documentation](https://ai.pydantic.dev/)**: free, official docs. Type-safe agents and dependency injection built by the Pydantic team; the newest of the four and the least likely to already be on your radar.

Prereqs: [Tool use](#tool-use).

---

## Phase 8: RAG

Build a production RAG pipeline. See sibling project [rag-recipes](https://github.com/rajeshm71/rag-recipes) for the applied version.

### Vector embeddings

The representation a RAG system actually searches over: text turned into vectors so that "relevant to this query" becomes "close in vector space." Builds directly on [Embeddings](#embeddings) from Phase 4, applied to retrieval instead of general similarity.

- **[Pinecone: Learning Center](https://www.pinecone.io/learn/)**: free, guides. The practical, RAG-focused walkthrough of what embeddings are and how they get used in a retrieval pipeline, from a team that builds a vector database for a living.
- **[What are embeddings?](https://vickiboykis.com/what_are_embeddings/)** (Boykis): free, long-form write-up. Goes further than any vendor guide into the engineering tradeoffs (dimensionality, storage, drift) that show up once a RAG system is in production.
- **[Cohere: Embeddings documentation](https://docs.cohere.com/docs/embeddings)**: free, official docs. A second provider's take on the same API shape as OpenAI's, useful for seeing what's provider-specific versus universal.
- **[OpenAI Embeddings and Vector Databases Crash Course](https://www.youtube.com/watch?v=ySus5ZS0b94)** (Adrian Twarog): free, YouTube. A hands-on walkthrough of generating embeddings and storing/querying them in a vector database, the full workflow in one sitting.

Prereqs: [Embeddings](#embeddings).

### Retrieval basics

How a query actually finds its matches: similarity search over the vectors above, and the approximate nearest neighbor algorithms that make it fast at scale.

- **[LlamaIndex: Introduction to RAG](https://developers.llamaindex.ai/python/framework/understanding/rag/)**: free, official docs. The clearest map of the moving parts (loading, indexing, retrieving, querying) from a framework built around exactly this problem.
- **[Vector Search Explained](https://weaviate.io/blog/vector-search-explained)** (Weaviate): free, blog. Explains HNSW and the ANN speed/accuracy tradeoff without requiring a systems background.
- **[Faiss](https://github.com/facebookresearch/faiss)** (Meta): free, repo + wiki. The library actual similarity search under the hood usually runs on; most people never open it because a managed vector database hides it, but the wiki's tutorial is short and worth the hour.
- **[Vector Search & Approximate Nearest Neighbors (ANN) | FAISS (HNSW & IVF)](https://www.youtube.com/watch?v=chz74Mtd1AA)** (Mustafa Zaki): free, YouTube. Walks through the two ANN index types you'll actually pick between (HNSW, IVF) directly in Faiss.

Prereqs: [Vector embeddings](#vector-embeddings).

### Hybrid search + reranking

Pure vector search misses exact keyword matches; pure keyword search misses paraphrases and synonyms. Hybrid search runs both and combines the scores; reranking then re-scores the combined candidates with a model that reads the query and each candidate together.

- **[Getting Started with Hybrid Search](https://www.pinecone.io/learn/hybrid-search-intro/)** (Pinecone): free, guide. Explains the dense-plus-sparse combination and the tradeoff of weighting one against the other.
- **[Cohere Rerank overview](https://docs.cohere.com/docs/rerank-overview)**: free, official docs. The most commonly reached-for reranking API; read this to see what a second-stage reranker actually buys you over retrieval alone.
- **[Sentence Transformers: Retrieve & Rerank](https://sbert.net/examples/sentence_transformer/applications/retrieve_rerank/README.html)**: free, official docs. A free, self-hosted cross-encoder reranker; most people reach for a paid API without realizing this ships in a library they likely already have installed.
- **[Don't naive-RAG, do hybrid search instead](https://www.youtube.com/watch?v=d_WwEdxyuGs)** (LLMs for Devs): free, YouTube. Compares hybrid-search implementations across Pinecone, Weaviate, and pgvector plus a reranking step, the concrete version of the tradeoff described above.

Prereqs: [Retrieval basics](#retrieval-basics).

### Chunking strategies

How a document gets split before it's embedded. The chunking decision shapes what the retriever can find; too large and irrelevant text dilutes the match, too small and the model loses context it needed.

- **[5 Levels of Text Splitting](https://github.com/FullStackRetrieval-com/RetrievalTutorials/blob/main/tutorials/LevelsOfTextSplitting/5_Levels_Of_Text_Splitting.ipynb)** (Kamradt): free, notebook. The reference nearly everyone in RAG eventually finds; walks character splitting through semantic and agent-driven chunking with runnable code.
- **[LangChain: text splitters](https://docs.langchain.com/oss/python/integrations/splitters)**: free, official docs. Practical reference once you know which strategy you want and just need the class to call.
- **[Is Semantic Chunking Worth the Computational Cost?](https://arxiv.org/abs/2410.13070)**: free, paper. A rare counter-argument paper: tests whether the trendier semantic-chunking approach actually beats naive fixed-size chunks, and finds the answer is often no.
- **[The 5 Levels Of Text Splitting For Retrieval](https://www.youtube.com/watch?v=8OJC21T2SL4)** (Greg Kamradt): free, YouTube. The video version of the notebook above, from the same author; watch first if you'd rather see the levels demoed before reading code.

Prereqs: [Vector embeddings](#vector-embeddings).

### Evaluation and observability

How you know a RAG pipeline is actually retrieving the right context and answering from it honestly, and how you watch it in production once it's live.

- **[Ragas documentation](https://docs.ragas.io/)**: free, docs. The canonical library for RAG-specific metrics (context precision, faithfulness, answer relevance); the concepts page is useful even without adopting the library.
- **[Langfuse documentation](https://langfuse.com/docs)**: free, open source. Tracing and evaluation for a live pipeline: see the actual retrieved chunks and prompts behind any given answer.
- **[Your AI Product Needs Evals](https://hamel.dev/blog/posts/evals/)** (Husain): free, blog post. A more skeptical, practitioner-first take than most vendor docs on what a useful eval loop actually requires.
- **[RAGAS - Evaluate RAG Pipelines](https://www.youtube.com/watch?v=tqRDHaY923o)** (Fahd Mirza): free, YouTube. A direct demo of Ragas (the docs pick above) actually computing context precision and faithfulness on a pipeline, rather than just reading the metric definitions.

Prereqs: [Retrieval basics](#retrieval-basics), [Evaluation](#evaluation).

---

## Phase 9: Fine-tuning

Know when fine-tuning is the right call and can run one.

### When to fine-tune

Prompting and RAG solve most problems more cheaply than fine-tuning does. Fine-tuning is worth the cost only once you have a specific, proven gap neither of those closes, and the labeled data to close it.

- **[Patterns for Building LLM-based Systems & Products](https://eugeneyan.com/writing/llm-patterns/)** (Yan): free, blog post. Places fine-tuning alongside evals, RAG, caching, and guardrails, so you see it as one tool among several rather than the default.
- **[PEFT documentation](https://huggingface.co/docs/peft/)**: free, official docs. The practical landing page for what parameter-efficient fine-tuning actually involves once you decide to do it.

Prereqs: [Evaluation](#evaluation).

### LoRA and adapters

Freezing the pretrained weights and training a small pair of low-rank matrices instead. Reduces trainable parameters and memory enough to fine-tune a large model on a single consumer GPU.

- **[LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)** (Hu et al.): free, paper. The original; the core idea fits in a page even though the paper covers more ground.
- **[PEFT: LoRA](https://huggingface.co/docs/peft/en/package_reference/lora)**: free, official docs. The practical API for applying LoRA to a Hugging Face model in a few lines.
- **[Practical Tips for Finetuning LLMs Using LoRA](https://magazine.sebastianraschka.com/p/practical-tips-for-finetuning-llms)** (Raschka): free, blog post. Answers the questions the paper doesn't: what rank to pick, whether more epochs help, what actually moves the needle in practice.

Prereqs: [When to fine-tune](#when-to-fine-tune).

### Data preparation

Fine-tuning data quality matters more than quantity: a small, carefully curated set of examples regularly beats a large noisy one.

- **[Preprocess](https://huggingface.co/docs/transformers/en/preprocessing)** (Hugging Face): free, official docs. The mechanical side: turning raw text into the tokenized, batched format training actually consumes.
- **[LIMA: Less Is More for Alignment](https://arxiv.org/abs/2305.11206)** (Zhou et al.): free, paper. Fine-tuned a 65B model on just 1,000 curated examples and matched much larger instruction-tuning efforts; the strongest evidence that curation beats volume.

Prereqs: [When to fine-tune](#when-to-fine-tune).

### Open-model serving

Running a fine-tuned or open-weight model yourself instead of calling a provider's API: the inference engine that actually serves requests.

- **[vLLM documentation](https://docs.vllm.ai/en/latest/)**: free, official docs. The throughput-focused serving engine most production open-model deployments are built on.
- **[Ollama documentation](https://docs.ollama.com/)**: free, official docs. The fastest path to running an open-weight model locally with no infrastructure setup.
- **[llama.cpp](https://github.com/ggml-org/llama.cpp)**: free, repo. The low-level inference engine underneath Ollama and many other tools; worth knowing it's there once you need something Ollama doesn't expose.
- **[The State of vLLM](https://www.youtube.com/watch?v=4HPRf9nDZ6Q)** (Anyscale, Ray Summit 2024): free, YouTube, ~35 minutes. A conference talk by core vLLM contributors covering the project's architecture and why it became the default throughput-focused serving engine.

Prereqs: [LoRA and adapters](#lora-and-adapters).

---

## Phase 10: MLOps & Production

Deploy + monitor + iterate on a shipped AI system.

### Deployment (HF Spaces, Modal, Replicate)

Getting a model or pipeline from a notebook into something other people can call. Three different tradeoffs: easiest demo, most control, least setup.

- **[Hugging Face Spaces](https://huggingface.co/docs/hub/spaces)**: free tier, official docs. Fastest way to a public demo; a Gradio or Streamlit app deployed straight from a git repo.
- **[Modal documentation](https://modal.com/docs)**: free tier, official docs. Serverless GPU functions with no Dockerfile or standing instance; pay per second of actual compute.
- **[Replicate documentation](https://replicate.com/docs)**: free to browse, paid to run, official docs. Run someone else's published model behind a stable API without hosting anything yourself; useful before you've decided your own deployment needs custom infrastructure at all.

Prereqs: [Open-model serving](#open-model-serving).

### Monitoring: cost, latency

Once a system is live, the questions change from "does it work" to "is it still working, and what is it costing." Token spend and response time are the two numbers that silently blow up first.

- **[An Introduction to Observability for LLM-based applications using OpenTelemetry](https://opentelemetry.io/blog/2024/llm-observability/)**: free, official blog. The emerging standard (GenAI semantic conventions) for what to actually measure and how to attach cost to a span.
- **[Data Distribution Shifts and Monitoring](https://huyenchip.com/2022/02/07/data-distribution-shifts-and-monitoring.html)** (Huyen): free, blog post. Goes beyond cost and latency to the quieter failure mode: the data your system sees in production drifting away from what it was built on.

Prereqs: [Deployment](#deployment-hf-spaces-modal-replicate).

### Model versioning

Tracking which model (or prompt, or fine-tuned adapter) produced which result, so a regression can be traced back and rolled back.

- **[W&B Registry](https://docs.wandb.ai/models/registry)**: free tier, official docs. Versioning, lineage, and promotion stages for models treated as first-class tracked artifacts.
- **[DVC documentation](https://dvc.org/doc)**: free, official docs. Git-style versioning for models and datasets together; less turnkey than a hosted registry but keeps everything in your own repo.

Prereqs: [Deployment](#deployment-hf-spaces-modal-replicate).

### Observability (Langfuse, Helicone, Phoenix)

Seeing what actually happened inside a request: the full prompt, the retrieved context, the tool calls, and the final output, not just an aggregate latency number.

- **[Langfuse documentation](https://langfuse.com/docs)**: free, open source. Full tracing plus evaluation and prompt management in one open-source platform.
- **[Helicone documentation](https://docs.helicone.ai/getting-started/quick-start)**: free, open source. A proxy-based alternative: point requests at Helicone's gateway and get logging with no SDK changes.
- **[Arize Phoenix](https://arize.com/docs/phoenix)**: free, open source. Runs locally with one `pip install`, no signup or API key required, built on OpenTelemetry; the lowest-friction way to see traces before committing to a hosted platform.

Prereqs: [Evaluation and observability](#evaluation-and-observability).

### Safety and guardrails

Checking inputs and outputs against policy before they reach a user or an action executes: jailbreak attempts, PII leakage, off-topic responses, unsafe tool calls.

- **[OWASP Top 10 for LLM Applications](https://genai.owasp.org/)**: free, official project. The security-first framing of the same failure modes; treats prompt injection and data leakage as first-class vulnerabilities to defend against, not edge cases.
- **[NeMo Guardrails](https://docs.nvidia.com/nemo/guardrails/latest/about/overview.html)** (NVIDIA): free, official docs. Open-source, programmable guardrails with a dialog-flow language, rather than a single input/output filter.

Prereqs: [Structured output](#structured-output).

---

## Phase 11: Staying current

Keep up + find work.

### Weekly newsletters

A standing habit, not a topic to finish once. Pick one and actually read it every week rather than subscribing to five and reading none.

- **[The Batch](https://www.deeplearning.ai/the-batch/)** (DeepLearning.AI): free, weekly. Turns the week's research and industry news into short, plainly explained items; the easiest one to actually keep up with.
- **[Ahead of AI](https://magazine.sebastianraschka.com/)** (Raschka): free, with some paid deep-dive posts. More technical than most news digests: architecture explainers and paper roundups instead of headline summaries.

Prereqs: none.

### Communities

Places where practitioners argue about what actually works, ahead of it showing up in a blog post.

- **[r/MachineLearning](https://www.reddit.com/r/MachineLearning/)**: free. The largest general ML community; research discussion, paper threads, and hiring posts.
- **[r/LocalLLaMA](https://www.reddit.com/r/LocalLLaMA/)**: free. The center of gravity for open-weight models, quantization, and local inference; moves faster than most blogs on this specific topic.
- **[Latent Space](https://www.latent.space/)**: free. Smaller and more focused than the subreddits; the podcast and newsletter both go deep on the AI engineer's specific concerns rather than ML research broadly.

Prereqs: none.

### Papers to track

Not "read everything." A lightweight habit for noticing what's actually worth reading.

- **[Hugging Face Papers](https://huggingface.co/papers)**: free. Daily trending papers ranked by community attention, with discussion threads; the successor to what Papers with Code used to do.
- **[arxiv-sanity-lite](https://github.com/karpathy/arxiv-sanity-lite)** (Karpathy): free, self-hosted tool. Tag papers you like and get recommendations back, instead of scrolling an unranked firehose.

Prereqs: none.

### Landing a job

The role itself is new enough that the job title and the interview loop are both still settling.

- **[The Rise of the AI Engineer](https://www.latent.space/p/ai-engineer)** (swyx): free, blog post. The essay that named the role; worth reading to understand what a hiring manager means by the title before walking into an interview for it.
- **[Introduction to Machine Learning Interviews](https://huyenchip.com/ml-interviews-book/)** (Huyen): free, book. Covers the interview process itself: what roles exist, what each one screens for, and over 200 practice questions.

Prereqs: none.

---

## Tracks

### Default AI engineer path

Phase 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11 in order. Full breadth. ~6-9 months full-time equivalent.

### LLM engineer fast-track

Phase 1, 2 (lighter pass), 3 (lighter pass through PyTorch + training loops), 4 (tokenization, embeddings, transformer architecture), 6, 7, 8, 9, 10, 11. Skips computer vision and classical NLP tasks; goes through classical ML and deep-learning fundamentals at reduced depth. For readers who already program and want the modern LLM/agent/RAG stack fast. ~3-4 months full-time equivalent.

---

## Resource index

Every topic and its picks in one table, for scanning or searching instead of reading phase by phase.

| Topic | Resources |
|---|---|
| [Python](#python) | [Automate the Boring Stuff with Python](https://automatetheboringstuff.com/)<br>[Real Python: Python Basics learning path](https://realpython.com/learning-paths/python-basics/)<br>[Fluent Python, 2nd ed.](https://www.oreilly.com/library/view/fluent-python-2nd/9781492056348/)<br>[Learn Python - Full Course for Beginners](https://www.youtube.com/watch?v=rfscVS0vtbw) |
| [Math](#math) | [Mathematics for Machine Learning](https://mml-book.github.io/)<br>[Essence of Linear Algebra](https://www.3blue1brown.com/topics/linear-algebra)<br>[Introduction to Probability](https://www.edx.org/learn/probability/harvard-university-introduction-to-probability) |
| [Git](#git) | [Pro Git](https://git-scm.com/book/en/v2)<br>[Oh Shit, Git!?!](https://ohshitgit.com/)<br>[Git and GitHub for Beginners - Crash Course](https://www.youtube.com/watch?v=RGOj5yH7evk) |
| [CLI / Linux](#cli--linux) | [The Missing Semester of Your CS Education](https://missing.csail.mit.edu/)<br>[The Linux Command Line](https://linuxcommand.org/tlcl.php) |
| [Supervised learning](#supervised-learning) | [Machine Learning Specialization](https://www.coursera.org/specializations/machine-learning-introduction)<br>[Hands-On Machine Learning with Scikit-Learn, Keras & TensorFlow, 3rd ed.](https://www.oreilly.com/library/view/hands-on-machine-learning/9781098125967/)<br>[StatQuest with Josh Starmer](https://www.youtube.com/@statquest) |
| [Unsupervised learning](#unsupervised-learning) | [An Introduction to Statistical Learning, 2nd ed.](https://www.statlearning.com/)<br>[StatQuest: Clustering and PCA playlists](https://www.youtube.com/@statquest/playlists)<br>[scikit-learn: clustering user guide](https://scikit-learn.org/stable/modules/clustering.html) |
| [Model evaluation](#model-evaluation) | [scikit-learn: model evaluation user guide](https://scikit-learn.org/stable/modules/model_evaluation.html)<br>[Google's Machine Learning Crash Course: Classification](https://developers.google.com/machine-learning/crash-course/classification/video-lecture)<br>[Designing Machine Learning Systems](https://www.oreilly.com/library/view/designing-machine-learning/9781098107956/) |
| [Feature engineering](#feature-engineering) | [Feature Engineering for Machine Learning](https://www.oreilly.com/library/view/feature-engineering-for/9781491953235/)<br>[Kaggle Learn: Feature Engineering](https://www.kaggle.com/learn/feature-engineering)<br>[Applied Predictive Modeling](http://appliedpredictivemodeling.com/)<br>[100 Days of Machine Learning](https://www.youtube.com/playlist?list=PLKnIA16_Rmvbr7zKYQuBfsVkjoLcJgxHH) |
| [scikit-learn](#scikit-learn) | [scikit-learn: user guide](https://scikit-learn.org/stable/user_guide.html)<br>[Hands-On ML](https://www.oreilly.com/library/view/hands-on-machine-learning/9781098125967/)<br>[Machine Learning in Python with scikit-learn](https://www.fun-mooc.fr/en/courses/machine-learning-python-scikit-learn/) |
| [Neural network basics](#neural-network-basics) | [Deep Learning Specialization](https://www.coursera.org/specializations/deep-learning)<br>[3Blue1Brown: Neural Networks playlist](https://www.3blue1brown.com/topics/neural-networks)<br>[Neural Networks: Zero to Hero](https://karpathy.ai/zero-to-hero.html) |
| [PyTorch](#pytorch) | [Deep Learning with PyTorch, 2nd ed.](https://www.manning.com/books/deep-learning-with-pytorch-second-edition)<br>[PyTorch: 60 Minute Blitz](https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html)<br>[Zero to Hero: makemore series](https://karpathy.ai/zero-to-hero.html)<br>[fast.ai Practical Deep Learning](https://course.fast.ai/)<br>[PyTorch Discuss](https://discuss.pytorch.org/) |
| [Training loops](#training-loops) | [PyTorch: Training a Classifier tutorial](https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html)<br>[fast.ai: lessons 1-4](https://course.fast.ai/)<br>[A Recipe for Training Neural Networks](https://karpathy.github.io/2019/04/25/recipe/) |
| [Regularization](#regularization) | [Deep Learning](https://www.deeplearningbook.org/)<br>[Sebastian Raschka's blog: regularization posts](https://sebastianraschka.com/blog/)<br>[Deep Learning Specialization, course 2](https://www.coursera.org/learn/deep-neural-network) |
| [Transfer learning](#transfer-learning) | [PyTorch: Transfer Learning Tutorial](https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html)<br>[fast.ai: chapter on transfer learning](https://github.com/fastai/fastbook)<br>[HuggingFace: Transfer Learning](https://huggingface.co/learn/nlp-course/chapter1/4) |
| [Tokenization](#tokenization) | [Hugging Face LLM Course: tokenizers](https://huggingface.co/learn/llm-course/chapter6/1)<br>[Let's build the GPT Tokenizer](https://www.youtube.com/watch?v=zduSFxRajkE)<br>[tiktoken](https://github.com/openai/tiktoken) |
| [Embeddings](#embeddings) | [The Illustrated Word2vec](https://jalammar.github.io/illustrated-word2vec/)<br>[On word embeddings, Part 1](https://www.ruder.io/word-embeddings-1/)<br>[OpenAI: Vector embeddings guide](https://platform.openai.com/docs/guides/embeddings)<br>[Word Embedding and Word2Vec, Clearly Explained!!!](https://www.youtube.com/watch?v=viZrOnJclY0) |
| [Transformer architecture](#transformer-architecture) | [Attention Is All You Need](https://arxiv.org/abs/1706.03762)<br>[The Annotated Transformer](https://nlp.seas.harvard.edu/annotated-transformer/)<br>[Formal Algorithms for Transformers](https://arxiv.org/abs/2207.09238)<br>[Attention in transformers, step-by-step](https://www.youtube.com/watch?v=eMlx5fFNoYc) |
| [Classical NLP tasks](#classical-nlp-tasks) | [Advanced NLP with spaCy](https://course.spacy.io/en/)<br>[Speech and Language Processing, 3rd ed. draft](https://web.stanford.edu/~jurafsky/slp3/)<br>[spaCy 101](https://spacy.io/usage/spacy-101)<br>[Natural Language Processing with spaCy & Python: Course for Beginners](https://www.youtube.com/watch?v=dIUTsFT2MeQ) |
| [Image basics](#image-basics) | [Practical Deep Learning for Coders](https://course.fast.ai/)<br>[Torchvision: transforms](https://docs.pytorch.org/vision/stable/transforms.html)<br>[But what is a convolution?](https://www.3blue1brown.com/lessons/convolutions/) |
| [Convnets](#convnets) | [CS231n: Convolutional Neural Networks](https://cs231n.github.io/convolutional-networks/)<br>[A Guide to Convolution Arithmetic for Deep Learning](https://arxiv.org/abs/1603.07285)<br>[CS231n Winter 2016: Lecture 7: Convolutional Neural Networks](https://www.youtube.com/watch?v=LxfUGhug-iQ) |
| [Object detection](#object-detection) | [Ultralytics YOLO: Object Detection](https://docs.ultralytics.com/tasks/detect)<br>[TorchVision Object Detection Finetuning Tutorial](https://docs.pytorch.org/tutorials/intermediate/torchvision_tutorial.html)<br>[mAP (mean Average Precision) for Object Detection](https://naokishibuya.github.io/blog/2022-05-27-object-detection-mean-average-precision/) |
| [Vision transformers](#vision-transformers) | [An Image is Worth 16x16 Words](https://arxiv.org/abs/2010.11929)<br>[Hugging Face: Vision Transformer (ViT)](https://huggingface.co/docs/transformers/model_doc/vit)<br>[Hugging Face Computer Vision Course: ViT for image classification](https://huggingface.co/learn/computer-vision-course/unit3/vision-transformers/vision-transformers-for-image-classification)<br>[ViT (Vision Transformer) - An Image Is Worth 16x16 Words (Paper Explained)](https://www.youtube.com/watch?v=8phM16htKbU) |
| [How LLMs work](#how-llms-work) | [Intro to Large Language Models](https://www.youtube.com/watch?v=zjkBMFhNj_g)<br>[Let's build GPT: from scratch, in code, spelled out](https://www.youtube.com/watch?v=kCc8FmEb1nY)<br>[The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) |
| [Prompt engineering](#prompt-engineering) | [Anthropic: Prompt Engineering Overview](https://docs.claude.com/en/docs/build-with-claude/prompt-engineering/overview)<br>[OpenAI: Prompt engineering](https://platform.openai.com/docs/guides/prompt-engineering)<br>[Learn Prompting](https://learnprompting.org/)<br>[Simon Willison's blog: llms tag](https://simonwillison.net/tags/llms/)<br>[AI prompt engineering: A deep dive](https://www.youtube.com/watch?v=T9aRN5JkmL8) |
| [Structured output](#structured-output) | [Instructor library docs](https://python.useinstructor.com/)<br>[OpenAI: Structured Outputs](https://platform.openai.com/docs/guides/structured-outputs)<br>[Pydantic AI docs](https://ai.pydantic.dev/)<br>[OpenAI DevDay 2024: Structured outputs for reliable applications](https://www.youtube.com/watch?v=kE4BkATIl9c) |
| [Evaluation](#evaluation) | [Ragas documentation](https://docs.ragas.io/)<br>[Eugene Yan: Evals blog series](https://eugeneyan.com/writing/evals/)<br>[OpenAI Evals](https://github.com/openai/evals)<br>[How to Systematically Setup LLM Evals](https://www.youtube.com/watch?v=a3SMraZWNNs) |
| [Provider SDKs](#provider-sdks) | [OpenAI Python SDK docs](https://platform.openai.com/docs/libraries)<br>[Anthropic Python SDK docs](https://docs.claude.com/en/api/client-sdks)<br>[Google Gen AI SDK docs](https://ai.google.dev/gemini-api/docs/quickstart)<br>[LiteLLM docs](https://docs.litellm.ai/) |
| [Tool use](#tool-use) | [Claude: Tool use overview](https://platform.claude.com/docs/en/agents-and-tools/tool-use/overview)<br>[OpenAI: Function calling guide](https://platform.openai.com/docs/guides/function-calling)<br>[PydanticAI: Tools](https://ai.pydantic.dev/tools/)<br>[Function Calling Demo at OpenAI DevDay](https://www.youtube.com/watch?v=D1Oqt7vig9k) |
| [ReAct pattern](#react-pattern) | [ReAct: Synergizing Reasoning and Acting in Language Models](https://arxiv.org/abs/2210.03629)<br>[LLM Powered Autonomous Agents](https://lilianweng.github.io/posts/2023-06-23-agent/) |
| [Multi-agent collaboration](#multi-agent-collaboration) | [Building Effective Agents](https://www.anthropic.com/engineering/building-effective-agents)<br>[How we built our multi-agent research system](https://www.anthropic.com/engineering/multi-agent-research-system)<br>[CrewAI documentation](https://docs.crewai.com/)<br>[Anthropic: How to Build Multi-Agent Systems](https://www.youtube.com/watch?v=0HljLKVdtjo) |
| [Framework survey (LangGraph, CrewAI, OpenAI Agents SDK, PydanticAI)](#framework-survey-langgraph-crewai-openai-agents-sdk-pydanticai) | [LangGraph](https://docs.langchain.com/oss/python/langgraph/overview)<br>[CrewAI documentation](https://docs.crewai.com/)<br>[OpenAI Agents SDK](https://openai.github.io/openai-agents-python/)<br>[PydanticAI documentation](https://ai.pydantic.dev/) |
| [Vector embeddings](#vector-embeddings) | [Pinecone: Learning Center](https://www.pinecone.io/learn/)<br>[What are embeddings?](https://vickiboykis.com/what_are_embeddings/)<br>[Cohere: Embeddings documentation](https://docs.cohere.com/docs/embeddings)<br>[OpenAI Embeddings and Vector Databases Crash Course](https://www.youtube.com/watch?v=ySus5ZS0b94) |
| [Retrieval basics](#retrieval-basics) | [LlamaIndex: Introduction to RAG](https://developers.llamaindex.ai/python/framework/understanding/rag/)<br>[Vector Search Explained](https://weaviate.io/blog/vector-search-explained)<br>[Faiss](https://github.com/facebookresearch/faiss)<br>[Vector Search & Approximate Nearest Neighbors (ANN) \| FAISS (HNSW & IVF)](https://www.youtube.com/watch?v=chz74Mtd1AA) |
| [Hybrid search + reranking](#hybrid-search--reranking) | [Getting Started with Hybrid Search](https://www.pinecone.io/learn/hybrid-search-intro/)<br>[Cohere Rerank overview](https://docs.cohere.com/docs/rerank-overview)<br>[Sentence Transformers: Retrieve & Rerank](https://sbert.net/examples/sentence_transformer/applications/retrieve_rerank/README.html)<br>[Don't naive-RAG, do hybrid search instead](https://www.youtube.com/watch?v=d_WwEdxyuGs) |
| [Chunking strategies](#chunking-strategies) | [5 Levels of Text Splitting](https://github.com/FullStackRetrieval-com/RetrievalTutorials/blob/main/tutorials/LevelsOfTextSplitting/5_Levels_Of_Text_Splitting.ipynb)<br>[LangChain: text splitters](https://docs.langchain.com/oss/python/integrations/splitters)<br>[Is Semantic Chunking Worth the Computational Cost?](https://arxiv.org/abs/2410.13070)<br>[The 5 Levels Of Text Splitting For Retrieval](https://www.youtube.com/watch?v=8OJC21T2SL4) |
| [Evaluation and observability](#evaluation-and-observability) | [Ragas documentation](https://docs.ragas.io/)<br>[Langfuse documentation](https://langfuse.com/docs)<br>[Your AI Product Needs Evals](https://hamel.dev/blog/posts/evals/)<br>[RAGAS - Evaluate RAG Pipelines](https://www.youtube.com/watch?v=tqRDHaY923o) |
| [When to fine-tune](#when-to-fine-tune) | [Patterns for Building LLM-based Systems & Products](https://eugeneyan.com/writing/llm-patterns/)<br>[PEFT documentation](https://huggingface.co/docs/peft/) |
| [LoRA and adapters](#lora-and-adapters) | [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)<br>[PEFT: LoRA](https://huggingface.co/docs/peft/en/package_reference/lora)<br>[Practical Tips for Finetuning LLMs Using LoRA](https://magazine.sebastianraschka.com/p/practical-tips-for-finetuning-llms) |
| [Data preparation](#data-preparation) | [Preprocess](https://huggingface.co/docs/transformers/en/preprocessing)<br>[LIMA: Less Is More for Alignment](https://arxiv.org/abs/2305.11206) |
| [Open-model serving](#open-model-serving) | [vLLM documentation](https://docs.vllm.ai/en/latest/)<br>[Ollama documentation](https://docs.ollama.com/)<br>[llama.cpp](https://github.com/ggml-org/llama.cpp)<br>[The State of vLLM](https://www.youtube.com/watch?v=4HPRf9nDZ6Q) |
| [Deployment (HF Spaces, Modal, Replicate)](#deployment-hf-spaces-modal-replicate) | [Hugging Face Spaces](https://huggingface.co/docs/hub/spaces)<br>[Modal documentation](https://modal.com/docs)<br>[Replicate documentation](https://replicate.com/docs) |
| [Monitoring: cost, latency](#monitoring-cost-latency) | [An Introduction to Observability for LLM-based applications using OpenTelemetry](https://opentelemetry.io/blog/2024/llm-observability/)<br>[Data Distribution Shifts and Monitoring](https://huyenchip.com/2022/02/07/data-distribution-shifts-and-monitoring.html) |
| [Model versioning](#model-versioning) | [W&B Registry](https://docs.wandb.ai/models/registry)<br>[DVC documentation](https://dvc.org/doc) |
| [Observability (Langfuse, Helicone, Phoenix)](#observability-langfuse-helicone-phoenix) | [Langfuse documentation](https://langfuse.com/docs)<br>[Helicone documentation](https://docs.helicone.ai/getting-started/quick-start)<br>[Arize Phoenix](https://arize.com/docs/phoenix) |
| [Safety and guardrails](#safety-and-guardrails) | [OWASP Top 10 for LLM Applications](https://genai.owasp.org/)<br>[NeMo Guardrails](https://docs.nvidia.com/nemo/guardrails/latest/about/overview.html) |
| [Weekly newsletters](#weekly-newsletters) | [The Batch](https://www.deeplearning.ai/the-batch/)<br>[Ahead of AI](https://magazine.sebastianraschka.com/) |
| [Communities](#communities) | [r/MachineLearning](https://www.reddit.com/r/MachineLearning/)<br>[r/LocalLLaMA](https://www.reddit.com/r/LocalLLaMA/)<br>[Latent Space](https://www.latent.space/) |
| [Papers to track](#papers-to-track) | [Hugging Face Papers](https://huggingface.co/papers)<br>[arxiv-sanity-lite](https://github.com/karpathy/arxiv-sanity-lite) |
| [Landing a job](#landing-a-job) | [The Rise of the AI Engineer](https://www.latent.space/p/ai-engineer)<br>[Introduction to Machine Learning Interviews](https://huyenchip.com/ml-interviews-book/) |

---

## Contributing

Fixes for dead links, proposals for a better pick, missing topics you notice: see [`CONTRIBUTING.md`](CONTRIBUTING.md). Two rules:

1. One pick per resource type per topic. If the current pick is genuinely worse than yours, PR to replace it, don't PR to add a second.
2. Every new pick needs a one-sentence why it beats the current one.

## License

Content licensed [CC BY 4.0](LICENSE). Free to use, share, adapt with attribution.
