# Contributing to ai-engineer-roadmap

Everything lives in [`README.md`](README.md). There are no per-topic files
to navigate; a PR either edits a section of the README or fixes something
around it (CI, templates).

## Ways to contribute

- **Fix a dead link.** Open an issue or a PR directly; a one-line fix
  doesn't need discussion first.
- **Propose a better pick.** Replace an existing resource with one you
  think is better. State why in the PR description.
- **Fill in a missing topic.** If a heading is marked `_TBD, targeted for
  vX.Y_`, populate it following the rules below. If none are currently
  marked TBD, propose a new topic instead (open an issue first).
- **Report an issue** with a topic's content: wrong info, outdated
  material, a resource that no longer fits.

## The rules (R1-R6)

- **R1.** One canonical pick per resource type per topic: one course, one
  book, one video, at most one repo, at most one paper. Not five courses
  on the same thing. If you think the current pick is worse than yours,
  PR to replace it, don't PR to add a second.
- **R2.** Free where free exists. Paid picks are labeled `$` with a
  one-sentence reason they're worth paying for.
- **R3.** Links must resolve. CI checks weekly; a PR that adds a dead link
  will fail the same check.
- **R4.** Short prose. Two sentences of context per topic, then the
  resource list. No filler.
- **R5.** No template stubs. Never add "What it is / Why you need this /
  Estimated time / When you're done you can" style sections. Just a
  heading, context, the resource list, and a prereqs line.
- **R6.** Popular AND underrated. Every topic should have at least one
  pick that isn't the first thing that shows up in a search, tagged
  `` `[underrated]` `` inline.

## Every new pick needs

One sentence in the PR description: why this resource beats what's
already there, or why it fills a real gap. "Here's another good course on
the same topic" without that justification will get a request to trim
rather than add, per R1.

## Style

No em dashes or en dashes anywhere; use commas, periods, or colons.
Hyphens only for real technical terms (package names, established compound
nouns). No emojis. Conventional commit messages.
