## What is STITCH?

**Ongoing, come back in a few days!**

**STITCH** (_**S**mall **T**weaks **I**mpacting **T**ask **C**ompletion **H**andling_) is a small dataset and mini-framework to run experiments on how seemingly innocuous prompt changes can impact LLM reading comprehension. The aim of this is to provide some numbers to guide our understanding of prompt engineering, understand how models use context, and help make Q/A prompts less alchemical in nature!

## What are we evaluating exactly?



## How to run STITCH benchmarking?

STITCH uses a .yaml file to define which models, prompt formats and datasets you wish to evaluate. You may check `default_run.yaml` for an example YAML file, or stay tuned until this section is updated.


## The STITCH dataset

STITCH is composed of three main subsets, and two "control" subsets, which are questions that so-called `Frontier LLMs`, such as GPT-4 and Claude 3 Opus, can answer without the need for context. The three main subsets are from various domains: `bsard` concerns Belgian Law (in French), `biomrc` biomedical research papers, and `proxima` auto-generated academic reports on the future colonisation of Proxima Centauri b and its technical and social implications.

Each of these susbet is composed of 54 questions, and they all aim to evaluate slightly different things: 
- `bsard` requires both reasoning on non-English (French) documents and combining multiple relevant documents to answer a question.
- `biomrc` requires reasoning on a single mid-sized relevant passage to answer a tricky question about experimental settings.
- `proxima` requires reading a long document to find the answer in the relevant section.


| Name           | Lang      | Relevant Document Type       | Num entries | Known to frontier LLMs | Domain     | Source                                                                             |
| -------------- | --------- | ---------------------------- | ----------- | ---------------------- | ---------- | ---------------------------------------------------------------------------------- |
| bsard          | 🇫🇷      | multiple short relevant docs | 54          | ❌                      | Legal      | [maastrichtlawtech/bsard](https://huggingface.co/datasets/maastrichtlawtech/bsard) |
| bsard_control  | 🇫🇷      | multiple short relevant docs | 54          | ✅                      | Legal      | [maastrichtlawtech/bsard](https://huggingface.co/datasets/maastrichtlawtech/bsard) |
| biomrc         | 🇬🇧/🇺🇸 | single short relevant doc    | 54          | ❌                      | Biomedical | [biomrc](https://huggingface.co/datasets/biomrc)                                   |
| biomrc_control | 🇬🇧/🇺🇸 | single short relevant doc    | 54          | ✅                      | Biomedical | [biomrc](https://huggingface.co/datasets/biomrc)                                   |
| proxima        | 🇬🇧/🇺🇸 | long document                | 54          | ❌                      | Sci-fi     | Synthetic                                                                          |
|                |           |                              |             |                        |            |                                                                                    |


## Results

TBD