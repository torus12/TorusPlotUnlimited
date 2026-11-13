# Local-model licence research (Ollama backend candidates)

Research date: 2026-11-13. Goal: find local models under **Apache-2.0 or MIT only** for
TorusPlot's optional Ollama backend - one text model (FigureSpec / analysis-request JSON
generation) and one vision model (figure critique, "match this figure's look").

## Read this first: what could and could not be verified

| Source | Reachable from this session? | Result |
|---|---|---|
| `ollama.com/library/<model>` (the official Ollama cards) | **NO** - blocked by the session's network policy. `curl` returned "CONNECT tunnel failed, response 403" for all 13 pages; the web-fetch tool returned "ENOTFOUND ollama.com". | **No Ollama card was read.** Nothing below is "as stated on the Ollama card". |
| `huggingface.co` (upstream model cards) | **NO** - same block ("ENOTFOUND huggingface.co"). | No Hugging Face model card was read. |
| Vendor GitHub repositories (`raw.githubusercontent.com`, `git clone`) | **YES** | README and LICENSE files of the vendors' own repositories were read. These are official upstream vendor pages, and are the only "verified" evidence in this file. |

Consequences:

1. "Verified" below means **verified from the vendor's own GitHub repository** (file and
   commit given), not from the Ollama card.
2. The mapping "Ollama tag X = upstream checkpoint Y" could not be checked for any model
   and is always marked *from memory*.
3. Before shipping a default, someone with normal internet access must open the two
   Ollama pages named in the recommendation and/or run `ollama show <tag> --license` on a
   machine with the model pulled, and compare with this file. To let a future session do
   it, add `ollama.com` and `huggingface.co` to the session's allowed network domains.

Ollama itself (the runtime) is **MIT** - verified from `LICENSE` in
https://github.com/ollama/ollama (`main`, read 2026-11-13: "MIT License, Copyright (c)
Ollama").

## Recommendation

| Use | Recommended tag | Licence | Evidence level |
|---|---|---|---|
| **Text / JSON generation** | **`qwen2.5:7b`** | Apache-2.0 | Verified from vendor README (all Qwen2.5 sizes except 3B and 72B). Ollama card not read. |
| **Vision (figure critique)** | **`granite3.2-vision`** (2B) | Apache-2.0 | Verified from IBM's vendor repository README + LICENSE (all Granite Vision models). Ollama card not read. |

- Text URL read: https://github.com/QwenLM/Qwen2.5 (now redirects to QwenLM/Qwen3),
  `README.md` at commit `2abc0f3243e20cf45b6a3ab60308db701c0c3111`. Page to confirm:
  https://ollama.com/library/qwen2.5
- Vision URL read: https://github.com/ibm-granite/granite-vision-models (`README.md`,
  `LICENSE`, `main`). Page to confirm: https://ollama.com/library/granite3.2-vision
- Verified alternates for text: `qwen3` (Apache-2.0, all open-weight sizes),
  `granite3.3` (Apache-2.0, 2b / 8b).
- Never default to: `qwen2.5:3b`, `qwen2.5:72b` (not Apache per the vendor), `llava`
  (checkpoints inherit base-model licences), `deepseek-r1:8b` / `:70b` (Llama-derived).
- `qwen2.5-coder:7b` is probably the stronger choice for code-like structured output but
  its licence is **unverified** here; do not make it the default until confirmed.
- The UI should let the user pick any model they have pulled, while TorusPlot's own
  defaults, docs and auto-pull suggestions name only tags on a verified allow-list.

## A. Verified from an official vendor page (GitHub)

| Model tag (Ollama) | Licence as stated by the vendor | Exactly what was read | Image input? |
|---|---|---|---|
| `qwen2.5` (7b; also 0.5b, 1.5b, 14b, 32b) | **Apache-2.0**. Vendor README licence section: all open-source Qwen2.5 models *except the 3B and 72B variants* are Apache 2.0; licence files live in the respective Hugging Face repositories. So **3b and 72b are NOT Apache-2.0** (from memory: Qwen Research licence for 3B, Qwen licence for 72B). | https://github.com/QwenLM/Qwen2.5 - `README.md` at commit `2abc0f3243e2` (the last Qwen2.5-era README, parent of the 2025-04-28 "add qwen3" commit); same README lists sizes 0.5B / 1.5B / 3B / 7B / 14B / 32B / 72B. | No (text only). |
| `qwen3` | **Apache-2.0**. Vendor README: "All our open-weight models are licensed under Apache 2.0." | https://github.com/QwenLM/Qwen3 - `README.md` on `main` (HEAD `7a2f61ffc7a2`, read 2026-11-13). README itself shows `ollama run qwen3:8b`. | No (text only; the vision line is the separate Qwen3-VL family). |
| `granite3.3` (2b, 8b) | **Apache-2.0**. Vendor README: all Granite 3.3 language models are publicly released under an Apache 2.0 licence for both research and commercial use. Repository `LICENSE` is the Apache License 2.0 text. | https://github.com/ibm-granite/granite-3.3-language-models - `README.md`, `LICENSE` (`main`). Lists granite-3.3-2b / 8b, base and instruct. | No (text only). |
| `granite3.2-vision` (2b) | **Apache-2.0**. Vendor README licence section: "All Granite Vision Models are distributed under Apache 2.0 license." Repository `LICENSE` is the Apache License 2.0 text. | https://github.com/ibm-granite/granite-vision-models - `README.md`, `LICENSE` (`main`). README names granite-vision-3.2-2b, 3.3-2b, 3.1-2b-preview and 4.1-4b. | **Yes.** Described by IBM as vision-language models for document understanding incl. charts and tables and image-to-text - a good fit for critiquing plots. |
| `deepseek-r1` | **MIT** for "this code repository and the model weights" (README section 7; `LICENSE` is MIT, Copyright (c) 2023 DeepSeek). **Caveat stated by the vendor:** the distilled checkpoints derive from other models - Distill-Qwen 1.5B / 7B / 14B / 32B from Qwen2.5 (Apache-2.0 base); Distill-Llama-8B from Llama 3.1 (Llama 3.1 licence); Distill-Llama-70B from Llama 3.3 (Llama 3.3 licence). | https://github.com/deepseek-ai/DeepSeek-R1 - `README.md`, `LICENSE` (`main`). | No (text only). |
| `minicpm-v` | **Apache-2.0**. Vendor docs "Model License": the MiniCPM-o/V model weights and code are open-sourced under Apache-2.0 (an optional registration questionnaire is requested, not required). Repository `LICENSE` is Apache-2.0. | https://github.com/OpenBMB/MiniCPM-V - `README.md`, `docs/project_en.md`, `LICENSE` (`main`). | **Yes.** |
| `moondream` | Repository licence **Apache-2.0** (`LICENSE`). The README contains no separate statement about the weights. | https://github.com/vikhyat/moondream - `LICENSE`, `README.md` (`main`). | **Yes** (tiny VLM: captioning, visual question answering, detection). |
| `llava` | **Mixed - not cleanly Apache/MIT.** Code licence badge and `LICENSE`: Apache-2.0. But the README's "Usage and License Notices" says the project uses datasets and checkpoints subject to their original licences, including OpenAI terms of use for the dataset and the base language model's licence for checkpoints (it names the Llama community licence for LLaMA-2 and Vicuna v1.5). | https://github.com/haotian-liu/LLaVA - `README.md`, `LICENSE` (`main`). | **Yes** - but **do not use as a default.** |
| `mistral` (7B) | **Only indirectly verified.** The vendor's inference-library README lists Mistral 7B among downloadable open-weight models and singles out only Codestral 22B (Mistral AI Non-Production Licence) and Mistral Large (Mistral AI Research Licence) as having custom non-commercial licences; the library itself is Apache 2.0. It does not contain an explicit "Mistral 7B is Apache-2.0" sentence. | https://github.com/mistralai/mistral-inference - `README.md` (`main`). | No (text only). |

## B. Unverified (from memory only - no official page could be read)

Treat every line in this section as a hypothesis to confirm, not as a finding.

| Model tag (Ollama) | Licence I expect (from memory) | Why unverified | Image input? (from memory) |
|---|---|---|---|
| `qwen2.5-coder` (7b) | Apache-2.0 for 0.5B / 1.5B / 7B / 14B / 32B; the **3B is under the non-commercial Qwen Research licence**. | The vendor GitHub README (QwenLM/Qwen2.5-Coder, now redirecting to Qwen3-Coder) has never contained a licence statement (README history searched); the licence is only on the Hugging Face / Ollama cards, which were unreachable. | No. |
| `qwen2.5vl` | 7B and 32B Apache-2.0; **3B Qwen Research licence (non-commercial); 72B Qwen licence**. | The Qwen2.5-VL README has no licence statement. (Its Qwen2-VL-era README, commit `a4b6507`, did state: 2B and 7B Apache 2.0, 72B Qwen licence - that is the *previous* generation, shown only as a pattern.) Hugging Face / Ollama cards unreachable. | Yes. |
| `mistral` (7B) | Apache-2.0 (Mistral 7B v0.1 - v0.3; the Ollama `mistral` tag is 7B v0.3). | See section A: only indirect vendor evidence; model card unreachable. | No. |
| `phi4` (14B) | MIT. | Microsoft publishes the licence on the Hugging Face / Azure model card (unreachable). The GitHub "PhiCookBook" repository is MIT-licensed, but that licence covers the cookbook, and its README only calls Phi "open source AI models" - not evidence for the weights. | No (`phi4` is text only; multimodal variants are separate models). |
| `phi3.5` (3.8B mini) | MIT. | Same as above. | No (the vision variant is a separate model). |
| `deepseek-r1` tag mapping | `1.5b`, `7b`, `14b`, `32b` = Qwen2.5-based distills (MIT weights on an Apache-2.0 base); `8b` and `70b` = **Llama-based distills (Llama licence terms apply - exclude)**; `671b` = the full MIT model. | Vendor licence text is verified (section A); the tag-to-checkpoint mapping lives on the Ollama page. | No. |
| `minicpm-v` tag mapping | The Ollama `minicpm-v` tag carries MiniCPM-V 2.6 (8B). Older MiniCPM-V releases were originally published under a custom "MiniCPM Model License" with conditions on commercial use; the vendor's *current* statement (section A) says Apache-2.0 for weights and code. | Whether the licence layer shipped inside the Ollama blob is the old custom text or Apache-2.0 must be checked with `ollama show minicpm-v --license`. | Yes. |
| `moondream` weights | Apache-2.0 (Ollama `moondream` = moondream2, about 1.8B). | Repository licence verified; a weights-specific statement is on Hugging Face only. | Yes. |
| `llava` tag mapping | Ollama `llava` = LLaVA 1.6 with different base LLMs per size (Mistral-7B, Vicuna-13B, a 34B base). | Not needed for a decision: section A already shows the family is not uniformly Apache/MIT. | Yes. |
| `granite3.2-vision`, `granite3.3`, `qwen2.5`, `qwen3` on Ollama | The Ollama cards should show Apache 2.0, matching section A. | Ollama cards unreachable. | as in section A |

## C. Screening result against the "Apache-2.0 or MIT only" rule

| Model tag | Verdict | Note |
|---|---|---|
| `qwen2.5:7b` | **PASS (vendor-verified)** | Pin the size; never `:3b` or `:72b`. |
| `qwen3` | **PASS (vendor-verified)** | Verified alternate text model. |
| `granite3.3` | **PASS (vendor-verified)** | Verified alternate text model. |
| `granite3.2-vision` | **PASS (vendor-verified)** | Recommended vision model. |
| `deepseek-r1` | PASS only for `1.5b` / `7b` / `14b` / `32b` (and the full model), pending tag-mapping check | **FAIL for `8b` and `70b`** (Llama licence). Reasoning output must be stripped before JSON parsing. |
| `minicpm-v` | PASS per current vendor statement, pending check of the licence shipped in the Ollama tag | Backup vision model. |
| `moondream` | PASS (repository Apache-2.0), weights statement pending | Smallest backup vision model. |
| `mistral` (7B) | PENDING | Expected Apache-2.0; confirm on the card. |
| `phi4`, `phi3.5` | PENDING | Expected MIT; confirm on the card. |
| `qwen2.5-coder:7b` | PENDING | Expected Apache-2.0; `:3b` expected to FAIL. |
| `qwen2.5vl` | PENDING | `:7b` / `:32b` expected Apache-2.0; `:3b` and `:72b` expected to FAIL. |
| `llava` | **FAIL as a default** | Checkpoints inherit base-model and dataset terms per the vendor's own notice. |

## D. How to close the open items (5 minutes with normal internet)

1. Open https://ollama.com/library/qwen2.5 and https://ollama.com/library/granite3.2-vision
   and record the licence shown on each card and each size tag.
2. On a machine with Ollama installed, run `ollama show qwen2.5:7b --license` and
   `ollama show granite3.2-vision --license`; the output is the licence text packaged with
   that exact tag. Do the same for any other tag before adding it to the allow-list.
3. For the PENDING rows, read the Hugging Face model cards (`Qwen/Qwen2.5-Coder-7B-Instruct`,
   `Qwen/Qwen2.5-VL-7B-Instruct`, `mistralai/Mistral-7B-Instruct-v0.3`, `microsoft/phi-4`,
   `microsoft/Phi-3.5-mini-instruct`, `vikhyatk/moondream2`, `openbmb/MiniCPM-V-2_6`).
4. Record the date, URL and exact wording next to each tag in this file, then move the
   row from section B to section A.

Notes: a model licence governs the *weights*; TorusPlot does not bundle any weights - the
user pulls them with Ollama - but our defaults and documentation should still point only
at permissively licensed models. This file is an engineering record, not legal advice.
