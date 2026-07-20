# Llama-3-MemQ-v9

**Built with Meta Llama 3.** This is a merged fine-tune of
[`Meta-Llama-3-8B-Instruct`](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct)
for generating MemQ natural-language query plans. It is the v9 model used for
the best reported answer-level results in this repository.

Download the exact 14.97 GiB release with:

```bash
scripts/download_weights.sh
```

The downloader uses unauthenticated, public Backblaze B2 URLs and verifies all
files with SHA-256. No credentials are embedded in this repository or script.

## License and attribution

The weights are a derivative of Meta Llama 3 and are distributed under the
[Meta Llama 3 Community License](https://huggingface.co/meta-llama/Meta-Llama-3-8B/blob/main/LICENSE),
including its Acceptable Use Policy. The required notice is:

> Meta Llama 3 is licensed under the Meta Llama 3 Community License, Copyright
> © Meta Platforms, Inc. All Rights Reserved.

The model name begins with `Llama-3` and this document provides the required
“Built with Meta Llama 3” attribution. Downloaders must comply with the base
model license and the terms governing WebQSP and ComplexWebQuestions.

## Intended use and limitations

This is research code and a research fine-tune, not a factual QA system. It
produces plans that are resolved against a local Freebase/Virtuoso endpoint;
entity ambiguity, incomplete data, and incorrect generated plans can produce
wrong answers. Do not use it for high-stakes decisions.
