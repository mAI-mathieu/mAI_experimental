# SoL H3 third-party notices

`sol_refiner/models.py` and `sol_refiner/decoder.py` adapt the Apache-2.0
exceptions in Olli Sorjonen's
[ComfyUI-Olm-SoL-Refiner](https://github.com/o-l-l-i/ComfyUI-Olm-SoL-Refiner).
The exceptions explicitly include its ComfyUI integration changes. Original
authorship and license headers are retained in both files. mAI changes rename
the attention registration, defer compilation, and integrate independent
backend probing. No restricted node, runtime, or loading source is distributed.

The video transformer is copyright 2025 The Lightricks team and The HuggingFace
Team; the decoder is copyright 2026 Lightricks and The HuggingFace Team. The
ComfyUI adaptations are by Olli Sorjonen (2026). Both files derive from
Hugging Face Diffusers revision
`e0abab83b5df05de9e7abd788643c1a7c1e42e28`:

- [transformer_ltx2.py](https://github.com/huggingface/diffusers/blob/e0abab83b5df05de9e7abd788643c1a7c1e42e28/src/diffusers/models/transformers/transformer_ltx2.py)
- [ltx2_diffusion_decoder.py](https://github.com/huggingface/diffusers/blob/e0abab83b5df05de9e7abd788643c1a7c1e42e28/src/diffusers/models/autoencoders/ltx2_diffusion_decoder.py)

The Apache License is included in [licenses/Apache-2.0.txt](licenses/Apache-2.0.txt).
The referenced Diffusers revision contains no NOTICE file.

The independently written loader, runtime, pipeline, paths, and node wrappers
study the behavior of Olli's integration and
[NVIDIA's SoL H3 reference](https://github.com/NVlabs/Sana/tree/sol-engine/models/sol-refiner/MiniMax-H3).
These modules do not copy that integration source or NVIDIA's pipeline source.
They import the Lightricks LTX-2.5 implementations from Diffusers and text
components from Transformers. The math and tensor conventions follow NVIDIA's
released one-step model.

No weights are distributed or automatically downloaded. Code licenses do not
grant rights to the model weights. Consult the publisher for the applicable
terms of the [SoL H3 package](https://huggingface.co/Efficient-Large-Model/SoL-Refiner-LTX-2.5-for-MiniMax-H3)
and its LTX-2.5/Gemma components.
