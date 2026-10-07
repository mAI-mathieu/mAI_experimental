# mAI experimental ComfyUI nodes

Small experimental nodes for ComfyUI.

## mAI SoL H3 Loader and mAI SoL H3 Refiner

Native one-step generative video refinement/upscaling using
[NVIDIA SoL H3](https://github.com/NVlabs/Sana/tree/sol-engine/models/sol-refiner/MiniMax-H3).
These nodes appear under `mAI / Image`, registered as `MAISoLH3Loader` and
`MAISoLH3Refiner`. Connect ordinary decoded video `IMAGE` frames; LTX latents
and MiniMax generation are unnecessary. Intended especially for MiniMax H3
outputs, but other generators' frames can be supplied. Generative refinement
can change source details and identity. Audio bypasses these nodes.

Target: Linux, NVIDIA CUDA, a GPU with BF16 support, substantial VRAM and system
RAM. No subprocess, temporary video, image sequence, or model download runs
inside these nodes. Existing nodes still import without SoL dependencies.

Download the complete approximately 71 GB
[checkpoint package](https://huggingface.co/Efficient-Large-Model/SoL-Refiner-LTX-2.5-for-MiniMax-H3)
yourself and preserve this structure:

```text
ComfyUI/models/sol_refiner/SoL-Refiner-LTX-2.5-for-MiniMax-H3/
  model_index.json
  connectors/       diffusion_decoder/   latent_upsampler/
  scheduler/        text_encoder/        tokenizer/
  transformer/      vae/
```

The refiner package is separate from MiniMax H3 generation checkpoints. If it
has not been downloaded, run this from your ComfyUI directory in a terminal
with the Hugging Face `hf` CLI available:

```bash
hf download Efficient-Large-Model/SoL-Refiner-LTX-2.5-for-MiniMax-H3 \
  --local-dir models/sol_refiner/SoL-Refiner-LTX-2.5-for-MiniMax-H3
```

This explicitly downloads the entire package; see the
[Hugging Face CLI documentation](https://huggingface.co/docs/huggingface_hub/guides/cli#download-to-a-local-folder).
Wait for the download to finish, then refresh/restart ComfyUI and select the
model in **mAI SoL H3 Loader**. `model_index.json` must be immediately inside
the model folder, alongside all component directories. The loader checks the
package during queue validation, before upstream video generation starts.
Missing-package errors list the exact paths searched; incomplete packages
report the missing configuration or shard. This check reads local metadata
only and does not import SoL inference dependencies or load GPU weights.

The loader also searches existing `sol_refiner` and `diffusers` entries in
`extra_model_paths.yaml`. Entries point to the parent of the package folder.
The default model ID resolves to the original local folder name; alternate
folder names appear in the selector after refresh/restart. The pipeline class,
component files, shard presence, and unshifted scheduler are validated.

Required Diffusers APIs: `AutoencoderKLLTX2Video`, `LTX2ConditionPipeline`,
`LTX2TextConnectors`, `LTX2LatentUpsamplerModel`, `LTX2VideoTransformer3DModel`,
and `LTX2VideoDiffusionDecoderModel` with tiled/untiled decode support. Known
compatible reference revision:
`e0abab83b5df05de9e7abd788643c1a7c1e42e28` (0.41.0.dev0 APIs). Keep a compatible
installed build; there is no verified minimum released Diffusers version.
Transformers needs the package's `Gemma4UnifiedForConditionalGeneration`
support (reference >=5.12.1). Optional dependencies are documented in
`requirements-sol.txt`, separate from the existing pack requirements. Nothing
was installed or changed automatically. Review the optional requirements in
your ComfyUI environment before installing; do not replace/downgrade PyTorch.
Linux Flex requires PyTorch's compatible Triton compiler. NATTEN is optional,
and must match Linux PyTorch, CUDA, and GPU architecture; it is never installed
or compiled by the node. No Windows Triton dependency is added.

Prompt encoding uses ComfyUI's native text-attention selector, which handles
Gemma4Unified's 512-wide attention heads. This avoids the 256-head-dimension
limit of `--use-ck-attention` for the text encoder while the video transformer
continues to use ComfyUI's selected optimized attention. Decoder Flex/NATTEN
selection is independent of text attention.

Loader inputs:

| Input | Default / behavior |
| --- | --- |
| `model_name` | `Efficient-Large-Model/SoL-Refiner-LTX-2.5-for-MiniMax-H3`, local only |
| `precision` | `bf16`; `fp8` explicitly quantizes large transformer block linears through ComfyUI/comfy-kitchen |
| `decoder_backend` | `auto`: use NATTEN after a small compatibility probe, otherwise Flex; explicit `natten` fails clearly when unavailable |
| `keep_model_loaded` | `true`: retain reusable component weights between runs; false unloads and releases them after execution |
| `gpu_resident` | `true`: avoid deliberate stage offloads; ComfyUI may evict weights for other models/memory pressure. False stages components and intermediate tensors through CPU |

Loader output: `refiner` (`MAI_SOL_H3_REFINER`). Identical loader settings share
a weakly held runtime while workflows reference it. Cache keys include local
directory, precision, resolved backend, lifetime/residency policy, and device.
Components load lazily once and remain subject to ComfyUI ModelPatcher memory
management. Full text/VAE/decoder component loads are required; `--novram` is
unsupported. FP8 leaves boundary projections, modulation tables, and norms in
BF16; it requires supported hardware and ComfyUI's quantization API. NVFP4 is
not implemented. FP8 quality/stability still requires real GPU comparison.

Refiner inputs:

| Input | Default / behavior |
| --- | --- |
| `refiner`, `images` | Loader output and RGB float `IMAGE` batch `[B,H,W,3]` in 0..1 |
| `prompt` | Empty UI default; execution requires the original visible scene/action description; tokenized to at most 1024 tokens |
| `fps` | 24.0, range 1..240; sets temporal position encoding |
| `width`, `height` | 1920 x 1080, even dimensions 224..4096 |
| `sigma` | **0.9093750119**, the official one-step sigma; changes are experimental |
| `seed`, `decoder_seed` | 0/0; independent refinement and generative decoder noise |
| `decoder_mode` | `auto`: untiled first, retry once tiled only on `torch.cuda.OutOfMemoryError`; explicit `untiled` never enables decoder tiling |
| `decoder_tile_size` | 768 pixels; 256/384/512/768/1024 |
| `decoder_tile_frames` | 128; 16/32/64/128, temporal stride derived as 5/8 of size aligned to 2 frames (default 80) |
| `decoder_tile_stride` | 512 pixels; must be a multiple of 8 smaller than the spatial tile size; reduce it when selecting smaller tiles |

Output: `images` (`IMAGE`), CPU float32 RGB `[B,height,width,3]` in 0..1,
with exactly the input frame count. Internal spatial dimensions round up to
multiples of 64 (1920x1080 -> 1920x1088), with center cropping after decode.
Source conditioning uses half that canvas, deterministic VAE posterior mode,
latent upsampling, and LTX normalization. There is exactly **one** video
transformer prediction, no CFG and no negative prompt. The unshifted
FlowMatch Euler step from sigma to zero equals `noisy - sigma * velocity`.
The diffusion decoder is independently seeded and also uses its released
one-step x0 configuration. Tiled output can differ from untiled output.

Frame padding repeats the last frame to the next `8k+1` length and trims after
decode. Minimum is derived from the decoder configuration: the released final
11-frame attention kernel needs 17 frames on that grid. Thus 1, 8, and 9 frames
process internally as 17; 121 stays 121; 124 becomes 129; 158 becomes 161. A
9-frame clip is already temporally aligned but still below the decoder minimum.
Padding is logged. NVIDIA's reference truncates to a compatible length;
this integration preserves every supplied frame. Same settings/seeds reproduce
noise, subject to CUDA/backend numerical nondeterminism.

Testing: run `python -m pytest` for offline geometry, padding, tensor, seed,
backend, model validation, mocked pipeline, and recovery checks. No checkpoint
is downloaded by tests. For manual CUDA validation, restart ComfyUI, refresh
the browser, add the loader/refiner and connect decoded H3 frames -> refiner ->
Preview Image or your existing video output nodes. Start with a short clip,
its original prompt, 24 fps, BF16, Flex, untiled, 1920x1080, official sigma,
and seeds 0/0. Check frame count, resolution, range and repeatability. Compare
sigma 0.89 and 0.92 and tiled/untiled modes with everything else fixed; inspect
faces, eyes, hair, clothing, signage, architecture, textures, temporal flicker,
identity drift, and object consistency. No alternate sigma is claimed better.
Repeat with NATTEN if available, and compare FP8 to BF16 for finite pixels and
quality. Queue twice to verify component reuse, then test residency off,
keep-loaded off, interruption, and decoder OOM recovery on the target machine.

This integration has not been validated with the full checkpoint on Linux
CUDA and is experimental. Flex compilation, real NATTEN support, BF16/FP8
numerical behavior, VRAM use, and video quality require that manual test.
Attribution is in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

## mAI fal image edit

Generic image generation and editing node under `mAI / Image`, registered as
`MAIFalImageEdit`. Select the endpoint using `model_endpoint`:

| Endpoint / API documentation | Image input | Mask input |
| --- | --- | --- |
| [Ideogram 4.5 Edit](https://fal.ai/models/ideogram/v4.5/edit/api) (`ideogram/v4.5/edit`, default) | Required; exactly one image | Optional; white edits, black preserves |
| [FLUX 3 Text to Image](https://fal.ai/models/blackforestlabs/flux-3/text-to-image/api) (`blackforestlabs/flux-3/text-to-image`) | Leave disconnected | Leave disconnected |
| [FLUX 3 Edit Image](https://fal.ai/models/blackforestlabs/flux-3/edit-image/api) (`blackforestlabs/flux-3/edit-image`) | Required; a batch of 1-10 ordered references | Unsupported; disconnect or choose Ideogram |
| [Nano Banana 2.1 Edit](https://fal.ai/models/google/nano-banana-2.1/edit/api) (`google/nano-banana-2.1/edit`) | Required; exactly one RGB image | Optional; mask reference guidance followed by local compositing |

Inputs:

- `prompt`: non-empty generation prompt or editing instructions.
- `api_key`: fal key; blank uses the server's `FAL_KEY` environment variable.
  Clear an entered key before sharing a saved workflow.
- `image`, `mask`: optional sockets, validated for the selected endpoint before
  upload. Ideogram can edit the whole image when no mask is connected. Its mask
  is thresholded at 0.5 and inverted for fal, and must match the image dimensions
  and contain both edit and preserve regions.
- `quality`, `edit_precision`: Ideogram only; defaults `medium` and `high`.
- `seed`: Ideogram and Nano Banana; `-1` (default) lets the service choose.
  Omitted from FLUX 3 requests; the seed output remains `-1` when absent.
- `resolution`: FLUX 3 uses `512sq`, `768sq`, `1k` (default), `2k`, `4k`.
  Nano Banana accepts only `1k`, `2k`, `4k`, translated to API values `1K`, `2K`, `4K`.
- `aspect_ratio`: FLUX 3 and Nano Banana; `auto` (default) or a supported ratio.
  Nano Banana masked editing requires `auto`. Extreme ratios `4:1`, `1:4`,
  `8:1`, `1:8` are Nano Banana only; `2:1`, `7:5`, `5:7`, `1:2` are FLUX only.
- `output_format`: FLUX 3 and Nano Banana; `png` (default) or `jpeg`.
- `enable_prompt_expansion`: FLUX 3 only; default `false`.
- `safety_tolerance`: FLUX 3 only; 0-4, default 2; 0 is strictest.

Outputs, in order: `image` (`IMAGE`, first result), `image_url` (`STRING`),
`seed` (`INT`, `-1` when absent), `response_json` (`STRING`, full response).
FLUX 3 exposes no seed input or result seed in its current schema. Ideogram
always uses `image_size=auto`; the provider may downscale large inputs. FLUX 3
references must each be at least 256 pixels per side and at most 4 megapixels;
the node validates this without resizing. Describe local FLUX edits in the
prompt; its API has no mask field. Unsupported image/mask connections raise a
clear error instead of being ignored. Only the listed endpoints are supported.

The existing **mAI fal ideogram edit** node keeps its registration, required
mask, widget order, defaults, and outputs for saved workflows. Use **mAI fal
image edit** for the new model selection and optional image/mask sockets.

To test, install the existing requirements, restart ComfyUI, and refresh the
browser. Add **mAI fal image edit**, enter a key and prompt, select FLUX 3 Text
to Image, leave both sockets disconnected, and connect `image` to Preview
Image. For FLUX editing, select Edit Image and connect an input image; for
masked editing, select Ideogram and connect the mask as well. Each queue makes
one new billable fal request. Offline tests: `python -m pytest`; fake-client
tests check endpoint routing, reference order, validation, and decoded outputs.
Live image quality and provider behavior require a paid request.

### Nano Banana mask inpainting

Select `google/nano-banana-2.1/edit`, paste your key into `api_key`, connect a
single RGB image and matching-size `MASK`, set `aspect_ratio=auto`, and describe
the replacement in `prompt`. **White edits, black preserves**; gray mask values
blend the generated result with the original, preserving feathered edges.
Leave the mask disconnected for ordinary whole-image editing.

The endpoint has no native mask parameter. The node sends the source as image 1
and a grayscale mask as image 2, adds explicit region-editing instructions, and
makes one API request. It then resizes the returned image to source dimensions
when needed and composites it locally using your mask. Black-mask pixels stay
identical to the original tensor. Model compliance and alignment inside the
mask are not guaranteed; the model can still alter geometry, and resizing may
reduce detail. An empty mask, mismatched dimensions, or invalid mask values are
rejected before upload. A fully white mask edits the whole image.

With a mask, `image` contains the local composite at source dimensions;
`image_url` and `response_json` refer to the raw fal result before compositing.
Quality/precision, prompt expansion, and the FLUX safety widget are not sent to
Nano Banana; it uses its own API defaults for those service settings.
The [model gallery](https://fal.ai/models/google/nano-banana-2.1/edit) currently
labels this endpoint integration-only; live availability depends on fal.
No paid request was made during implementation. Run `python -m pytest` for mask
orientation, soft-edge compositing, source preservation, and fake-client tests.
For a live check, paint a small region and queue once, then compare the original
and node output outside the mask. Existing registrations and outputs are unchanged.

## mAI fal ideogram edit

Edits a source image using a mask and prompt through
[Ideogram 4.5 Edit on fal](https://fal.ai/models/ideogram/v4.5/edit/api).
Find it under `mAI / Image` as `mAI fal ideogram edit`, registered as
`MAIFalIdeogramEdit`.

Install the requirements as described below, restart ComfyUI, and refresh the
browser. Connect Load Image's `IMAGE` to `image`, paint a region using ComfyUI's
mask editor, and connect its `MASK` to `mask`. Enter your editing prompt and
connect the output `image` to Preview Image or Save Image.

Inputs:

- `image`: exactly one source image, uploaded as PNG without local resizing.
- `mask`: one matching-size ComfyUI mask. **White edits, black preserves**.
  Values >=0.5 become the edit region; lower values become the preserved region.
  The node automatically inverts this binary mask for fal. Both regions must
  exist; an entirely black or white mask is rejected before upload.
- `prompt`: required text describing the edit.
- `api_key`: fal key, or leave blank to use `FAL_KEY` set in the ComfyUI server
  environment before starting ComfyUI. An entered key takes precedence.
  Widget values are saved in workflows, so clear an entered key before sharing.
- `edit_precision`: `high` (default) restores unchanged pixels; `regular` uses
  standard editing.
- `quality`: `very_low`, `low`, `medium` (default), or `high`.
- `seed`: `-1` (default) lets fal choose; zero or higher requests that seed.

Outputs, in order: `image` (`IMAGE`), `image_url` (`STRING`), `seed` (`INT`),
`response_json` (`STRING`). The last output contains the complete fal response;
seed is `-1` if the response omits it.

The node requests one image with `image_size=auto`, as required for masked edits.
The provider may downscale large source images; exact output dimensions are not
guaranteed. Soft masks are thresholded, not feathered. Batches and reference
images are not supported by this node. A missing or mismatched mask raises a
clear error. Each queue uploads the source and mask and submits a new billable
request; fal credentials, credits, and network access are required.

Run `python -m pytest` for offline utility and fake-client tests, including mask
direction, validation before upload, and output decoding. For a live test, paint
a small region, describe a visible change, and queue once. Confirm the painted
region changes and the preserved area stays intact with high precision. The
live test spends fal credits.

## mAI fal image

Calls a fal.ai image generation, editing, or upscaling endpoint and returns its first image as a
ComfyUI `IMAGE`. Find it under `mAI / Image` as `mAI fal image`.

### Setup

Install the node pack requirements into the same Python environment used by
ComfyUI:

```powershell
python -m pip install -r requirements.txt
```

Paste your fal API key into the node's `api_key` field.

Warning: ComfyUI stores widget values in workflow JSON. Do not share or publish
a workflow containing your key. Clear the field before exporting a workflow.

### Inputs

- `api_key`: fal API key used for this request.
- `model_endpoint`: dropdown of endpoint IDs supported by this node. The default
  is `fal-ai/flux/dev/image-to-image`.
- `prompt`: text prompt sent to the endpoint.
- `image_size`: fal size preset used when `resolution_mode` is `preset`.
- `seed`: `-1` asks fal for a random seed; zero or higher sends that exact seed.
- `output_format`: `png` or `jpeg`.
- `resolution_mode`: `preset` or `custom`. Custom mode sends an exact
  `{width, height}` request only to compatible endpoints.
- `custom_width`, `custom_height`: requested output dimensions in custom mode.
- `image_1`, `image_2`, `image_3` (optional): ComfyUI images uploaded as PNG to
  fal before generation. Editing accepts batches; upscalers require exactly one
  image total, connected to any one of these sockets.
- `upscale_factor` (optional, default `4.0`): multiplies both source dimensions
  for upscalers only. A 1024x1024 input requests 4096x4096 at 4x.
- `topaz_model` (optional, default `High Fidelity V2`): Topaz enhancement model;
  ignored by other endpoints. Missing new widgets use these defaults, including
  when executing older saved workflows.

Supported endpoint choices:

- `fal-ai/flux/dev/image-to-image`
- `fal-ai/flux-2/edit`
- `fal-ai/flux-2-max/edit`
- `fal-ai/flux-2/lora/edit`
- `fal-ai/flux-2/klein/9b/edit`
- `fal-ai/flux-2/klein/9b/edit/lora`
- `openai/gpt-image-2/edit`
- `fal-ai/nano-banana-2/edit`
- `bytedance/seedream/v5/lite/edit`
- `bytedance/seedream/v5/pro/edit`
- `fal-ai/seedvr/upscale/image`
- `fal-ai/topaz/upscale/image`
- `fal-ai/clarity-upscaler`
- `clarityai/crystal-upscaler`
- `fal-ai/aura-sr`

Custom width and height are supported for editing dropdown endpoints except:

- `fal-ai/flux/dev/image-to-image`, whose schema has no output size input.
- `fal-ai/nano-banana-2/edit`, which uses aspect-ratio and resolution tiers
  instead of exact dimensions.

Selecting custom mode with either unsupported endpoint raises a clear error
before any billable request is submitted. Known custom-size limits are checked
locally: FLUX.2 Edit and FLUX.2 LoRA Edit require each dimension from 512 to
2048; Seedream 5.0 Lite requires total pixels from 2560x1440 to 4096x4096 to
avoid automatic scaling; Seedream 5.0 Pro requires total pixels from 1024x1024
to 2048x2048 and an aspect ratio from 1:16 to 16:1. Other compatible endpoints
may apply additional fal-side validation.

### Outputs

- `image`: the first returned image as a ComfyUI image tensor.
- `image_url`: the fal-hosted URL (or data URI) returned by the API.
- `seed`: the actual result seed when fal returns one, otherwise `-1`.
- `response_json`: complete fal response for metadata and debugging.

The node makes a new billable API request every time its workflow is queued.
The node automatically sends `image_urls` for edit endpoints that require a
list, including `fal-ai/flux-2-max/edit`,
`fal-ai/flux-2/klein/9b/edit/lora`, and
`fal-ai/nano-banana-2/edit`, and `bytedance/seedream/v5/pro/edit`, even when
only one image is connected. For Nano Banana 2 Edit, the node translates
`image_size` to the endpoint's `aspect_ratio` and sends the default `1K`
resolution. Seedream 5.0 Pro Edit accepts the node's existing `image_size`
presets but does not support a seed, so that widget is ignored for this
endpoint and the seed output is `-1`. Seedream 5.0 Lite Edit also accepts the
existing size presets. Its API does not accept requested seed or output format
values, so those widgets are not sent; the generated seed returned by fal is
still available from the node's seed output. At least one image must be
connected for these edit endpoints. The node sends `image_url` for the default
FLUX.1 image-to-image endpoint and `image_urls` whenever multiple images are
connected. Model schemas differ, so other custom endpoints selected from the
[fal model gallery](https://fal.ai/models) must support those standard field
names. The node returns the first output image.

### Upscaling generated images

Connect your generator's `IMAGE` output to a second **mAI fal image** node's
`image_1`, select an upscaler, and leave `upscale_factor` at `4.0`. Connect the
result to Save Image or Preview Image. Upscalers ignore `image_size`,
`resolution_mode`, `custom_width`, and `custom_height`; the input is uploaded at
its original dimensions without local resizing. All five options support 4x.

Choose according to the image; there is no single best model for every source:

| Endpoint / API documentation | Use and node settings | Scale supported by this node |
| --- | --- | --- |
| [Topaz](https://fal.ai/models/fal-ai/topaz/upscale/image/api) | Start with `High Fidelity V2` for detail preservation; try `CGI` for rendered art. `Standard MAX`, `Wonder 3`, and `Redefine` provide generative alternatives. Face enhancement and cropping are disabled. | 1-4x |
| [SeedVR2](https://fal.ai/models/fal-ai/seedvr/upscale/image/api) | General restoration option; uses factor mode and noise scale 0.1. | 1-10x |
| [Clarity](https://fal.ai/models/fal-ai/clarity-upscaler/api) | Prompt-guided enhancement, with conservative creativity 0.2 and resemblance 0.8. | 1-4x |
| [Crystal](https://fal.ai/models/clarityai/crystal-upscaler/api) | Portrait and facial detail enhancement, creativity 0. | 1-10x |
| [AuraSR](https://fal.ai/models/fal-ai/aura-sr/api) | Uses v2 with overlapping tiles to reduce seams. | Exactly 4x |

`prompt` is used only by Clarity and Topaz `Redefine` when upscaling. Replace the
node's example lake prompt with a description of your source, or clear it to
use the service default. Redefine uses creativity 1 and accepts at most 1024
prompt characters. `seed` is sent only to SeedVR2 and Clarity; the seed output
is `-1` when absent from the response. `output_format` applies to Topaz,
SeedVR2, and Crystal (`jpeg` is translated to `jpg` where needed); Clarity and
AuraSR use the service's output format.

Invalid scale factors and multiple input images/batches are rejected before
upload or submission. No automatic extra paid passes are made. Larger images
increase cost and memory use (4x width and height means 16x the pixels), and fal
may impose additional image-size limits. Generative models can alter fine
details. API compatibility is checked against public schemas; image quality
and service-side output dimensions require a live comparison on your images.
The registered class remains `MAIFalImage` and all existing outputs retain
their names, types, and order. No additional dependencies are required.

### Testing

Pure utility tests do not contact fal:

```powershell
python -m pytest
```

For an end-to-end ComfyUI test, connect `image` to Preview Image, enter your API
key in the node, and queue the workflow.
For upscaling, restart ComfyUI and refresh the browser, connect one generated
image, select Topaz or SeedVR2 at 4x with PNG output, and queue once. Verify that
the saved image has four times the source width and height and inspect fine
details at 100%. Each queued request is billable. The automated tests use a
fake fal client and do not spend credits.
