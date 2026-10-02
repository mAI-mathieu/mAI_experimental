# mAI experimental ComfyUI nodes

Small experimental nodes for ComfyUI.

## mAI fal image edit

Generic image generation and editing node under `mAI / Image`, registered as
`MAIFalImageEdit`. Select the endpoint using `model_endpoint`:

| Endpoint / API documentation | Image input | Mask input |
| --- | --- | --- |
| [Ideogram 4.5 Edit](https://fal.ai/models/ideogram/v4.5/edit/api) (`ideogram/v4.5/edit`, default) | Required; exactly one image | Optional; white edits, black preserves |
| [FLUX 3 Text to Image](https://fal.ai/models/blackforestlabs/flux-3/text-to-image/api) (`blackforestlabs/flux-3/text-to-image`) | Leave disconnected | Leave disconnected |
| [FLUX 3 Edit Image](https://fal.ai/models/blackforestlabs/flux-3/edit-image/api) (`blackforestlabs/flux-3/edit-image`) | Required; a batch of 1-10 ordered references | Unsupported; disconnect or choose Ideogram |

Inputs:

- `prompt`: non-empty generation prompt or editing instructions.
- `api_key`: fal key; blank uses the server's `FAL_KEY` environment variable.
  Clear an entered key before sharing a saved workflow.
- `image`, `mask`: optional sockets, validated for the selected endpoint before
  upload. Ideogram can edit the whole image when no mask is connected. Its mask
  is thresholded at 0.5 and inverted for fal, and must match the image dimensions
  and contain both edit and preserve regions.
- `quality`, `edit_precision`, `seed`: Ideogram only. Defaults remain `medium`,
  `high`, and `-1` (random). These fields are not sent to FLUX 3.
- `resolution`: FLUX 3 only; `512sq`, `768sq`, `1k` (default), `2k`, `4k`.
- `aspect_ratio`: FLUX 3 only; `auto` (default) or a supported ratio. Auto follows
  the first reference when editing.
- `output_format`: FLUX 3 only; `png` (default) or `jpeg`.
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
