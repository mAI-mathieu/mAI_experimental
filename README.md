# mAI experimental ComfyUI nodes

Small experimental nodes for ComfyUI.

## mAI fal image

Calls a fal.ai text-to-image endpoint and returns its first generated image as a
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
- `image_size`: one of fal's standard image size presets.
- `seed`: `-1` asks fal for a random seed; zero or higher sends that exact seed.
- `output_format`: `png` or `jpeg`.
- `image_1`, `image_2`, `image_3` (optional): ComfyUI images uploaded as PNG to
  fal before generation. Each socket also accepts an image batch.

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

### Testing

Pure utility tests do not contact fal:

```powershell
python -m pytest
```

For an end-to-end ComfyUI test, connect `image` to Preview Image, enter your API
key in the node, and queue the workflow.
