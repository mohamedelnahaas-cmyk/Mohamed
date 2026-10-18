---
name: typesafe-images
description: >
  Use TypeSafe Jev 1.14's experimental image inputs and Box primitive: send images as
  state, ask Choice/Noul/Score questions about them, and locate one region per Box question.
  Use when a feature needs to judge, classify, verify or locate something in a photo,
  scan, screenshot, chart or log image. Complements the typesafe-ai skill (load it too).
---

# TypeSafe images and Box (Jev 1.14 preview)

Load the `typesafe-ai` skill as well. Its rules still apply: read live docs first, code owns
the workflow, ask narrow questions, and validate thresholds on your own data.

Sources and what was verified (2026-10-18):

- **Contract** below is taken from `typesafe-sdk` 0.7.5, generated from the API's OpenAPI spec.
  This is the authoritative interface.
- **Release facts** come from secondary coverage of the 2026-10-14 launch: TypeSafe's X post,
  OpenRouter and Cloudflare model docs, and Marktechpost. The official pages
  (docs.typesafe.ai/images, /primitives/box, typesafe.ai/blog/jev-1-14) could not be read when
  this skill was written. Re-check them before relying on limits or aliases.

## Status: experimental, not production

- Image input and Box are **preview / beta**. TypeSafe describes them as "for dreamers" and
  does not call them production-grade.
- They are opt-in per request with the header `TypeSafe-Danger-Zone-Images: true`.
- Images need a vision model. Coverage names Jev 1.14 under the `jev-preview` alias.
  Confirm with `client.models.list()`. Box is rejected (HTTP 400) by Jev 1.13.
- Gate any image feature behind a flag, with a non-image fallback.

## Request contract (SDK 0.7.5)

```python
from pathlib import Path
from typesafe_sdk import Box, Choice, Image, Noul, TypeSafeClient

client = TypeSafeClient(
    headers={"TypeSafe-Danger-Zone-Images": "true"},
    model="jev-preview",              # or TYPESAFE_DEFAULT_MODEL; verify with client.models.list()
)
photo = Image(Path("bit_dull.jpg"))   # also Image(bytes), Image.from_base64(s), Image.from_pil(img)

result = client.system_one(
    state={"photo": photo, "bit": {"type": "PDC", "size_in": 8.5, "blades": 6}},
    questions={
        "has_broken_cutters": Noul(instructions="Does `photo` show broken or chipped PDC cutters?"),
        "worst_region": Box(instructions="The area of `photo` with the most severe cutter damage."),
    },
)
p = result.nouls["has_broken_cutters"].noul
box = result.boxes["worst_region"]       # BoxAnswer: .box.top/.bottom/.left/.right, .confidence
```

- **Images are assets.** An `Image` placed anywhere in `state`, `instructions` or `criteria`
  (including inside an f-string, via `str(image)`) is sent once and referenced by
  placeholder. Several images can go in one request and be compared.
- **Formats:** PNG and JPEG (`png`, `jpg`, `jpeg`). The API detects the format from the bytes.
  The SDK uploads as `multipart/form-data` automatically.
- **Choice, Noul and Score** work on images exactly as on text.
- **Box question:** `Box(instructions="what to locate")`. There are no criteria. It returns one box.
- **Box answer:** `box.top`, `box.bottom`, `box.left`, `box.right` are fractions in 0–1 of image
  height and width, measured from the top-left corner. Each is the expected value of that edge's
  position. `confidence` is 0–1, or `None` when the model gives none.
  Pixels: `x0 = left * W`, `y0 = top * H`, `x1 = right * W`, `y1 = bottom * H`.

## Design rules for images

- **Box has no "absent" answer.** It always returns four coordinates, even when the object
  isn't there. Pair every Box with a Noul presence question in the same request, and ignore
  the box when presence is low.
- **One Box per object.** For N objects, ask N Box questions, or crop and re-ask.
- **Check crops.** Coverage reports TypeSafe's own test: 4 edge-check Nouls ("is the object
  cut off at the top/bottom/left/right edge of the crop?") caught 30 of 36 cut-off crops. Use
  edge checks before trusting a crop for a downstream judgment.
- **Coarse-to-fine:** Box the region, crop it in code (Pillow), then ask detailed
  Choice/Score questions on the crop.
- **Keep facts in state.** Put known metadata (IDs, depths, sizes, dates) in state next to the
  image, so the question compares the image to context instead of guessing it.
- **Expected-value edges blur under ambiguity.** If the target could be in two places, the box
  can land between them. Low confidence or a very large box means: re-ask, or hand it to a person.

## Other 1.14 changes (from coverage)

- Improved base model and a long-context mode (~262k tokens) on the preview alias.
- Pricing reported unchanged: $0.042 per million input tokens, output free. Images consume input tokens.

## Test before shipping

Build a small labeled set (20–50 images per decision) and measure each Noul/Choice threshold
and Box IoU against your labels. Log the state, questions and answers of failures. Typed output
guarantees the shape of the answer, not its truth.
