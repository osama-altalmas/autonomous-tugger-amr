# AI stage 1: floor-marker reader (YOLO-pose)

Record of how the floor-marker reader of the autonomous tugger was built:
what was tried, what went wrong, what was changed and why. Every dataset,
trial and result mentioned here is kept in this folder on the development
laptop; the repository holds this record, its charts and the result
summaries (the datasets are several GB). The model was trained with
Ultralytics YOLO, which is AGPL-3.0, so this branch (AMR-V3) is AGPL-3.0.

## Goal

At every station the tugger stops over a floor marker. A downward camera
must tell **which station** it is and **where the red docking frame is**,
precisely enough to dock within 10 mm and 1 degree, also when the marker
is worn, dirty or partly covered. The reader must say how sure it is:

- **OK**: recognised with high confidence;
- **ALERT** (confidence below 80 %): still readable, clean or replace it soon;
- **FAULT** (below 50 %, unreadable, wrong station, or a frame that cannot
  be measured precisely): the robot does not dock on it.

The classic reader (OpenCV QR decoding) failed in fault testing: a worn
strip covering 5 % of a QR code made it unreadable (2026-11-14).

## Iterations

### Iteration 0: first trial (`iteration0_trial/`)

- Markers: standard QR codes (`TUGGER:S1`, ...) in a red frame.
- Generator v1 (`tools/ai/marker_dataset/generate.py`): Gazebo only, the
  robot placed over each marker at random offsets (8 cm) and headings; a
  clean image labels the pose (OpenCV finds the frame and the QR), then
  damage patches (tape, dirt, worn strips, tyre marks, paint) are laid on
  and the light changes; the damaged image keeps the clean labels.
- Finding: 23 of 50 samples could not be labelled because the 8 cm offset
  pushed part of the frame out of view. Offset reduced to 6 cm.

### Iteration 1: background shortcut (`iteration1_background_shortcut/`)

- 433 images generated before the run was stopped on purpose.
- Finding: the two drop-off stations stood on **green plates**, the others
  on grey floor. A model can learn "green floor = drop-off" instead of
  reading the marker, which breaks as soon as a floor changes.
- Changes: the green plates were removed from the world (they had no
  function, visual only); random floor mats (6 colours) are laid under the
  marker in 40 % of the images so no floor colour is tied to a station.

### Marker redesign: four AprilTags instead of a QR

- Measured on the five QR codes: three of a QR's four corners hold the same
  finder squares on every code, and all five encode the same `TUGGER:`
  prefix. Between the two most similar markers (S1, D2) the top-left quarter
  differed in only 8 of 100 modules (8 %); most of the identity was in the
  bottom-right quarter. Losing that quarter left almost nothing to tell the
  stations apart.
- New marker (user idea, 2026-11-15): **four AprilTags (36h11), one per
  quarter**, all naming the same station (HOME 0-3, S1 4-7, D2 8-11, S3 12-15,
  D4 16-19). Any single surviving quarter identifies the station and shows
  which corner it is. 36h11 codes differ in at least 11 of 36 bits and carry
  their own error correction. The red frame is unchanged.
- Verified: a full simulated cycle docked at all five stations with the new
  markers (2.2 to 5.6 mm), and any one quarter alone identifies a station
  (unit test).

### Iteration 2: pose labels (`iteration2_pose_trial/`, then `markers_v2/`)

- A detection box is not precise enough to dock, and it shrinks when a frame
  side is wiped. The model is therefore **YOLO-pose**: besides the station it
  predicts **20 keypoints** (4 frame corners, 4 corners of each tag), also
  where they are covered, learnt from labels that keep covered points.
- Labels are exact: in the clean image the 16 tag corners map the marker
  plate onto the image (homography); all 20 keypoints are projected from the
  plate's known geometry. A keypoint under a damage patch is marked covered.
- New damage: a whole frame side wiped out (20 % of damaged samples).
- Classes: the station while at least one tag is readable (at most 15 %
  covered), `unknown_marker` when no tag is but the frame still shows (the
  robot can align but cannot confirm the station: FAULT), no object when
  neither shows. In-between samples are left out so the model is not taught
  either way.

### The datasets (2026-11-15)

| Set | Images | Make-up |
|---|---|---|
| `markers_v2` (training) | 3508: 2987 train, 521 validation | about 500 markers per station and 513 `unknown_marker`; damage: 569 clean, 1152 light, 803 medium, 584 heavy; 463 with a frame side wiped; 400 empty floor |
| `markers_v2_test` (held out) | 673 | 89-100 per class; 82 side-wiped; 100 empty floor; unseen damage, mats and light |

Generating took about 0.5 s per sample; 149 poses could not be labelled
(part of the frame out of view) and 743 in-between samples were left out on
purpose. Previews and charts: `markers_v2/previews/`, `markers_v2_test/previews/`.

### Known weakness of dataset v2 (user, 2026-11-15)

v2's damage came from a fixed pool of 60 patches (shapes and colours) laid
at random positions: across 3508 images the same patches recur thousands of
times, and there is no dust at all. The risk is a model that learns "these
strips mean damage" instead of "anything over a tag". Model v2 is trained
anyway, as the measured baseline: a clear gap between its validation score
and its held-out test score (unseen damage) would confirm the overfitting.

### Iteration 3: clean images + augmentation (`markers_v3_clean/`, `augment.py`)

Design (user, 2026-11-15): give one clear image per marker and pose,
labelled automatically from its tags, and create every damage scenario from
it by augmentation; nothing is captured again, and the same works for real
photographs. Every damage is generated afresh (size, colour, angle, count,
opacity), so no patch repeats; dust film, specks, smears, faded print,
scratches, glare and blur are procedural (`surface_wear.py`). Dust first
showed a square outline (the edge of the drawing area), a shortcut a model
could learn; it now fades out around a random centre with an irregular edge.

Scaling (user, 2026-11-15: more than 5 stations): the model has one class,
`marker`; the station is read from the tags with the keypoints
(`marker_identity.py`: 587 codes, up to 5 wrong bits corrected, 4 tags
vote), so a new station is a line in the station table, not a retraining.

### Training environment issue (Colab, 2026-11-15)

The first v2 training run stopped after epoch 1 at the validation step:
`AttributeError: module 'numpy' has no attribute 'trapz'`. Colab now runs
Python 3.13 with numpy 2, which renamed `trapz` to `trapezoid`; Ultralytics
8.3.40 still calls the old name. Fix: `np.trapz = np.trapezoid` before
training (added to every notebook). The data was not involved.

### Dataset v3 (2026-11-15)

| Set | Clean images | After augmentation | Make-up |
|---|---|---|---|
| `markers_v3` (**trained**, LED ring) | 2118 (`markers_v3_clean/`) | 3 variants per marker image, made on Colab (seed 5) | as below; keypoints in image order |
| `markers_v3_ambient` (first v3, no LED, kept) | 2119 (`markers_v3_clean_ambient/`) | 7876: 6658 train, 1218 validation (4 variants per marker image) | 1169 clean, 2336 light, 2272 medium, 1899 heavy, 200 empty floor; 5320 with random tag codes; 1282 with a wiped frame side |
| `markers_v3_test` (held out, LED ring) | 538 (`markers_v3_clean_test/`) | 1514 (3 variants, seed 97) | other poses, unseen damage colours and shapes, other mats and light |

Random tag codes were checked: 120 of 120 swapped tags read back correctly
with both the keypoint reader and the classic detector. Augmenting the
training set takes about 21 minutes on a laptop CPU. Previews:
`markers_v3_ambient/previews/` (`one_clean_many_variants.png` shows one clean
image turned into four damaged ones) and `markers_v3_clean/previews/`
(labels in image order).

### Lighting under the robot (user, 2026-11-15)

Under the robot, and more so under cargo, almost no room light reaches the
floor: a real camera there sees mostly darkness. The simulation hid this:
the floor under the robot was lit by a fixed background light only, so the
camera image stayed the same (grey 64.5) with the room light normal, dimmed
or off. The "light 0.35-1.1" varied in datasets v2 and v3 therefore never
reached the camera; their light variety came from augmentation only.

Decision (user): the camera brings its own light, **an LED ring around the
lens inside a light shield** (a brush skirt from the chassis to 10 mm above
the floor), as warehouse robots that read floor markers do. The image then
looks the same at every station, under cargo or not. Added to the robot
model: an open LED ring around the camera (the camera itself now lies
flat) with 12 wide spot LEDs at 70 mm from the lens. Ring size from an
estimate of a real ring (LEDs spreading light as cos, 6 cm above the
floor): with LEDs at 4.5 cm the frame corners of the 12 cm marker got only
36 % of the centre's light; at 7 cm tags, frame sides and corners get 1.19,
1.32 and 1.05 (user, 2026-11-15). The simulator's lights are simpler and
look more even than a real ring. The skirt is
not in the model (user, 2026-11-15): in simulation it does not change the
image, but a real robot needs one (about 0.40 x 0.34 m, inside the wheels). The floor now reads 192 in the middle and 155 at the edges, the
falloff of a real ring light (`lighting/`: `ring12_marker.png` with the ring,
`before_sun_normal.png` and `dim2_*.png` before it, `led_aug.png` for the
augmented light faults).
Augmentation adds what can go wrong with it: weak LEDs, one LED out (one
side darker), LED reflections on a glossy marker, more noise when darker.
The reader checks the light on every image: FAULT "camera image too dark"
(LEDs off: grey 68 instead of about 180), ALERT when weak, so a light
failure is not mistaken for a worn marker. The v3 clean images are
captured again with the LED ring before v3 is trained.

## Against overfitting

- One image per robot pose: no near-duplicates across train and validation.
- Floor mats, every heading, damage from none to heavy; in v3 every damaged
  image is new (augmentation), not drawn from a fixed pool.
- A **held-out test set** per dataset (`markers_v2_test/`, `markers_v3_test/`,
  seed 97): damage colours and shapes, floor mats and light never used in
  training. Scored once per model.
- Small model (YOLO11n-pose), colour/brightness/scale augmentation, no flips
  (a mirrored tag is not valid), no rotation in v3 (it would break the image
  order of the keypoints), weight decay, early stopping (patience 25 in v2,
  20 in v3).
- The notebook plots training against validation losses; a widening gap
  means overfitting.
- Later: real camera photos added, to close the gap between simulated and
  real images.

## Precision: model + geometry

YOLO-pose places corners to a few pixels. The frame is then fitted exactly:
the known square is fitted to the visible outer edge of the red frame
(`src/tugger_control/tugger_control/marker_geometry.py`). The red frame
measures 248.3 px across in the camera (5.63 cm from the camera; about 0.48 mm
per pixel; corrected 2026-11-16, see "Model v3"), so two adjacent sides fix
the whole frame. On synthetic frames: full
frame within 0.8 px, one side wiped within 1 px, two adjacent sides within
1.5 px of the centre; one side or two opposite sides are refused.

## Files

| What | Where |
|---|---|
| Generators v1 / v2 | `tools/ai/marker_dataset/generate.py`, `generate_pose.py` |
| Records and charts | `tools/ai/marker_dataset/report.py`, `report_pose.py` -> `*/previews/` |
| Colab training | `tools/ai/marker_dataset/train_markers_pose_colab.ipynb` (v2), `train_markers_v3_colab.ipynb` (v3) |
| Damage + LED augmentation | `tools/ai/marker_dataset/augment.py`, `surface_wear.py` |
| Reader comparison | `tools/ai/marker_dataset/evaluate_readers.py` -> `models/v3/eval/` |
| Reader (model, state OK / ALERT / FAULT, light check) | `src/tugger_control/tugger_control/marker_reader.py` |
| Station from the tags, image order | `src/tugger_control/tugger_control/marker_identity.py` |
| Frame fit | `src/tugger_control/tugger_control/marker_geometry.py` |
| Docking camera node (runs the reader, publishes `/tugger/marker_reader`) | `src/tugger_control/tugger_control/qr_docking_vision.py` |
| Trained model used by the robot | `src/tugger_control/models/marker_reader_v3.onnx` (from `models/v3/best.pt`) |
| Marker design | `src/tugger_control/tugger_control/docking_markers.py`, textures in `src/tugger_gazebo/media/markers/` |

## Results

### Model v2 (YOLO11n-pose, 150 epochs, Colab T4, 2026-11-15)

| | Validation | Held-out test (unseen damage, mats, light) |
|---|---|---|
| Marker found (box mAP50-95) | 0.994 | 0.952 |
| Keypoints (pose mAP50-95) | 0.197 | 0.179 |

Held-out confusion matrix (`models/v2/heldout_test/`): 460 of 487 markers
named correctly; 5 named as the **wrong station** (about 1 %; the expected
station check turns them into a FAULT, so the robot refuses to dock);
**20 of 100 empty floors were taken for a marker**, all on floor mats never
seen in training. This confirms the overfitting expected from v2's repeated
damage pool and fixed mats: the model partly learnt the training floors.
Keypoints are rough (a few pixels), as expected: the geometric frame fit
does the precise part. v3 is the answer: it does not learn stations at all
(tags are read) and every damage is new.

Measured on the laptop CPU (OpenCV DNN, ONNX): **184 ms per image**. The
keypoints showed why the pose score was low: the model often put a corner
on **another corner's place** (its "top-right" on the true bottom-left). A
square frame with four tags looks the same turned by 90 degrees, so "the
marker's own top-left corner" cannot be seen in the image; labels in marker
order asked the model for the impossible, and it averaged. Points matched
to the nearest true corner had a median error of 4.8 px. The frame fit,
given these mixed corners, refused all 573 test markers.

### Fix: keypoints in image order (v3, 2026-11-16)

Labels now number the corners by their place in the image: the frame
corner nearest the image's up-left first, then clockwise; the tags and each
tag's corners follow the same turn. The model only learns what it can see.
The tag reader (`marker_identity.py`) tries the four turns and keeps the
one where the tags' IDs sit in their own quarters (tag ID % 4); that turn
gives the marker's own corner order back. Unit-tested at every 15 degrees.
The reader node now follows the v3 design: one class, station from the
tags, frame fit, OK/ALERT/FAULT (ALERT also when only one tag still reads).
No rotation augmentation in training: it would break the image order.

### LED ring check (2026-11-16)

With the 12-LED ring the white plate is no longer washed out (3 % of its
pixels saturated with the first ring, 0 % now), and a full five-station
docking cycle with the classic tag reader docked at 2.7 to 5.7 mm (before
the ring: 2.2 to 5.6 mm). The v3 clean images are captured with the ring;
the earlier no-LED sets are kept as `*_ambient`.

### Model v3 (YOLO11n-pose, Colab T4, 2026-11-16)

Trained on the LED-lit `markers_v3` (2,118 clean captures x 3 damaged
variants, keypoints in image order, no rotation augmentation). Early stopping
ended the run at epoch 72; the best epoch was 52. Training and validation
losses fell together to the end (`models/v3/training/markers_v3/overfitting_check.png`):
no overfitting. The Colab session was reset after training (free GPU quota
used up); nothing was lost, as the run was saved to Google Drive, and the
ONNX export and all testing were done on the laptop CPU instead.

| | v2 | v3 |
|---|---|---|
| Validation, keypoints (pose mAP50-95) | 0.197 | **0.994** |
| Validation, marker found (box mAP50-95) | 0.994 | 0.994 |
| Held-out test, keypoints (pose mAP50-95) | 0.179 | **0.982** |
| Held-out test, marker found (box mAP50-95) | 0.952 | **0.982** |

(Each model on its own held-out test set; v3's scored on the laptop CPU,
`models/v3/heldout_test/`.) Validation and held-out test differ by 0.01: the
model holds up on damage, mats and light it never saw.

The image-order labels removed the corner mix-up: the keypoint score went
from 0.20 to 0.99.

**One held-out test for all three readers** (`markers_v3_test`, rebuilt on
the laptop with seed 97, identical to the Colab one; 1,514 images; damage,
mats and light never used in training). `evaluate_readers.py` scores v3, v2
and the classic OpenCV tag reader on the same images:

| 401 real markers | Classic tags | v2 | v3 |
|---|---|---|---|
| Station named right | 236 (59 %) | 190 (47 %) | **327 (82 %)** |
| Station named wrong | 0 | 11 | **0** |
| clean / light / medium / heavy damage, right | 66/66, 105/120, 56/121, 9/94 | 66/66, 73/120, 39/121, 12/94 | **66/66, 114/120, 105/121, 42/94** |

| Other images | Classic tags | v2 | v3 |
|---|---|---|---|
| 50 empty floors: marker reported | 0 | 0 | 0 (highest score 0.18) |
| 1,002 markers with unknown tag codes (a "new station"): known station named | 53 * | 414 | 23 * |
| 61 markers damaged beyond reading: station named | 0 | 5 wrong | 6, all right |

\* only where the random codes happened to include a real station's tag:
both readers read what was printed. v2, which learnt stations as classes,
guessed a station for 41 % of foreign markers; v3 never named a station
whose tags it did not read.

Keypoints (v3, against the labels): frame corners median 1.5 px (90 %:
6.0 px), tag corners median 1.1 px (90 %: 4.2 px).

**Frame fit:** 785 frames fitted, all within 1.5 px of the true red frame,
median 0.3 px (0.15 mm on the floor). Of the 401 real markers 211 were
fitted: 54/66 clean, 92/120 light, 57/121 medium, 8/94 heavy damage. The
reader then says OK or ALERT for 203 of them and FAULT for the rest; the
fit is now the limit, not the model (it refuses rather than guesses: a
refused frame is a FAULT, never a wrong position).

**CPU time per image** (laptop, OpenCV DNN): v3 134 ms (reader with tags and
fit), v2 103 ms, classic 3 ms.

Three faults in the frame fit were found by this test and fixed at their
cause:

1. *Wrong frame size.* Every fit was refused at first, even on clean images
   and with the true corners. In the simulation the tag plate lies 1 mm above
   the red frame; at 5.5 cm that makes the tags look 2.0 % larger. The camera
   height had been measured from the tags (0.055 m, frame 253.5 px); the red
   frame is 248.3 px (a square fit leaves 0.3 px rms), i.e. 0.0563 m from the
   camera. `CAMERA_HEIGHT_M` is now the distance to the red frame. The labels'
   frame corners carry the same 2 % (they are extrapolated from the tags);
   this does not matter, since the fit corrects them, and the test compares
   against the corners moved onto the frame's plane (0.3 px after the move).
   A real printed marker is flat; on a real robot the frame is measured.
2. *Dim light.* The red threshold was fixed; with ageing LEDs or one LED out
   the frame vanished from the mask. The image is now brought to the
   full-LED level first (as an auto-exposure would, gain up to 4).
3. *Frame cut by the image edge.* The image border was taken for a frame
   side. Edge pixels at the border are now ignored.
Unit tests for 2 and 3 (they fail without the fixes).

An error in the first scoring was also fixed: corners outside the image are
stored as (0, 0) in YOLO labels, which made correct fits look 170-310 px off;
fits are now scored against the captured (unclipped) corners.

### Docking with the v3 reader (2026-11-16)

`qr_docking_vision` now reads the marker with v3 by default (parameter
`marker_reader`: `v3` or `classic`; model installed as
`src/tugger_control/models/marker_reader_v3.onnx`): the model finds the
marker, its tags name the station, the frame fit gives position and heading.
The classic tag reader runs beside it as a check: if both name a station and
they differ, nothing is published and the docking holds. The node also
publishes the reader's state (OK / ALERT / FAULT with the reason and the
light check) on `/tugger/marker_reader` while a marker is in view; the
separate `marker_reader` node that did this before was removed, so the model
does not run twice.

Two more faults found on the way, both fixed at their cause:
- *Heading:* the fitted frame is an exact square, the face-on case in which
  OpenCV's IPPE returned the wrong one of its two pose solutions (182 px
  reprojection error instead of 0.03 px). The heading now comes from the
  checked SQPNP pose already used for the frame position: within 0.27 degrees
  of the tags' heading on 195 clean test images (median 0.001 degrees).
- *Simulation slowed by 20 %:* the model ran on every camera image, also in
  the aisles. It now runs only when at least 3,000 red pixels are in view
  (every measured frame had 7,800 or more; the plain floor 0).

Five-station docking cycles in the isolated simulation (LED ring on):

| Run | Docking error (mm) | Wall time per trip (s) |
|---|---|---|
| Classic reader | 3.6, 2.7, 4.4, 5.7, 5.6 | 217, 251, 324, 268, 209 |
| v3, model on every image | 2.7, 3.0, 3.2, 3.7, 5.0 | 257, 308, 403, 332, 259 |
| v3 with red-pixel gate | 4.8, 5.3, 5.5, 5.4, HOME timeout * | 214, 265, 332, 277, - |
| v3 with red-pixel gate, again | 1.2, 2.7, 6.6, 5.7, 5.5 | 211, 256, 352, 289, 216 |

All docks within the 10 mm requirement; v3 matches the classic reader in the
simulation (where markers are clean). Its gain is on worn markers (82 % read
against 59 %, see above), which the simulated stations do not have yet.
Readers never disagreed; no frame pose was rejected.

\* Not a reading fault: after setting the cart down at the southeast station
the robot did not move in Gazebo for 15 minutes (ground truth fixed to
0.1 mm while the wheels turned at -0.2 m/s and odometry counted 45 cm). It
did not happen in the other three runs. Open item: an intermittent hold of
the robot after DETACH (cargo joint or contact in the physics), to be looked
at separately.
