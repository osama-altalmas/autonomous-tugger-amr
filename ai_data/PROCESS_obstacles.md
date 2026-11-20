# AI stage 2: obstacle recognition (YOLO11n)

Record of how the obstacle recogniser of the autonomous tugger is built:
what was tried, what went wrong, what was changed and why. Datasets and
models stay on the development laptop; the repository holds this record,
its charts and the result summaries. Stage 1 (floor-marker reader):
[PROCESS.md](PROCESS.md).

## Goal

Today the robot knows only *that* something is in its way, from the laser
scanners: standing or moving, and how wide. It should also know *what* it
is, and react to that:

| Class | Reaction (user, 2026-11-16) |
|---|---|
| person | Keep 50 cm, stop and wait; never drive around a person, even a standing one. After 30 s another aisle may be taken |
| pallet, cart | Drive around as today (wait 5 s, 0.5 m gap) |
| forklift (and pallet jack) | Keep a larger distance and wait: it can move suddenly |
| not recognised | Today's behaviour |

The safety fields do not change: a protective stop never waits for the AI.

## Decisions (user, 2026-11-16)

- **Sensors:** the lidars only *detect and locate* objects, accurately, as
  today (tracks, safety fields). They do not recognise anything (no leg
  detector). The cameras recognise; each camera detection is matched onto
  the lidar track at the same place. A track no camera sees stays
  "unknown".
- **Classes:** person, pallet, cart, forklift.
- **Detector:** YOLO11n boxes (not segmentation): the lidar and the depth
  image give distance and width; the box only has to say what it is.
- **CPU budget:** 2-3 images per second, only while the robot moves: the
  front camera when driving forward, the rear one when reversing.

## The cameras

The depth cameras (front and rear, RealSense D435 class) give 160 x 120
pixels and see to 4 m: enough for the safety and obstacle point clouds,
too little for recognition (a person 3 m away is about 15 pixels wide).
Raising their resolution would have made the point clouds 16 times larger,
which the safety monitor reads. Instead a **separate colour camera** was
added beside each depth camera, as the D435 has its own RGB sensor: 640 x
480, the same 87 degree view, drawn to 12 m, 5 images per second. Its ROS
bridge is *lazy*: Gazebo renders it only while a node subscribes, so it
costs nothing when unused. The camera sits 20 cm above the floor; a person
1 m away is seen from about the knees down, which the model has to learn.

## Models of the objects

Gazebo Fuel (Open Robotics' model library), CC0 or CC-BY-4.0, downloaded
into Gazebo's cache by `tools/ai/obstacle_dataset/download_models.sh`
(about 110 MB, not in the repository):

| Class | Training | Held-out test only |
|---|---|---|
| person | Standing person, Walking person, Scrubs, OpScrubs | Casual female |
| pallet | Euro pallet, Euro pallet with stacked boxes | Pallet_Standard |
| cart | Utility Cart, Hand_Cart, BMWCart, TrolleyBox1 | BPCart, ParkingTrolleyMin |
| forklift | Forklift, Pallet_Jack | (same two, other places) |

Animated walking people (four actors) were tried and dropped: Gazebo's
`set_pose` does not move actors, so they never appeared where they were
placed. Models marked "non-commercial, no derivatives" were not used.

## Labels without hand work

Gazebo's **bounding-box camera** sits at the same place as the colour
camera and reports the visible 2D box of every object that carries a
class label (the `Label` system), with hidden parts already taken off. A
first test scene (2026-11-16) showed correct boxes for a person, a flat
pallet, a hand cart, a forklift and a pallet jack, partly behind each
other. (The sensors' own `save` option wrote nothing: Gazebo renders a
camera only while something subscribes; the generator subscribes.)

## Dataset v1 (`obstacles_v1/`, `obstacles_v1_test/`)

`tools/ai/obstacle_dataset/generate.py` runs Gazebo on its own (no robot
stack) with the real warehouse world, the camera rig and the labelled
models parked outside the building. For each image: the rig goes to a
random free spot of the warehouse map (from the map's free space, at least
0.45 m from racks and walls) with a random heading; 0-4 objects go to free
spots 0.6-8 m in front of it, a little beyond the image edges too (cut-off
objects), at random headings; the class is drawn first, so the four classes
come up equally often; the light changes (0.35-1.2). About 15 % of the
images have no object: racks, walls, columns, station legs and stands,
which must not be called obstacles. Objects hidden behind racks get no box,
as they are not visible.

Speed on the laptop: 2.4 images per second (5,000 in about 35 minutes).

| Set | Images | Boxes | Images with no box |
|---|---|---|---|
| `obstacles_v1` (seed 31) | 5,000: 4,233 train, 767 validation | 6,344: 2,197 person, 1,771 cart, 1,421 pallet, 955 forklift | 1,362 |
| `obstacles_v1_test` (held out, seed 97, unseen people/pallet/cart models) | 1,000 | 1,297: 445 person, 352 cart, 287 pallet, 213 forklift | 290 |

More images without a box than the planned 15 %: objects behind racks are
not visible and rightly get none. Forklifts are fewer: their 1.4 m footprint
fits fewer free spots. Examples: `obstacles_v1/previews/labelled_examples.png`
(people seen from the knees down when close, small ones far away, objects
cut by the image edge), and `first_bbox_camera_test.png`.

## Model v1: results (2026-11-16)

Trained on Colab (T4, `train_obstacles_v1_colab.ipynb`): YOLO11n from the
COCO weights, 640 px, 100 epochs, early stopping after 20 epochs without
gain (not reached), left-right flips, mosaic, scale and colour changes. The
training and validation losses fall together to the end, the validation
loss at or under the training one: no sign of memorising
(`obstacles_v1/previews/v1_overfitting_check.png`; the drop at epoch 90 is
mosaic being switched off for the last 10 epochs, as Ultralytics does).

| mAP50-95 | person | pallet | cart | forklift | all |
|---|---|---|---|---|---|
| validation (models seen in training) | 0.95 | 0.93 | 0.90 | 0.93 | 0.93 |
| held-out test (unseen people, pallet, carts) | 0.88 | **0.06** | 0.59 | 0.80 | 0.59 |

The held-out test found what the validation score hid: **v1 does not know
pallets**, only the two pallet models it was trained on. Training and test
crops side by side (`v1_pallets_train_vs_heldout.png`) show why: the Fuel
"Euro pallet" has no wood texture (plain grey, checked in its mesh file), the
other training pallet is hidden under a large carton; v1 never saw an empty
wooden pallet. The confusion matrix (`v1_heldout_confusion.png`) shows the
wooden test pallets found but called **forklift** (82 %): brown slats like
the forklift's and pallet jack's. Carts: half of the unseen carts missed
(recall 0.54) after only four cart shapes, and most false alarms on racks and
floor are called "cart". People and forklifts carry over well.

Model files (laptop): `models/obstacles_v1/` (`best.pt`, single-file
`best.onnx` made on the laptop, which OpenCV 4.10 loads; the Colab records
under `training/`; scores in `heldout_test/summary.json`).

## Dataset v2: variety where v1 failed

The fix aims at the cause (too few pallet and cart shapes and colours), not
at the test: the held-out models stay out of training and the **v1 test set
is reused unchanged**, so v1 and v2 are compared on the same images.

- **Pallets built from parts** in the generator: block pallets (EUR-like:
  bottom boards, nine blocks, cross boards, deck), stringer pallets (US-like:
  three long stringers under deck boards) and plastic pallets (plate on feet
  or runners); sizes 1200 x 800, 1200 x 1000, 1219 x 1016, 800 x 600 and
  square ones; seven wood shades with per-board variation, five plastic
  colours; 30 % carry a load (carton stacks or a wrapped block), as the
  held-out pallet is empty. Ten per world, a new world every 1,500 images:
  40 pallets.
- **Colour made useless, so shape decides** (user, 2026-11-16: "the colour
  shouldn't be the factor"): every cart, the forklift and the pallet jack
  also come as a recoloured copy (one copy original, one recoloured, new
  colours in every world). A first try painted each model one flat colour,
  which turned the one-piece forklift into a silhouette (tyres and mast the
  same colour); it was stopped after 400 images. The copies now keep their
  texture with only its coloured parts shifted in hue, saturation and
  brightness (also inside the `.glb` files), so tyres, steel and detail stay.
  Only the untextured grey Euro pallet gets a flat colour (wood or plastic).
- The station floor markers now appear in the images (v1 left them blank:
  the capture world did not find their pictures), so the model also learns
  them as no obstacle.
- **Five more carts** from Fuel (rigid-suspension trolley, MiR cart,
  TrolleyBox2, surgical trolley, mop cart), none like the held-out ones.
- **Distractors without a class**: cardboard box, cone, barrel, bin, crate;
  0-2 per image, so a box on the floor is not a pallet and not a cart.
- Same training settings as v1, so only the data changes
  (`train_obstacles_v2_colab.ipynb`).

| Set | Images | Boxes | Images with no box |
|---|---|---|---|
| `obstacles_v2` (seed 41, 4 worlds) | 6,000: 5,052 train, 948 validation | 7,864: 2,662 person, 2,120 cart, 1,927 pallet, 1,155 forklift | 1,575 |

Examples: `obstacles_v2/previews/labelled_examples.png`.

## Model v2: results (2026-11-16)

Same settings as v1. Colab's free GPU ran out at epoch 89; the run was
finished on Kaggle (free T4) from the saved `last.pt`, epochs 90-100.
(Two Kaggle-only fixes: copy the 7,000 input files 32 at a time, as Kaggle's
input storage is slow per file; switch off Ultralytics' Ray Tune link, as
Kaggle's Ray version does not match.) Losses again fall together, no
memorising (`obstacles_v2/previews/v2_overfitting_check.png`).

| Held-out test, mAP50-95 | person | pallet | cart | forklift | all |
|---|---|---|---|---|---|
| v1 | 0.88 | 0.06 | 0.59 | 0.80 | 0.59 |
| v2 at epoch 88 (laptop check) | 0.88 | 0.68 | 0.65 | 0.89 | 0.78 |
| **v2 final** | **0.89** | **0.71** | **0.65** | **0.90** | **0.79** |

Validation (seen models): 0.93. What is left, looked at image by image:

Each held-out object compared with the model's boxes (OpenCV running
`best.onnx`, as on the robot; full frames with true and predicted boxes were
looked at too):

- **Held-out pallets** (287): 151 get only a pallet box, 40 a pallet box
  *and* a forklift box, 42 only a forklift box, 54 nothing
  (`v2_pallet_failures.png`). The "forklift only" ones are the closest
  (median 262 px wide), where the wood grain shows most; the missed ones are
  small and far (median 81 px). The forklift does not carry a pallet; the
  likely cause is the surface: the held-out pallet has a photographed wood
  texture, the built pallets are flat-coloured boxes, and in training only
  the forklift, pallet jack and carts had such detailed textures.
- **"False" cart boxes** (204): 198 lie inside a real cart: a second,
  smaller box on part of a cart already found (mostly the lower platform of
  the unseen BPCart). Only 2 are on nothing. A first reading of the crops
  (`v2_false_cart_alarms.png`) took them for the warehouse's cargo stands;
  the full frames showed they are the held-out BPCart. For the robot these
  are harmless (both boxes say "cart" on the same lidar track); for the
  score they count as false alarms.

## Data audit and literature check (2026-11-17)

The user asked whether the training data are reliable, whether the best way
to train such a model had been researched (it had not: v1 and v2 followed
general practice), and what the data lack.

**Labels.** Correct and tight in a random sample of full images (Gazebo
computes them). But 410 boxes (5 %) were thinner than 12 px: fragments the
camera barely sees (a strip of a person behind a rack edge, a piece of a cart
between two tyres, far pallets that are a faint line). 293 of them pallets,
15 % of all pallet labels. 26 % of the images had no box (guidance: 0-10 %).

**What the literature says.**
- Domain randomisation (Tobin et al.; Tremblay et al., NVIDIA 2018):
  randomise textures, light, pose, camera, add distractors; synthetic
  training plus fine-tuning on a little real data beats real data alone.
- A 2025 study (YOLOv8, synthetic only) found realistic materials, image
  noise and blur, and distractors to matter most (removing distractors cost
  up to 7.5 points, removing noise and blur up to 6.4).
- Pallet detection with Omniverse (2025): synthetic data alone was not
  enough; training on synthetic and then fine-tuning on real data was best;
  strongly randomised data helped most on unseen conditions.
- Ultralytics: about 10,000 instances per class, every object labelled,
  tight boxes, 0-10 % background images. v2 had 1,155-2,662 per class.
- Real data: LOCO (TU Munich, CC0): real photos from five logistics sites
  with pallets, pallet trucks and forklifts.

**What the data lacked** (most important first): real images anywhere
(training and test were from the same simulator); more than one environment
(the same red rack blocks, grey floor and white walls in training and test);
pallet surfaces (flat colours, no wood); distinct models (4 people, all
standing; the 2 forklift models were also in the test, so forklift's 0.90 was
not a held-out score); camera realism (no blur, exposure or white balance
changes, fixed camera height and tilt); situations (no lying person, no
pallets on racks); quantity.

## Dataset v3 (user: "do it", 2026-11-17)

Generator v3 (`generate.py`; v2 frozen as `generate_v2.py`):
- **Pallets with wood-grain or plastic textures** (made per world: grain
  streaks, knots, darker edges, wear).
- **Forklifts and pallet jacks built from parts** (counterbalance forklifts
  with mast, forks at random height, overhead guard; manual and electric
  pallet jacks; proportions and colours random). The Fuel Forklift and
  Pallet_Jack are now **test-only**.
- **Pallet racking** in six of eight worlds in place of the red blocks:
  uprights, beams, decks, two to four levels with loaded pallets (labelled,
  about a third of the loads) and cartons. The first world keeps the
  original look, which the robot's own simulation uses; one more keeps the
  blocks in another colour.
- **Floor, walls and columns change colour** per world.
- **Camera**: height 0.18-0.21 m, pitch +-3 deg, roll +-2 deg; images get
  motion blur, defocus, exposure, white balance, contrast, noise and JPEG
  compression, each at random strength. Saved as JPEG (quality 95).
- **No boxes thinner than 10 px.**
- **More people**: Nurse, FemaleVisitor, MaleVisitorOnPhone, and Survivor
  Male lying on the floor (a fallen worker); two hand trucks as carts.
- Training set: 12,000 images in 8 worlds. **v3 test**: 1,500 images in two
  worlds (original blocks; racking with cartons only, so every test pallet
  is the held-out model), with held-out people (Casual female, Survivor
  Female lying), pallet, carts, forklift and pallet jack.

**Real images** (`tools/ai/obstacle_dataset/loco_to_yolo.py`): LOCO, 5,097
annotated photos scaled to 640 px. pallet -> pallet; forklift and pallet
truck -> forklift; small load carriers and stillages unlabelled. People are
not annotated in LOCO, so the COCO-trained YOLO11m labels them (confidence
0.5 or more); otherwise the model would learn that a person is background.
Split by site as LOCO's own split: subsets 2, 3, 5 train (12 % of it
validation), **subsets 1 and 4 (other sites) are the real held-out test**.

Training (`train_obstacles_v3_kaggle.ipynb`, Kaggle): synthetic and real
mixed, same settings as v1 and v2. Scored on the v3 test, the v1 test (for
comparison) and the real LOCO test.

### What the v3 data contains (generated 2026-11-17)

Synthetic run: 12,000 + 1,500 images, about 1.7 images/s (about 2 h on the
laptop), no stalls, no skipped images, no missing textures in the Gazebo
logs. Worlds: 1 original, 6 racking, 1 recoloured blocks. Example sheet:
`obstacles_v3/previews/labelled_examples.png`.

Boxes per class (images; background images = no labelled object):

| Set | Images | Background | Person | Pallet | Cart | Forklift |
|---|---|---|---|---|---|---|
| v3 train | 10,222 | 1,218 | 4,845 | 30,266 | 4,093 | 1,945 |
| v3 val | 1,778 | 206 | 815 | 5,539 | 744 | 364 |
| LOCO train | 2,468 | 77 | 551* | 40,231 | - | 1,853 |
| LOCO val | 352 | 8 | 83* | 5,570 | - | 227 |
| **Training total** | 14,820 | 1,509 (10 %) | 6,294 | 81,606 | 4,837 | 4,389 |
| v3 test | 1,500 | 380 | 568 | 432 | 633 | 319 |
| LOCO test (real) | 2,277 | 49 | 404* | 41,095 | - | 1,273 |

\* people labelled automatically by YOLO11m. A check sheet showed correct
boxes, including a forklift driver, a person behind a wire cage and people cut
off at the edge (LOCO blurs faces).

Notes:
- **Pallets far outnumber the rest**, mostly from LOCO's racking photos (up
  to dozens of pallets per photo). Every other class still has more than
  4,000 boxes, in the range Ultralytics advises. If carts or forklifts score
  worse than in v2, this imbalance is the first suspect.
- Background share is 10 %, at the top of the advised 0-10 %.
- **Roll cages**: real warehouses have wheeled roll cages. LOCO does not label
  them as any of our classes, so in the real photos they count as
  background. That may teach the model that roll cages are not carts. To be
  checked in the results.
- 33,621 LOCO boxes were dropped as thinner than 10 px at 640 px width
  (mostly far pallets seen edge-on in racks).

## Model v3: results (2026-11-17)

Trained on Kaggle (T4), 100 epochs at about 2.5 min each (about 5 h), never
stopped early. Validation loss fell to the end without turning up: no
overfitting (`obstacles_v3/previews/v3_overfitting_check.png`). Model and
records: `models/obstacles_v3/` (`best.pt`, single-file `best.onnx` checked
against PyTorch with OpenCV, `training/`).

mAP50-95 per test set (v1 and v2 scored on the new tests on the laptop):

| Test | Model | All | Person | Pallet | Cart | Forklift |
|---|---|---|---|---|---|---|
| v1 test (old) | v1 | 0.59 | 0.88 | 0.06 | 0.59 | 0.80 |
| | v2 | 0.79 | 0.89 | 0.71 | 0.65 | 0.90* |
| | **v3** | 0.78 | **0.90** | **0.84** | 0.65 | 0.71 |
| v3 test (all models unseen) | v1 | 0.15 | 0.21 | 0.00 | 0.06 | 0.34 |
| | v2 | 0.13 | 0.23 | 0.01 | 0.04 | 0.22 |
| | **v3** | **0.71** | **0.82** | **0.77** | **0.60** | **0.64** |
| LOCO real photos (2 unseen sites) | v1 | 0.00 | 0.01 | 0.00 | - | 0.00 |
| | v2 | 0.00 | 0.00 | 0.00 | - | 0.00 |
| | **v3** | **0.27** | **0.39** | **0.28** | - | **0.15** |

\* v1 and v2 trained on the same Forklift and Pallet_Jack models that are in
the v1 test (`generate_v2.py` marks them "both"). Their forklift score there
was partly memorisation. v3 never saw them, so its 0.71 is the honest number.

What this shows:
- **v1 and v2 only worked in the world they were made in.** On real photos
  they score zero and call almost everything "forklift". On the v3 test
  (new models, racking, camera effects) they fall to 0.13-0.15. The v1 test
  shared their look, which hid this.
- **v3 holds up on unseen models (0.71) and is the first model that works
  at all on real photos (0.27; 0.55 at the looser mAP50).**
- On the real photos the model **does not confuse the classes** (the
  confusion matrix is near diagonal; cart is never predicted, so roll cages
  are not taken for carts). **The weakness is misses**: 39 % of pallets,
  37 % of people and **64 % of real forklifts and pallet trucks** go
  unfound. 97 % of its false boxes are pallets, many probably real pallets
  whose boxes were dropped as thinner than 10 px
  (`obstacles_v3/previews/v3_loco_real_confusion.png`).
- People in the LOCO test are labelled by YOLO11m, so that score is
  agreement with YOLO11m, not with hand labels.
- Carts are the weakest synthetic class (0.60-0.65 on unseen carts).

Recall by object size, the YOLO-World comparison and the review of every weak spot with the
plan for the next version: `AI_REVIEW.md` (2026-11-18, kept on the development laptop).
Tool: `tools/ai/obstacle_dataset/recall_by_size.py`.

## Targets for the next versions (decided before training, 2026-11-18)

| Measure | v3 | Target |
|---|---|---|
| Near person found, real photos | 82 % | >= 95 % |
| Near forklift found, real photos | 61 % | >= 80 % |
| Near pallet found, real photos (100-200 px) | 76 % | >= 85 % |
| Synthetic test mAP50-95 | 0.71 | >= 0.75 |
| Real test mAP50 | 0.55 | >= 0.65 |
| CPU time per image | 161 ms | <= 200 ms |

If a target is missed, the record says so and why. Nothing is tuned against the test sets.
User decisions: roll cages count as "cart"; real data may come from open (CC BY) sets; a
larger YOLO11s is used only if it beats YOLO11n.

## Real data: what was checked and what was kept (2026-11-18)

Every set was opened and a labelled sample sheet looked at before use.

| Set | Verdict | Why |
|---|---|---|
| Roboflow "Forklift" (University of Maryland) | **kept** (`rf_forklift_umd`) | real counterbalance forklifts from ground level, many angles |
| Roboflow "KP_FullFrame" | **kept in part** (`rf_kp`) | reach trucks, workers and pallets with complete labels; video frames, so only 546 distinct frames kept |
| COCO train2017 people | **kept** (`coco_people`) | 3,000 photos with 1-6 fully labelled people, open licences only; keeps the model good at people |
| Roboflow "tnedh" | **kept** (`rf_tnedh`, CC BY 4.0) | 120 real hand pallet jacks seen from floor level |
| "Warehouse Safety", "trolley", "Cart Detection", "Warehouse Detection", several others | rejected | ceiling cameras, wrong objects, unlabelled people, or copies of the LOCO test photos (which would make the real test meaningless) |

In every kept set, people missing from the labels were added automatically (YOLO11m +
YOLO-World), and boxes thinner than 10 px were greyed out.

## Dataset v4 and model v4n (2026-11-18)

The generator (`generate_v4.py`) builds more objects from parts with random sizes and
colours: platform carts with thin push handles, roll cages, shelf trolleys, reach
trucks, walk-behind stackers and order pickers, loads on the forks of every truck.
Training: YOLO11n on Kaggle, 100 epochs, with repeat-factor sampling for rare classes.

| Test | v3 mAP50 / mAP50-95 | v4n mAP50 / mAP50-95 |
|---|---|---|
| LOCO real photos (sites 1 + 4) | 0.534 / 0.264 | 0.560 / 0.299 |
| Synthetic v4 test | 0.840 / 0.713 | 0.898 / 0.787 |

Against the targets: synthetic met; real photos, near people (89 %), near forklifts
(57 %) and pallets (75 %) not met. **Not good enough.**

**Why (miss analysis, `miss_analysis.py`):**
- **Forklifts:** most missed real forklifts had no box of any class near them: the
  model did not see them. Nearly all are **hand pallet trucks** (26 % found), while big
  orange powered trucks were found. Training had real pallet trucks from only 3
  sites and 4 crude built pallet jacks.
- **Pallets:** the misses were pallets standing on edge against walls and pallets in
  racks; the generator built no standing pallets.
- **People:** 91 % of near people found; the misses are partly hidden people and a few
  wrong test labels (the automatic person labels boxed two hanging jackets).

## Dataset v5 and model v5n, the model in use (2026-11-19 / 20)

Changes, each aimed at a measured miss:
- **Pallet trucks built from parts:** manual hand jacks, electric walk-behind and rider
  trucks, ten colours, 40 % loaded; 10 per world, plus 4 unseen kinds in the test.
- **Pallets standing on edge** (upright or leaning), 4 per world.
- **A 4th real site:** LOCO site 4 moved into training (user decision); **site 1 stays
  the unseen-site test**.
- Real sets: all kept sets above.

| Set | Images | person | pallet | cart | forklift |
|---|---|---|---|---|---|
| obstacles_v5 train / val | 10,212 / 1,788 | 4,610 / 872 | 32,396 / 5,847 | 3,978 / 683 | 2,293 / 385 |
| obstacles_v5_test (unseen models) | 1,500 | 573 | 448 | 646 | 367 |
| LOCO train / val (sites 2-5) | 3,539 / 483 | 860 / 124 | 67,106 / 8,319 | - | 2,450 / 306 |
| LOCO test (site 1, unseen) | 1,075 | 193 | 11,471 | - | 597 |

Training: Kaggle T4, 100 epochs in 6.1 h, no overfitting in the loss plot.

| Test | v3 | v4n | **v5n** |
|---|---|---|---|
| Real photos, unseen site (LOCO site 1), mAP50 / mAP50-95 | 0.551 / 0.296 | 0.582 / 0.336 | **0.591 / 0.339** |
| ... near people found | 79 % | 91 % | 91 % |
| ... near forklifts found | 58 % | 53 % | 55 % |
| Synthetic v5 test, mAP50 / mAP50-95 | 0.761 / 0.623 | 0.820 / 0.692 | **0.879 / 0.747** |
| ... near forklifts found (unseen pallet trucks) | 81 % | 87 % | **94 %** |

(This table uses the v5 tests: LOCO site 1 only. The targets above were measured on
the older real test with sites 1 and 4, so the v3 numbers differ slightly.)

**The built pallet trucks fixed pallet trucks in simulation, but not at the real
unseen site.** Measured why, on the same model:

| v5n, forklift class found | Held-out images of the training sites | Unseen site 1 |
|---|---|---|
| overall | 68 % | 24 % |
| > 200 px | 88 % | 55 % |

This is a **cross-site gap**: real pallet trucks at a new warehouse are limited by the
number of real sites in the data (4, mostly one truck brand each), not by the number
of images, and the open real data is used up. The standard fix is a few hundred
labelled images from the robot's own warehouse at deployment (needs hardware). The
safety fields do not depend on this class: they stop for every object.

**Export and speed** (laptop CPU, 640 × 640): single-file ONNX (opset 18; opset 17
gave an invalid node), outputs equal to PyTorch within 0.008. Time per image: OpenCV
DNN 140 ms, ONNX Runtime 55 ms. The robot uses ONNX Runtime when installed, else
OpenCV (ROS's Python has no ONNX Runtime by default). Model file:
`src/tugger_control/models/obstacles_v5n.onnx`.

## The recognition node in the robot (`obstacle_recognition.py`)

The scanners and depth cameras find and track objects (`obstacle_monitor`); the node
names them:

1. **Which camera:** 2.5 images per second from the camera that has a tracked object
   in view; with objects in both or neither, the camera facing the driving direction.
2. **Match:** from the object's tracked position and the camera's pose at the moment
   the image was taken, the node computes where the object must be in the image
   (columns and the row where it stands on the floor). A detection there gives the
   class. (Its decoder gives the same boxes as Ultralytics: 163 of 163 on test images.)
3. **Vote:** a class is kept after two sightings within 3 s. One sighting means
   "being recognised": the robot waits for the second before it decides.

How the robot reacts to each class is written in the main README (Obstacles). In
short: people and forklifts are waited for and never driven around; pallets and
carts are driven around; an object the cameras could not name keeps the full 50 cm,
because it could be a person.

## Tests in the simulation (2026-11-20)

Gazebo Fuel models (a standing person, a Euro pallet) were put on the routes; every
run starts from a full reset (`tools/mission_validation/obstacle_world.py`,
`run_one.sh`). The recogniser writes what it saw in every frame
(`TUGGER_RECOGNITION_DEBUG`), which made each fault visible.

| Campaign | What went wrong | Root cause and fix |
|---|---|---|
| debug run | the person was never named | not the model: the scanners saw the person round a corner, outside the cameras' view, and the robot decided before the cameras could look. Fix: **look before deciding** (drive on slowly until the cameras have looked) |
| 1-2 | a standing pallet was taken for a **moving** object | the scanners saw the pallet in pieces that merged and split; the track centre jumped. Fix: pieces close together are one object; later the tracker was rebuilt (**one track per real object**), and growth into view no longer counts as movement |
| 3 | a person was named late; every replan found "no other route" | the camera pose was taken at the latest time, not at the image's time (15 px off while turning); the live A* path ended 3.5 cm off the station entry and was always rejected. Fixes: camera pose at the image time; path ends placed exactly |
| 3-4 | too close to go around after looking; a crash in the new back-up | Fix: **back up, then go around** (user decision), each candidate checked with its own heading |
| 5-6 | the back-up never helped | detours sized the pallet by its circle (1.40 m) instead of its outline (0.8 m); the robot stopped 15 cm short of the back-up point. Fixes: outline-based sizing, 0.25 m margin |
| **7** | - | **5/5 SUCCESS**, docked 1.0-5.5 mm; the corner pallet: looked, backed up 0.99 m, went around |

**Half cycle with the Gazebo window open** (HOME → Station 3 → Station 4 → HOME, two
pallets and a person on the routes): the first try showed a person passed slowly
for about 1 s before its class arrived (two sightings 3.2 s apart while turning), and
a test-tool time limit that ran out in wall-clock time. After the fixes (recognition
in progress is not judged, unnamed objects keep 50 cm, the camera facing the object
looks, time limit in simulated time): **3/3 legs SUCCESS**, docked 3.8-5.1 mm.

Measured in these runs:

| | Person | Euro pallet (14 cm high) |
|---|---|---|
| Named correctly in camera frames | 88-100 % from 3 to 10 m | 93 % at 3-4 m; 56 % at 2-3 m; 33 % at 4-5 m |
| Usually named at | about 6 m (at most 8 m) | about 3 m |

## Weak spots and next steps

- **Low grey pallets seen from far down** are a thin line in the image (about 14-20 px
  high at 2.5 m): sometimes not named at all (from behind: best guess "cart" 14 % on
  a clean frame, below the 35 % threshold) or named "forklift". Checked with clean
  frames: a model weakness, not a software bug. The robot then uses the safe rule.
  Next: more training images of exactly this view; later perhaps a high-resolution
  camera with a cut-out around the expected object, or a camera tilted down (both
  need new training images; the user keeps the camera as it is for now).
- **Real forklifts and pallet trucks at a new site:** images from that site.
- **YOLO11s** on the same v5 data, to see if a larger model closes part of the
  cross-site gap (waits for the Kaggle GPU quota).
- The simulated colour cameras are 87° wide; a real RealSense D435 colour camera is 69°.
