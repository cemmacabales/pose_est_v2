# Data Collection Checklist for the Team

## What We're Doing
We need to record exercise videos to teach our pose estimation model to recognize 9 different rehab exercises correctly. Think of it like teaching a smart camera what "good form" and "bad form" look like.

**We need 5 people to be filmed doing exercises.** Each person will be recorded doing several exercises. The more different people we have, the smarter the model gets.

---

## How to Record a Video (Important!)

1. **Camera setup:** Put your phone on a tripod or lean it against something. Don't hold it.
2. **Position:** Stand about 6-8 feet away from the camera. Your whole body (head to toes) must be visible.
3. **Lighting:** Face the light source. Don't stand with a bright window behind you.
4. **Background:** Plain wall is best. Avoid busy patterns.
5. **Clothing:** Wear fitted clothes (t-shirt and shorts/pants). Avoid baggy hoodies or dresses.
6. **Duration:** Each video should be **15–30 seconds** — do 3–5 repetitions of the exercise.
7. **No talking** during the recording.
8. **Format:** MP4 (default on iPhone/Android is fine).

---

## Naming Your Video Files (Very Important!)

After you record, rename the file BEFORE sending it. Use this exact format:

```
<exercise_number>_<correct_or_incorrect>_<your_initials>_<take_number>.mp4
```

Examples:
- `01_correct_JD_1.mp4` = Deep Squat, good form, John Doe's first take
- `01_incorrect_JD_1.mp4` = Deep Squat, bad form, John Doe's first take
- `07_correct_SM_2.mp4` = Shoulder Abduction, good form, Sarah Miller's second take

**Your initials = first and last name initials. If there's a conflict, use first 3 letters of first name.**

---

## The 9 Exercises

| Number | Exercise Name | What It Is |
|--------|--------------|------------|
| 01 | Deep Squat | Squat down as low as you can, stand back up |
| 02 | Hurdle Step | Step over an imaginary hurdle, bring knee up high |
| 03 | Inline Lunge | Step forward into a lunge, back foot in line with front |
| 04 | Side Lunge | Step out to the side and squat on one leg |
| 05 | Sit to Stand | Sit in a chair, stand up without using hands |
| 06 | Standing Leg Raise | Stand on one leg, lift the other straight out to the side |
| 07 | Shoulder Abduction | Raise arms straight out to the sides, up to shoulder height |
| 08 | Shoulder Extension | Reach both arms straight back behind you |
| 10 | Shoulder Scaption | Raise arms diagonally forward-up (like a V shape) |

**Note:** Exercise 09 (Shoulder Rotation) is skipped for now. Don't record it.

---

## What "Correct" vs "Incorrect" Means

For EVERY exercise, you will record **6 takes of CORRECT form** and **4 takes of INCORRECT form**.

### What is "Correct"?
Do the exercise the way a physical therapist would teach it. Smooth, controlled, full range of motion.

### What is "Incorrect"? (Bad Form)
Below are SPECIFIC mistakes to show for each exercise. **Pick ONE mistake per incorrect video.** Don't do multiple mistakes at once.

---

## Exercise-Specific Mistakes for "Incorrect" Videos

### 01 — Deep Squat
**Correct:** Hips go below knee level, heels stay down, knees track over toes.

**Incorrect mistakes (pick one per video):**
- [ ] Knees cave inward toward each other
- [ ] Don't go down far enough (stop above parallel)
- [ ] Heels lift off the ground
- [ ] Lean your chest way forward
- [ ] One side goes lower than the other (uneven)

### 02 — Hurdle Step
**Correct:** Lift knee high, step over imaginary barrier, keep hips level.

**Incorrect mistakes (pick one per video):**
- [ ] Let your hip drop or stick out on one side
- [ ] Knee collapses inward as you step
- [ ] Lose balance and wobble
- [ ] Don't lift knee high enough (lazy step)

### 03 — Inline Lunge
**Correct:** Front knee stays over ankle, torso upright, back knee lowers toward ground.

**Incorrect mistakes (pick one per video):**
- [ ] Front knee goes past your toes
- [ ] Lean your upper body way forward
- [ ] Back knee doesn't lower enough
- [ ] Wobble or lose balance

### 04 — Side Lunge
**Correct:** Step wide to the side, keep the bent knee over the foot, other leg stays straight.

**Incorrect mistakes (pick one per video):**
- [ ] Bent knee caves inward
- [ ] Don't go down low enough
- [ ] Heel of the bent leg lifts up
- [ ] Lean your torso to the side instead of staying upright

### 05 — Sit to Stand
**Correct:** Stand up from chair without using hands, hips fully extend at top.

**Incorrect mistakes (pick one per video):**
- [ ] Push off the chair with your hands
- [ ] Lean your chest way forward to stand up
- [ ] Come up unevenly (favor one leg)
- [ ] Don't fully straighten hips at the top

### 06 — Standing Leg Raise
**Correct:** Stand tall, lift one leg straight out to the side without leaning.

**Incorrect mistakes (pick one per video):**
- [ ] Hike your hip up on the standing side (hip goes up instead of leg going out)
- [ ] Lean your torso to the opposite side
- [ ] Bend the knee of the leg you're lifting
- [ ] Lose balance and put foot down

### 07 — Shoulder Abduction
**Correct:** Raise both arms straight out to the sides, up to shoulder height, no shrugging.

**Incorrect mistakes (pick one per video):**
- [ ] Shrug your shoulders up toward your ears
- [ ] Lean your trunk to compensate
- [ ] Don't raise arms all the way to shoulder height
- [ ] Turn palms down or rotate arms inward

### 08 — Shoulder Extension
**Correct:** Reach both arms straight back behind your body, squeeze shoulder blades.

**Incorrect mistakes (pick one per video):**
- [ ] Arch your lower back (stick your belly out)
- [ ] Jut your head forward (chin pokes out)
- [ ] Don't reach back very far (small motion)
- [ ] Rotate your trunk to one side

### 10 — Shoulder Scaption
**Correct:** Raise arms diagonally forward-up at about 30 degrees from straight front, thumbs up.

**Incorrect mistakes (pick one per video):**
- [ ] Shrug shoulders up
- [ ] Rotate your trunk to reach higher
- [ ] Don't raise arms high enough
- [ ] Go straight forward (like a front raise) instead of diagonal

---

## Why We Need Different Angles / Variations

**Important:** If you record 6 identical videos from the same angle, the model won't learn much. Each "correct" video should be **slightly different** so the model learns to recognize the exercise no matter how the camera is positioned.

### Variations to Use (spread across your 6 correct takes):
1. **Camera angle:** Frontal (facing camera), Lateral (side view), 45-degree diagonal
2. **Distance:** Close (4–5 feet), Medium (6–8 feet), Far (10–12 feet)
3. **Speed:** Normal speed, Slightly slower and controlled, Slightly faster
4. **Leading side:** For one-sided exercises (lunges, leg raises), do some leading with left leg/arm, some with right
5. **Environment:** Slightly different spot in the room, different background wall

**You don't need to change everything every time.** Just pick 2–3 variations per video and mix them up.

---

## Split Among 5 People

**GOAL: 150 videos total. Each person records 30 videos.**

Each person will be the SUBJECT (the person being filmed) for their assigned exercises. **You also need to help film the other people on your team.**

### Person 1 (Subject: YOU)
**Record these exercises:**

**01 Deep Squat — Correct:**
- [ ] Deep Squat (`01_correct_P1_1.mp4`) — Frontal view, normal speed
- [ ] Deep Squat (`01_correct_P1_2.mp4`) — Lateral (side) view, normal speed
- [ ] Deep Squat (`01_correct_P1_3.mp4`) — 45° diagonal view, normal speed
- [ ] Deep Squat (`01_correct_P1_4.mp4`) — Frontal view, slower & controlled
- [ ] Deep Squat (`01_correct_P1_5.mp4`) — Lateral view, slightly faster
- [ ] Deep Squat (`01_correct_P1_6.mp4`) — Farther from camera, frontal view

**01 Deep Squat — Incorrect:**
- [ ] Deep Squat (`01_incorrect_P1_1.mp4`) — Knees cave inward
- [ ] Deep Squat (`01_incorrect_P1_2.mp4`) — Heels lift off ground
- [ ] Deep Squat (`01_incorrect_P1_3.mp4`) — Don't go down far enough
- [ ] Deep Squat (`01_incorrect_P1_4.mp4`) — Lean chest way forward

**02 Hurdle Step — Correct:**
- [ ] Hurdle Step (`02_correct_P1_1.mp4`) — Frontal view, lead with left leg
- [ ] Hurdle Step (`02_correct_P1_2.mp4`) — Lateral view, lead with left leg
- [ ] Hurdle Step (`02_correct_P1_3.mp4`) — Frontal view, lead with right leg
- [ ] Hurdle Step (`02_correct_P1_4.mp4`) — 45° diagonal view, lead with right leg
- [ ] Hurdle Step (`02_correct_P1_5.mp4`) — Frontal view, slower speed
- [ ] Hurdle Step (`02_correct_P1_6.mp4`) — Farther from camera, lateral view

**02 Hurdle Step — Incorrect:**
- [ ] Hurdle Step (`02_incorrect_P1_1.mp4`) — Hip drops on one side
- [ ] Hurdle Step (`02_incorrect_P1_2.mp4`) — Knee collapses inward
- [ ] Hurdle Step (`02_incorrect_P1_3.mp4`) — Lose balance and wobble
- [ ] Hurdle Step (`02_incorrect_P1_4.mp4`) — Don't lift knee high enough

**03 Inline Lunge — Correct:**
- [ ] Inline Lunge (`03_correct_P1_1.mp4`) — Frontal view, lead with left leg
- [ ] Inline Lunge (`03_correct_P1_2.mp4`) — Lateral view, lead with left leg
- [ ] Inline Lunge (`03_correct_P1_3.mp4`) — Frontal view, lead with right leg
- [ ] Inline Lunge (`03_correct_P1_4.mp4`) — 45° diagonal view, lead with right leg
- [ ] Inline Lunge (`03_correct_P1_5.mp4`) — Slower and controlled, frontal view
- [ ] Inline Lunge (`03_correct_P1_6.mp4`) — Farther from camera, lateral view

**03 Inline Lunge — Incorrect:**
- [ ] Inline Lunge (`03_incorrect_P1_1.mp4`) — Front knee goes past toes
- [ ] Inline Lunge (`03_incorrect_P1_2.mp4`) — Lean upper body forward
- [ ] Inline Lunge (`03_incorrect_P1_3.mp4`) — Back knee doesn't lower enough
- [ ] Inline Lunge (`03_incorrect_P1_4.mp4`) — Wobble and lose balance

**Total for Person 1: 30 videos**

---

### Person 2 (Subject: YOU)
**Record these exercises:**

**01 Deep Squat — Correct:**
- [ ] Deep Squat (`01_correct_P2_1.mp4`) — Frontal view, normal speed
- [ ] Deep Squat (`01_correct_P2_2.mp4`) — Lateral (side) view, normal speed
- [ ] Deep Squat (`01_correct_P2_3.mp4`) — 45° diagonal view, normal speed
- [ ] Deep Squat (`01_correct_P2_4.mp4`) — Frontal view, slower & controlled
- [ ] Deep Squat (`01_correct_P2_5.mp4`) — Lateral view, slightly faster
- [ ] Deep Squat (`01_correct_P2_6.mp4`) — Farther from camera, frontal view

**01 Deep Squat — Incorrect:**
- [ ] Deep Squat (`01_incorrect_P2_1.mp4`) — Knees cave inward
- [ ] Deep Squat (`01_incorrect_P2_2.mp4`) — Heels lift off ground
- [ ] Deep Squat (`01_incorrect_P2_3.mp4`) — Don't go down far enough
- [ ] Deep Squat (`01_incorrect_P2_4.mp4`) — Lean chest way forward

**04 Side Lunge — Correct:**
- [ ] Side Lunge (`04_correct_P2_1.mp4`) — Frontal view, lunge to left side
- [ ] Side Lunge (`04_correct_P2_2.mp4`) — Lateral view, lunge to left side
- [ ] Side Lunge (`04_correct_P2_3.mp4`) — Frontal view, lunge to right side
- [ ] Side Lunge (`04_correct_P2_4.mp4`) — 45° diagonal view, lunge to right side
- [ ] Side Lunge (`04_correct_P2_5.mp4`) — Slower and controlled, frontal view
- [ ] Side Lunge (`04_correct_P2_6.mp4`) — Farther from camera, lateral view

**04 Side Lunge — Incorrect:**
- [ ] Side Lunge (`04_incorrect_P2_1.mp4`) — Bent knee caves inward
- [ ] Side Lunge (`04_incorrect_P2_2.mp4`) — Don't go down low enough
- [ ] Side Lunge (`04_incorrect_P2_3.mp4`) — Heel of bent leg lifts up
- [ ] Side Lunge (`04_incorrect_P2_4.mp4`) — Lean torso to the side

**05 Sit to Stand — Correct:**
- [ ] Sit to Stand (`05_correct_P2_1.mp4`) — Frontal view, normal chair height
- [ ] Sit to Stand (`05_correct_P2_2.mp4`) — Lateral view, normal chair height
- [ ] Sit to Stand (`05_correct_P2_3.mp4`) — 45° diagonal view, normal chair height
- [ ] Sit to Stand (`05_correct_P2_4.mp4`) — Frontal view, slower & controlled
- [ ] Sit to Stand (`05_correct_P2_5.mp4`) — Frontal view, slightly faster
- [ ] Sit to Stand (`05_correct_P2_6.mp4`) — Farther from camera, lateral view

**05 Sit to Stand — Incorrect:**
- [ ] Sit to Stand (`05_incorrect_P2_1.mp4`) — Push off chair with hands
- [ ] Sit to Stand (`05_incorrect_P2_2.mp4`) — Lean chest way forward
- [ ] Sit to Stand (`05_incorrect_P2_3.mp4`) — Come up unevenly (favor one leg)
- [ ] Sit to Stand (`05_incorrect_P2_4.mp4`) — Don't fully straighten hips at top

**Total for Person 2: 30 videos**

---

### Person 3 (Subject: YOU)
**Record these exercises:**

**02 Hurdle Step — Correct:**
- [ ] Hurdle Step (`02_correct_P3_1.mp4`) — Frontal view, lead with left leg
- [ ] Hurdle Step (`02_correct_P3_2.mp4`) — Lateral view, lead with left leg
- [ ] Hurdle Step (`02_correct_P3_3.mp4`) — Frontal view, lead with right leg
- [ ] Hurdle Step (`02_correct_P3_4.mp4`) — 45° diagonal view, lead with right leg
- [ ] Hurdle Step (`02_correct_P3_5.mp4`) — Frontal view, slower speed
- [ ] Hurdle Step (`02_correct_P3_6.mp4`) — Farther from camera, lateral view

**02 Hurdle Step — Incorrect:**
- [ ] Hurdle Step (`02_incorrect_P3_1.mp4`) — Hip drops on one side
- [ ] Hurdle Step (`02_incorrect_P3_2.mp4`) — Knee collapses inward
- [ ] Hurdle Step (`02_incorrect_P3_3.mp4`) — Lose balance and wobble
- [ ] Hurdle Step (`02_incorrect_P3_4.mp4`) — Don't lift knee high enough

**03 Inline Lunge — Correct:**
- [ ] Inline Lunge (`03_correct_P3_1.mp4`) — Frontal view, lead with left leg
- [ ] Inline Lunge (`03_correct_P3_2.mp4`) — Lateral view, lead with left leg
- [ ] Inline Lunge (`03_correct_P3_3.mp4`) — Frontal view, lead with right leg
- [ ] Inline Lunge (`03_correct_P3_4.mp4`) — 45° diagonal view, lead with right leg
- [ ] Inline Lunge (`03_correct_P3_5.mp4`) — Slower and controlled, frontal view
- [ ] Inline Lunge (`03_correct_P3_6.mp4`) — Farther from camera, lateral view

**03 Inline Lunge — Incorrect:**
- [ ] Inline Lunge (`03_incorrect_P3_1.mp4`) — Front knee goes past toes
- [ ] Inline Lunge (`03_incorrect_P3_2.mp4`) — Lean upper body forward
- [ ] Inline Lunge (`03_incorrect_P3_3.mp4`) — Back knee doesn't lower enough
- [ ] Inline Lunge (`03_incorrect_P3_4.mp4`) — Wobble and lose balance

**06 Standing Leg Raise — Correct:**
- [ ] Standing Leg Raise (`06_correct_P3_1.mp4`) — Frontal view, raise left leg
- [ ] Standing Leg Raise (`06_correct_P3_2.mp4`) — Lateral view, raise left leg
- [ ] Standing Leg Raise (`06_correct_P3_3.mp4`) — Frontal view, raise right leg
- [ ] Standing Leg Raise (`06_correct_P3_4.mp4`) — 45° diagonal view, raise right leg
- [ ] Standing Leg Raise (`06_correct_P3_5.mp4`) — Slower and controlled, frontal view
- [ ] Standing Leg Raise (`06_correct_P3_6.mp4`) — Farther from camera, lateral view

**06 Standing Leg Raise — Incorrect:**
- [ ] Standing Leg Raise (`06_incorrect_P3_1.mp4`) — Hike hip up on standing side
- [ ] Standing Leg Raise (`06_incorrect_P3_2.mp4`) — Lean torso to opposite side
- [ ] Standing Leg Raise (`06_incorrect_P3_3.mp4`) — Bend knee of lifted leg
- [ ] Standing Leg Raise (`06_incorrect_P3_4.mp4`) — Lose balance and put foot down

**Total for Person 3: 30 videos**

---

### Person 4 (Subject: YOU)
**Record these exercises:**

**04 Side Lunge — Correct:**
- [ ] Side Lunge (`04_correct_P4_1.mp4`) — Frontal view, lunge to left side
- [ ] Side Lunge (`04_correct_P4_2.mp4`) — Lateral view, lunge to left side
- [ ] Side Lunge (`04_correct_P4_3.mp4`) — Frontal view, lunge to right side
- [ ] Side Lunge (`04_correct_P4_4.mp4`) — 45° diagonal view, lunge to right side
- [ ] Side Lunge (`04_correct_P4_5.mp4`) — Slower and controlled, frontal view
- [ ] Side Lunge (`04_correct_P4_6.mp4`) — Farther from camera, lateral view

**04 Side Lunge — Incorrect:**
- [ ] Side Lunge (`04_incorrect_P4_1.mp4`) — Bent knee caves inward
- [ ] Side Lunge (`04_incorrect_P4_2.mp4`) — Don't go down low enough
- [ ] Side Lunge (`04_incorrect_P4_3.mp4`) — Heel of bent leg lifts up
- [ ] Side Lunge (`04_incorrect_P4_4.mp4`) — Lean torso to the side

**06 Standing Leg Raise — Correct:**
- [ ] Standing Leg Raise (`06_correct_P4_1.mp4`) — Frontal view, raise left leg
- [ ] Standing Leg Raise (`06_correct_P4_2.mp4`) — Lateral view, raise left leg
- [ ] Standing Leg Raise (`06_correct_P4_3.mp4`) — Frontal view, raise right leg
- [ ] Standing Leg Raise (`06_correct_P4_4.mp4`) — 45° diagonal view, raise right leg
- [ ] Standing Leg Raise (`06_correct_P4_5.mp4`) — Slower and controlled, frontal view
- [ ] Standing Leg Raise (`06_correct_P4_6.mp4`) — Farther from camera, lateral view

**06 Standing Leg Raise — Incorrect:**
- [ ] Standing Leg Raise (`06_incorrect_P4_1.mp4`) — Hike hip up on standing side
- [ ] Standing Leg Raise (`06_incorrect_P4_2.mp4`) — Lean torso to opposite side
- [ ] Standing Leg Raise (`06_incorrect_P4_3.mp4`) — Bend knee of lifted leg
- [ ] Standing Leg Raise (`06_incorrect_P4_4.mp4`) — Lose balance and put foot down

**07 Shoulder Abduction — Correct:**
- [ ] Shoulder Abduction (`07_correct_P4_1.mp4`) — Frontal view, normal speed
- [ ] Shoulder Abduction (`07_correct_P4_2.mp4`) — Lateral view, normal speed
- [ ] Shoulder Abduction (`07_correct_P4_3.mp4`) — 45° diagonal view, normal speed
- [ ] Shoulder Abduction (`07_correct_P4_4.mp4`) — Frontal view, slower & controlled
- [ ] Shoulder Abduction (`07_correct_P4_5.mp4`) — Frontal view, slightly faster
- [ ] Shoulder Abduction (`07_correct_P4_6.mp4`) — Farther from camera, lateral view

**07 Shoulder Abduction — Incorrect:**
- [ ] Shoulder Abduction (`07_incorrect_P4_1.mp4`) — Shrug shoulders up
- [ ] Shoulder Abduction (`07_incorrect_P4_2.mp4`) — Lean trunk to compensate
- [ ] Shoulder Abduction (`07_incorrect_P4_3.mp4`) — Don't raise arms all the way
- [ ] Shoulder Abduction (`07_incorrect_P4_4.mp4`) — Turn palms down / rotate inward

**Total for Person 4: 30 videos**

---

### Person 5 (Subject: YOU)
**Record these exercises:**

**05 Sit to Stand — Correct:**
- [ ] Sit to Stand (`05_correct_P5_1.mp4`) — Frontal view, normal chair height
- [ ] Sit to Stand (`05_correct_P5_2.mp4`) — Lateral view, normal chair height
- [ ] Sit to Stand (`05_correct_P5_3.mp4`) — 45° diagonal view, normal chair height
- [ ] Sit to Stand (`05_correct_P5_4.mp4`) — Frontal view, slower & controlled
- [ ] Sit to Stand (`05_correct_P5_5.mp4`) — Frontal view, slightly faster
- [ ] Sit to Stand (`05_correct_P5_6.mp4`) — Farther from camera, lateral view

**05 Sit to Stand — Incorrect:**
- [ ] Sit to Stand (`05_incorrect_P5_1.mp4`) — Push off chair with hands
- [ ] Sit to Stand (`05_incorrect_P5_2.mp4`) — Lean chest way forward
- [ ] Sit to Stand (`05_incorrect_P5_3.mp4`) — Come up unevenly (favor one leg)
- [ ] Sit to Stand (`05_incorrect_P5_4.mp4`) — Don't fully straighten hips at top

**08 Shoulder Extension — Correct:**
- [ ] Shoulder Extension (`08_correct_P5_1.mp4`) — Frontal view, normal speed
- [ ] Shoulder Extension (`08_correct_P5_2.mp4`) — Lateral view, normal speed
- [ ] Shoulder Extension (`08_correct_P5_3.mp4`) — 45° diagonal view, normal speed
- [ ] Shoulder Extension (`08_correct_P5_4.mp4`) — Frontal view, slower & controlled
- [ ] Shoulder Extension (`08_correct_P5_5.mp4`) — Frontal view, slightly faster
- [ ] Shoulder Extension (`08_correct_P5_6.mp4`) — Farther from camera, lateral view

**08 Shoulder Extension — Incorrect:**
- [ ] Shoulder Extension (`08_incorrect_P5_1.mp4`) — Arch lower back
- [ ] Shoulder Extension (`08_incorrect_P5_2.mp4`) — Jut head forward (chin pokes out)
- [ ] Shoulder Extension (`08_incorrect_P5_3.mp4`) — Don't reach back very far
- [ ] Shoulder Extension (`08_incorrect_P5_4.mp4`) — Rotate trunk to one side

**10 Shoulder Scaption — Correct:**
- [ ] Shoulder Scaption (`10_correct_P5_1.mp4`) — Frontal view, normal speed
- [ ] Shoulder Scaption (`10_correct_P5_2.mp4`) — Lateral view, normal speed
- [ ] Shoulder Scaption (`10_correct_P5_3.mp4`) — 45° diagonal view, normal speed
- [ ] Shoulder Scaption (`10_correct_P5_4.mp4`) — Frontal view, slower & controlled
- [ ] Shoulder Scaption (`10_correct_P5_5.mp4`) — Frontal view, slightly faster
- [ ] Shoulder Scaption (`10_correct_P5_6.mp4`) — Farther from camera, lateral view

**10 Shoulder Scaption — Incorrect:**
- [ ] Shoulder Scaption (`10_incorrect_P5_1.mp4`) — Shrug shoulders up
- [ ] Shoulder Scaption (`10_incorrect_P5_2.mp4`) — Rotate trunk to reach higher
- [ ] Shoulder Scaption (`10_incorrect_P5_3.mp4`) — Don't raise arms high enough
- [ ] Shoulder Scaption (`10_incorrect_P5_4.mp4`) — Go straight forward instead of diagonal

**Total for Person 5: 30 videos**

---

## Team Totals Summary

| Person | Exercises | Correct Videos | Incorrect Videos | Total |
|--------|-----------|----------------|------------------|-------|
| Person 1 | Deep Squat, Hurdle Step, Inline Lunge | 18 | 12 | **30** |
| Person 2 | Deep Squat, Side Lunge, Sit to Stand | 18 | 12 | **30** |
| Person 3 | Hurdle Step, Inline Lunge, Standing Leg Raise | 18 | 12 | **30** |
| Person 4 | Side Lunge, Standing Leg Raise, Shoulder Abduction | 18 | 12 | **30** |
| Person 5 | Sit to Stand, Shoulder Extension, Shoulder Scaption | 18 | 12 | **30** |
| **TEAM TOTAL** | | **90** | **60** | **150** |

---

## How Many Videos Per Exercise?

After all 5 people finish:

| Exercise | Correct Videos | Incorrect Videos | Total | People Who Did It |
|----------|---------------|------------------|-------|-------------------|
| Deep Squat | 12 | 8 | **20** | Person 1, Person 2 |
| Hurdle Step | 12 | 8 | **20** | Person 1, Person 3 |
| Inline Lunge | 12 | 8 | **20** | Person 1, Person 3 |
| Side Lunge | 12 | 8 | **20** | Person 2, Person 4 |
| Sit to Stand | 12 | 8 | **20** | Person 2, Person 5 |
| Standing Leg Raise | 12 | 8 | **20** | Person 3, Person 4 |
| Shoulder Abduction | 6 | 4 | **10** | Person 4 only |
| Shoulder Extension | 6 | 4 | **10** | Person 5 only |
| Shoulder Scaption | 6 | 4 | **10** | Person 5 only |

**Note:** Shoulder exercises (Abduction, Extension, Scaption) currently only have 1 person each in this plan. If possible, **Person 1, 2, or 3 should record 3–4 extra shoulder videos** to increase diversity. But 10 videos per shoulder exercise is still much better than before.

---

## Train / Validation / Test Split (For the Technical Person)

When all 150 videos come back, here's how we'll split them:

- **Training set:** Videos from Person 1 + Person 2 + Person 3 (~90 videos)
- **Validation set:** Videos from Person 4 (~30 videos)
- **Test set:** Videos from Person 5 (~30 videos)

**Why split by person instead of random?** Because we want to test if the model recognizes exercises from someone it has NEVER seen before. This is the real-world test.

With 150 videos, we expect to generate approximately **150,000–200,000 training windows** after augmentation (flipping, noise, time stretch). This is a very solid dataset for a small LSTM model.

---

## How to Submit Your Videos

1. Record all your assigned videos using your phone
2. Rename each file using the exact format above
3. Put them in a folder named: `Person1_videos` (or whatever person number you are)
4. Upload to the shared Google Drive / Dropbox / file sharing link
5. **Do NOT send via email or text — files are too big**
6. **Do NOT commit videos to GitHub**

---

## Quick Checklist Before You Start Recording

- [ ] My phone is fully charged or plugged in
- [ ] Camera is stationary (tripod, stack of books, etc.)
- [ ] I am 6–8 feet away from camera
- [ ] My full body is visible (head to toes)
- [ ] I am facing the light, not backlit
- [ ] Background is plain
- [ ] I am wearing fitted clothing
- [ ] I know which exercises I'm recording today
- [ ] I have this checklist open to check off videos as I finish them

---

## Questions?

If something is unclear, ask the technical lead before recording. It's easier to clarify now than to re-record later!
