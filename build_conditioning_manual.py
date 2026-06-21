#!/usr/bin/env python3
"""
build_conditioning_manual.py

Generates references/conditioning_manual.pdf — a coaching-tone exercise guide
covering all 9 movements in the classifier's training set.

Content draws on functional movement principles (grounded in FM 7-22 physical
readiness methodology) and the specific form criteria used when labeling the
training videos for this project.

Run with:  python build_conditioning_manual.py
Requires:  pip install fpdf2
"""

from pathlib import Path
from fpdf import FPDF

OUT_DIR = Path("references")
OUT_PATH = OUT_DIR / "exercise_form_guide.pdf"

TITLE = "Functional Movement Conditioning Manual"
SUBTITLE = "Exercise Form, Cues & Corrections for the Exercise Coaching System"

# ---------------------------------------------------------------------------
# Content
# ---------------------------------------------------------------------------

INTRO = """
Welcome to your functional movement guide. This manual covers the nine exercises
that form the foundation of your training programme: Deep Squat, Hurdle Step,
Inline Lunge, Side Lunge, Sit to Stand, Standing Leg Raise, Shoulder Abduction,
Shoulder Scaption, and Shoulder Extension.

Functional movement is about quality before quantity. Every repetition you perform
with good form builds neuromuscular patterns that carry over into daily life and
sport. Every repetition performed with sloppy form — even if it "feels fine" —
reinforces compensation patterns that will eventually lead to pain or injury.

The goal of this guide is simple: help you understand what each movement should
look and feel like, name the most common errors people make, and give you clear
cues to self-correct. Think of it less as a rulebook and more as a conversation
with a knowledgeable training partner who has your back.
"""

FOUNDATION_PRINCIPLES = {
    "Neutral Spine": (
        "A neutral spine isn't ramrod-straight — it's the natural S-curve your "
        "back makes when you stand tall without actively arching or rounding. "
        "For most exercises in this programme, maintaining a neutral spine "
        "throughout the movement is the single biggest predictor of whether the "
        "rep is good or not. Imagine a belt running from your navel to your lower "
        "back — a slight tension in that belt keeps your lumbar spine in a safe "
        "position under load."
    ),
    "Hip Hinge vs Knee Bend": (
        "Two of the most important patterns in functional movement are the hip hinge "
        "(bending at the hips while the spine stays long) and the knee bend (squatting "
        "by lowering the hips as the knees track forward). Most people over-flex the "
        "knees and under-use the hips. Learning to distinguish these patterns — and "
        "blend them appropriately — is the key to exercises like the Deep Squat, "
        "Sit to Stand, and the lunge variations."
    ),
    "Bracing and Breathing": (
        "Before each repetition, take a comfortable breath in and brace your core "
        "lightly — as if someone is about to gently poke your stomach. This intra-"
        "abdominal pressure protects your spine throughout the movement. Exhale "
        "smoothly through the effort phase (typically the upward or outward phase "
        "of the movement). Never hold your breath for multiple reps — it raises blood "
        "pressure unnecessarily and creates tension that limits your range of motion."
    ),
    "Foot Position and Weight Distribution": (
        "Your feet are your foundation. For standing exercises, your weight should be "
        "distributed evenly across the full foot — heel, ball of the foot, and little "
        "toe. Lifting your heels, rolling inward (pronation), or gripping the floor "
        "with your toes are all signs that something further up the chain isn't working "
        "correctly. Pay attention to your feet and you will often catch problems before "
        "they show up in your knees or hips."
    ),
    "Controlled Tempo": (
        "Moving slowly is not the same as moving weakly. A controlled three-second "
        "lowering phase (eccentric) followed by a one-to-two-second return is the "
        "sweet spot for building strength and motor control simultaneously. Fast, "
        "ballistic reps might feel more impressive, but they mask compensations and "
        "dramatically increase injury risk. Unless a programme specifically calls for "
        "explosive movement, default to slow and deliberate."
    ),
    "Symmetry and Balance": (
        "Most people have one side that is dominant — stronger, more flexible, or more "
        "coordinated. Single-leg and single-arm movements reveal these asymmetries "
        "immediately. If you find yourself leaning, rotating, or otherwise compensating "
        "more on one side, note it. Over time, unilateral training (one limb at a time) "
        "is one of the most effective tools for evening out these imbalances."
    ),
}

EXERCISES = [
    {
        "name": "Deep Squat",
        "overview": (
            "The deep squat is the foundational lower-body movement pattern. It tests "
            "and develops mobility through the hips, knees, and ankles simultaneously, "
            "while demanding that the thoracic spine and shoulders stay open and upright. "
            "The Army's physical readiness framework (FM 7-22) uses squat-pattern "
            "movements as the cornerstone of lower-body conditioning precisely because "
            "they recruit the largest muscle groups in the body — glutes, quads, and "
            "hamstrings — in a coordinated, functional way."
        ),
        "setup": (
            "Stand with your feet about shoulder-width apart, toes pointing slightly "
            "outward (roughly 10-30 degrees — whatever feels natural for your hip anatomy). "
            "Your arms can hang at your sides or reach forward for balance. Take a breath "
            "and brace your core lightly."
        ),
        "cues": [
            "Push your knees out over your toes as you descend — they should track in the "
            "same direction your toes are pointing.",
            "Keep your chest tall. Imagine someone is pulling a string attached to the "
            "crown of your head upward.",
            "Drive your hips back and down — think of sitting into a low chair that is "
            "just behind you.",
            "Keep your heels flat on the floor throughout. If they want to rise, you may "
            "need to work on ankle mobility (calf stretches help).",
            "At the bottom, your hips should ideally be at or below the height of your "
            "knees. If you cannot reach this depth without your lower back rounding, stop "
            "just above the point where rounding begins.",
            "Drive through your heels to stand, squeezing your glutes at the top.",
        ],
        "errors": [
            ("Knees caving inward (valgus collapse)",
             "Push your knees actively outward as you descend. Think 'spread the floor' "
             "with your feet — try to screw your feet into the ground without actually "
             "moving them. Strengthening your glute medius with side-lying clamshells "
             "and band walks between sessions will also help."),
            ("Heels lifting off the floor",
             "This usually means limited ankle dorsiflexion. Elevate your heels slightly "
             "on a small plate or folded mat as a temporary workaround, and add daily "
             "calf stretches (both straight-leg and bent-knee versions) to improve "
             "long-term mobility."),
            ("Not reaching adequate depth (not going down far enough)",
             "Depth comes from hip and ankle mobility. Work on the goblet squat hold "
             "(hold a counterweight in front of your chest and squat into the bottom "
             "position, rocking gently). Consistency over weeks is what moves the needle."),
            ("Chest collapsing forward (excessive forward lean)",
             "This is usually a hip flexor tightness or thoracic mobility issue. Open "
             "up your chest before the set with a few shoulder circle and chest stretch "
             "repetitions. During the squat, keep your arms out in front of you at "
             "shoulder height to give yourself a reference for torso position."),
        ],
        "rep_guidance": (
            "Perform 8-15 repetitions per set. Start with bodyweight only until you can "
            "consistently hit full depth with a neutral spine and flat heels. The Deep Squat "
            "is assessed in the Functional Movement Screen (FMS) as a measure of bilateral, "
            "symmetrical mobility and stability of the hips, knees, and ankles — meaning "
            "quality matters far more than load."
        ),
    },
    {
        "name": "Hurdle Step",
        "overview": (
            "The Hurdle Step assesses and trains single-leg stance stability, hip flexor "
            "mobility, and pelvis control. It mimics the mechanics of walking and running "
            "gait — specifically the moment when your trail leg swings forward. FM 7-22 "
            "emphasises stepping drills and single-leg balance work as essential for "
            "mobility and coordination on varied terrain, which is exactly what this "
            "movement targets."
        ),
        "setup": (
            "Stand tall with feet together, arms relaxed at your sides or lightly extended "
            "for balance. Imagine an imaginary hurdle at approximately knee height set "
            "directly in front of you."
        ),
        "cues": [
            "Lift one knee up toward your chest, keeping your foot dorsiflexed (toes pulled "
            "toward your shin).",
            "Step over the imaginary hurdle by rotating from the hip — the movement is "
            "hip flexion and extension, not just a knee lift.",
            "Keep your pelvis level. Your hip on the lifted-leg side should not drop lower "
            "than the standing-leg hip.",
            "The standing leg stays straight and strong — no bending the knee of the "
            "support leg.",
            "Return the foot to the starting position with control. Do not let it drop "
            "or crash down.",
            "Your torso should remain upright throughout — avoid leaning forward or "
            "sideways to compensate.",
        ],
        "errors": [
            ("Hip drop on the standing leg (Trendelenburg sign)",
             "This means your hip abductors (glute medius) are not strong enough to hold "
             "the pelvis level during single-leg stance. Add lateral band walks and "
             "single-leg glute bridges to your supplementary work. During the movement, "
             "consciously squeeze the glute of your standing leg."),
            ("Trunk sway or rotation",
             "You are compensating for limited hip mobility or core instability. Slow "
             "down and perform the step in a more controlled, smaller range until you "
             "can maintain a still torso. A good cue: keep your shoulders level and "
             "your bellybutton pointing forward throughout."),
            ("Foot turning out (eversion) or toe drop",
             "Keep your dorsiflexion (toes up toward shin) maintained throughout the step. "
             "Foot eversion often pairs with limited hip internal rotation — foam rolling "
             "the glutes and hip flexors between sessions can help."),
            ("Touching or not clearing the hurdle",
             "Focus on leading with the knee before the foot crosses the hurdle plane. "
             "Practice in front of a mirror to get visual feedback on your knee height "
             "and foot clearance."),
        ],
        "rep_guidance": (
            "Perform 8-12 repetitions per leg. Alternate legs each rep or complete all "
            "reps on one side before switching. The Hurdle Step is one of the clearest "
            "tests of left-right asymmetry — if one side feels significantly harder, "
            "give that side extra single-leg balance practice."
        ),
    },
    {
        "name": "Inline Lunge",
        "overview": (
            "The Inline Lunge challenges single-leg stability in the sagittal plane (front "
            "to back) while requiring coordinated hip and knee flexion under load. FM 7-22 "
            "includes lunge-pattern movements in nearly every conditioning drill as a "
            "fundamental transfer of force from the lower body to ground. The 'inline' "
            "version — where both feet land on the same narrow line — makes it harder "
            "than a standard lunge by reducing your base of support."
        ),
        "setup": (
            "Stand tall with feet together. Step forward with one foot, placing it directly "
            "in front of the other on an imaginary straight line. Both feet should be "
            "pointing forward (not flared out). Your feet are hip-width or narrower — "
            "this is the challenging part."
        ),
        "cues": [
            "Lower your back knee toward the floor in a controlled descent — aim to get it "
            "within a couple of inches of the ground without touching.",
            "Your front shin should be close to vertical, or only slightly forward. Avoid "
            "letting the front knee travel far past your front toes.",
            "Keep your torso upright throughout. Resist the urge to lean forward as you "
            "lower down.",
            "Drive through your front heel to return to standing, squeezing your glutes "
            "at the top.",
            "Both feet stay in line — this will feel unstable at first, which is exactly "
            "the point. Your hip stabilisers are working harder to control the narrow stance.",
        ],
        "errors": [
            ("Lateral wobble or trunk sway side-to-side during the Inline Lunge",
             "You are struggling to stabilise on a narrow base during the Inline Lunge. "
             "Start by practising a slightly wider stance Inline Lunge first, then "
             "progressively narrow it over sessions. Focus on gripping the floor with your "
             "toes to improve proprioception."),
            ("Trunk leaning forward",
             "Keep your chest tall and your hands on your hips to give yourself tactile "
             "feedback. Forward lean is often caused by tight hip flexors — add a kneeling "
             "hip flexor stretch to your warm-up."),
            ("Front knee collapsing inward",
             "Push your front knee outward to align with your second toe throughout the "
             "descent. Weak glutes are usually the root cause — add glute activation work "
             "before lunging."),
            ("Feet not staying in line",
             "Place a tape line on the floor and practise stepping along it. At first, aim "
             "for feet close together but not perfectly inline, and tighten the alignment "
             "progressively."),
        ],
        "rep_guidance": (
            "Perform 8-12 repetitions per leg. You can alternate legs each step or complete "
            "a full set on one side before switching. Add a light dumbbell in each hand "
            "once you can maintain perfect alignment for all reps."
        ),
    },
    {
        "name": "Side Lunge",
        "overview": (
            "The Side Lunge is the primary lateral movement pattern in functional training. "
            "While most exercises move you forward and backward, lateral strength and "
            "stability are equally important for athletic performance and injury prevention. "
            "FM 7-22 includes lateral squat movements in conditioning circuits to train "
            "the adductors and abductors that often get neglected in standard programmes. "
            "The Side Lunge also significantly challenges single-leg quad strength when "
            "you are at the bottom of the movement."
        ),
        "setup": (
            "Stand with feet together. Take a wide step directly to one side — roughly "
            "1.5 to 2 times your shoulder width. Both feet should point forward (or only "
            "slightly outward). Your weight is distributed across both feet initially."
        ),
        "cues": [
            "As you step out, shift your weight toward the stepping leg by pushing your "
            "hips back and bending that knee.",
            "The opposite (straight) leg stays long and extended. You should feel a "
            "stretch in the inner thigh of the straight leg -- that's normal and intentional.",
            "What you should feel during a correct Side Lunge: the primary sensation is a "
            "deep stretch through the inner thigh (adductor) of the straight leg as it "
            "extends outward. This is the adductor muscle group being lengthened under "
            "load. If you feel the stretch in the front of your hip or the back of your "
            "leg instead, check that your straight leg is fully extended and that you are "
            "pushing your hips back -- not forward -- as you lower.",
            "Keep your chest up. Push your hips back as you bend -- do not let the knee "
            "shoot forward first.",
            "Your bent knee should track over your toes (not cave inward).",
            "Your heel on the bent leg must stay flat on the floor throughout.",
            "Push off your bent-leg heel to return to center with control.",
        ],
        "errors": [
            ("Knee passing far in front of the toes on the bent leg",
             "Sit your hips back more aggressively as you step out. Think 'hips back first, "
             "then knee bends' rather than 'step and bend'. A useful cue: imagine sitting "
             "into a low chair that is directly beside you, not in front of you."),
            ("Heel lifting on the bent leg",
             "Limited ankle mobility is usually the culprit. Until ankle mobility improves, "
             "you can slightly elevate your heel on a small plate. Long-term, add daily "
             "ankle circles and calf stretches."),
            ("Chest dropping toward the floor at the bottom of the Side Lunge",
             "Your hip flexors and upper back are both limiting your ability to stay upright "
             "during the Side Lunge. Hold your arms out in front of you at shoulder height "
             "during the movement to act as a counterbalance and keep your chest tall."),
            ("Inner knee of bent leg caving inward",
             "This is glute medius weakness on the bent-leg side. Push that knee outward "
             "intentionally throughout the rep and add lateral band work between sessions."),
        ],
        "rep_guidance": (
            "Perform 8-12 repetitions per side. You can alternate sides or complete all "
            "reps on one side first. The Side Lunge is excellent as a mobility warm-up "
            "movement and as a standalone strength exercise with added load (dumbbell "
            "held at chest height)."
        ),
    },
    {
        "name": "Sit to Stand",
        "overview": (
            "The Sit to Stand is one of the most functionally significant movements in "
            "daily life — you do it dozens of times a day without thinking about it. "
            "Training it deliberately builds the hip and quad strength, balance, and "
            "coordination needed to perform this movement efficiently and safely over "
            "a lifetime. FM 7-22 physical readiness programming uses chair-rise variations "
            "in rehabilitation and general conditioning circuits precisely because it is "
            "a direct transfer to occupational and daily life demands."
        ),
        "setup": (
            "Sit at the front edge of a chair or bench with feet hip-width apart and flat "
            "on the floor. Your knees should be at roughly 90 degrees or slightly higher. "
            "Arms can be folded across your chest or extended forward for balance."
        ),
        "cues": [
            "Lean forward slightly from the hips — this shifts your centre of mass over "
            "your feet, which is necessary for standing. Do not round your back; hinge "
            "from the hips.",
            "Drive through both heels simultaneously to stand, squeezing your glutes as "
            "you reach full extension.",
            "Stand tall at the top — hips fully extended, shoulders back.",
            "To sit back down, push your hips back first (reverse the hip hinge), then "
            "lower with control until you make gentle contact with the chair. Do not "
            "drop or collapse onto the seat.",
        ],
        "errors": [
            ("Using arm momentum to swing forward to stand",
             "Fold your arms across your chest to remove this compensatory pattern. The "
             "movement should be driven purely from your hips and legs. If you cannot stand "
             "without arm swing, the chair may be too low — raise it and progressively "
             "lower it over weeks."),
            ("Rising asymmetrically (one side doing more work)",
             "Watch for the hips rotating or one knee pushing out more. Practise single-"
             "leg sit-to-stands (keeping one foot slightly raised) to identify and address "
             "the weaker side."),
            ("Back rounding as you lean forward to initiate the stand",
             "This is a hip hinge mobility and strength issue. Practice the hip hinge "
             "pattern separately — stand with feet hip-width, place your hands on your "
             "thighs, and practise bending forward from the hips while your spine stays "
             "long. Transfer this pattern into the Sit to Stand."),
            ("Knees caving inward during the stand",
             "Same correction as the Deep Squat — push the knees outward over the toes "
             "and strengthen the glutes."),
        ],
        "rep_guidance": (
            "Perform 10-15 repetitions per set. Progress from a higher chair to a lower "
            "one as your strength and control improve. The ultimate benchmark is rising "
            "smoothly from the floor — but the chair-based version builds the foundation "
            "to get there."
        ),
    },
    {
        "name": "Standing Leg Raise",
        "overview": (
            "The Standing Leg Raise tests and trains single-leg balance, hip flexor "
            "strength, and the ability to control the pelvis and spine under the challenge "
            "of holding one leg in the air. FM 7-22 balance and coordination drills "
            "feature single-leg stance extensively because the ability to maintain balance "
            "on one leg underpins almost every dynamic movement — walking, running, "
            "climbing, and landing from a jump."
        ),
        "setup": (
            "Stand tall with feet together. Choose a fixed point on the wall in front of "
            "you to focus on — this helps enormously with balance. Lightly rest your "
            "fingertips on a wall or chair back if you need initial support."
        ),
        "cues": [
            "Shift your weight onto one foot. Brace your core lightly and squeeze the "
            "glute of your standing leg.",
            "Raise the opposite leg forward — either with the knee bent (easier) or "
            "straight (harder). Aim for hip height or as high as your flexibility allows "
            "without compensating.",
            "Keep your pelvis level. The hip on the raised-leg side should not hike upward.",
            "Your torso should stay upright — resist leaning backward as you raise the leg.",
            "Hold the top position for a moment, then lower with control.",
        ],
        "errors": [
            ("Trunk leaning sideways or laterally during the Standing Leg Raise (leaning away from the raised leg)",
             "This is a sign that your hip flexors on the raised-leg side are struggling "
             "to lift the leg without recruiting the trunk as compensation. The torso leans "
             "away from the raised leg because the hip flexor is too weak to do the work "
             "alone. Reduce the height of the raise and focus on keeping your torso vertical. "
             "Add seated hip flexor strengthening (seated knee raises) to your programme."),
            ("Hip hiking on the raised-leg side during the Standing Leg Raise",
             "The hip should stay level during the Standing Leg Raise -- the pelvis does "
             "not tilt. Focus on the cue 'keep both hip bones at the same height'. "
             "Glute medius weakness on the standing side is often the cause of the hip "
             "rising on the raised-leg side."),
            ("Loss of balance or hopping to maintain position",
             "Balance improves with consistent practice. Reduce the range of motion and use "
             "fingertip support initially. Practise single-leg standing (without the leg "
             "raise component) for 20-30 seconds at a time as a precursor exercise."),
            ("Standing knee bending under load",
             "Your quadriceps need to hold the standing leg straight and strong. Actively "
             "contract your quad and squeeze your glute on the standing side throughout "
             "the movement."),
        ],
        "rep_guidance": (
            "Perform 10-15 repetitions per leg. Single-leg balance work is dose-dependent "
            "— even brief daily practice (2-3 sets of 10) produces noticeable gains in "
            "stability within a few weeks. Progress to raising the leg while performing "
            "light arm movements once balance is solid."
        ),
    },
    {
        "name": "Shoulder Abduction",
        "overview": (
            "Shoulder Abduction — raising both arms laterally away from the body — "
            "strengthens the deltoids and rotator cuff while testing thoracic and "
            "scapular (shoulder blade) mobility and control. FM 7-22 includes shoulder "
            "abduction in rehabilitative and preparatory drill sequences because poor "
            "scapular stability is one of the most common contributors to shoulder "
            "impingement and rotator cuff problems in both military and general populations. "
            "Performed correctly, this exercise builds the shoulder resilience that "
            "prevents injuries down the line."
        ),
        "setup": (
            "Stand tall with arms at your sides, palms facing your body. Feet are hip-"
            "width apart, knees soft (not locked out). Shoulders are relaxed — not "
            "hunched up toward your ears."
        ),
        "cues": [
            "Initiate the movement by gently retracting and depressing your shoulder blades "
            "(think: pull your shoulder blades together and then pull them down your back). "
            "This sets the scapula in a stable position before the arms move.",
            "Raise both arms out to the sides in a smooth arc, leading with the back of "
            "your hands (thumbs pointing downward or neutral).",
            "Stop when your arms are at shoulder height (90 degrees from your body). Going "
            "higher than this can compress the structures in the shoulder joint.",
            "Keep a very slight bend in your elbows throughout — do not lock them out or "
            "over-bend them.",
            "Lower with control over 2-3 seconds. Do not let gravity do the work.",
            "Your shoulders should not creep upward toward your ears. If they do, you are "
            "using your upper trapezius to compensate.",
        ],
        "errors": [
            ("Shoulder shrugging (upper trap dominance)",
             "This is the most common error. Before each rep, actively pull your shoulders "
             "away from your ears. A useful cue: imagine you are trying to put your shoulder "
             "blades into your back pockets. If the shrugging persists, reduce the load "
             "and focus on isolating the movement."),
            ("Arms going above shoulder height",
             "Stop at 90 degrees. Going beyond this rotates the humerus in a way that "
             "can pinch the rotator cuff tendons and bursa against the acromion bone "
             "(this is called subacromial impingement). Shoulder height is the target."),
            ("Trunk side bending to assist the raise",
             "Your body is compensating for weak shoulders by leaning. Keep your core "
             "engaged and torso perfectly vertical. Reduce the load if needed."),
            ("Arms internally rotating (thumbs pointing down at the top)",
             "Keep your thumbs neutral (pointing forward) or slightly upward during the "
             "lift. Internal rotation narrows the space in the shoulder joint."),
        ],
        "rep_guidance": (
            "Perform 10-15 repetitions per set with light resistance. Shoulder abduction "
            "is a precision movement — heavy loads and high reps work against you here. "
            "Focus on the quality of scapular retraction and controlled tempo above all else."
        ),
    },
    {
        "name": "Shoulder Scaption",
        "overview": (
            "Scaption — raising the arms in the scapular plane rather than directly to "
            "the side or directly to the front — is the most shoulder-friendly version of "
            "an overhead raise. The scapular plane sits at roughly 30 degrees in front of "
            "the body's lateral line, which is the natural resting position of the shoulder "
            "blade. Raising in this plane produces optimal muscle activation and the "
            "least joint compression. FM 7-22 rehabilitative shoulder protocols use the "
            "scapular-plane raise as a primary shoulder strengthening exercise for exactly "
            "this reason."
        ),
        "setup": (
            "Stand tall, feet hip-width. Hold your arms at your sides with thumbs pointing "
            "forward (supinated grip). Your starting position has your arms slightly in "
            "front of your body — roughly 30 degrees forward from your hips. This is "
            "already the scapular plane."
        ),
        "cues": [
            "Maintain the thumbs-up grip throughout the entire movement — this naturally "
            "externally rotates the shoulder, opening the joint space.",
            "Raise both arms simultaneously in the diagonal plane (30 degrees forward of "
            "lateral) to shoulder height.",
            "Keep your shoulder blades retracted and depressed — shoulders away from ears, "
            "as in the Abduction exercise.",
            "Stop at shoulder height (90 degrees). There is no benefit and real risk in "
            "going higher.",
            "Lower slowly and with control. Feel the muscles working on the way down — "
            "this eccentric loading is where much of the strengthening happens.",
        ],
        "errors": [
            ("Shoulder shrugging",
             "Same correction as Shoulder Abduction. Consciously depress your shoulder "
             "blades before and during the raise. If the shrug is unavoidable at a given "
             "load, reduce the weight or resistance."),
            ("Arms internally rotating (thumbs pointing down)",
             "Keep the thumbs-up position throughout. This is the defining characteristic "
             "of scaption that separates it from standard abduction. If your thumbs rotate "
             "inward, you lose the shoulder-protective benefit of the exercise."),
            ("Going above shoulder height",
             "90 degrees is the target. The rotator cuff has done its job at shoulder "
             "height — going beyond this changes the mechanics and increases impingement "
             "risk significantly."),
            ("Arms going too far to the side (becoming abduction instead of scaption)",
             "Keep the arms in the diagonal plane — about 30 degrees in front of true "
             "lateral. Practise in front of a mirror side-on so you can see the angle."),
        ],
        "rep_guidance": (
            "Perform 10-15 repetitions per set. Scaption can be performed with light "
            "dumbbells, resistance bands, or bodyweight only. It is particularly valuable "
            "as a warm-up exercise before upper body training and as rehabilitation "
            "following shoulder impingement."
        ),
    },
    {
        "name": "Shoulder Extension",
        "overview": (
            "Shoulder Extension moves the arm backward, behind the body's midline. It "
            "strengthens the posterior deltoid, latissimus dorsi, and teres major — the "
            "pulling muscles of the shoulder that are often undertrained relative to the "
            "pressing muscles. FM 7-22 conditioning protocols pair push and pull exercises "
            "to maintain balanced shoulder strength and prevent the rounded-shoulder "
            "posture that comes from too much pressing and not enough pulling. Strong "
            "shoulder extensors also contribute significantly to running efficiency "
            "through the arm-swing mechanics."
        ),
        "setup": (
            "Stand tall, feet hip-width, arms resting at your sides with palms facing "
            "back (or neutral). Shoulders are relaxed and slightly retracted."
        ),
        "cues": [
            "Keeping your arm straight (or with a slight elbow bend), sweep one or both "
            "arms backward in a controlled arc.",
            "The movement comes from the shoulder joint — your elbow does not drive it and "
            "your trunk should not rotate to extend the range.",
            "Go as far back as feels comfortable and controlled — typically 30-45 degrees "
            "behind your hip. Stop before your lower back begins to hyperextend to "
            "compensate.",
            "Hold the end position briefly, squeezing the muscles along the back of your "
            "shoulder and upper back.",
            "Return to the starting position with control — do not let the arm swing "
            "forward passively.",
        ],
        "errors": [
            ("Trunk leaning forward to create the appearance of greater range",
             "The extension should come entirely from the shoulder, not from hinging at "
             "the hips. Stand against a wall if needed — this gives you an immediate "
             "reference point for trunk position."),
            ("Shoulder shrugging upward during the extension",
             "Keep your shoulder depressed throughout. The upper trap does not need to "
             "assist in this movement — if it is, you may be trying to extend too far."),
            ("Excessive range (lower back hyperextending)",
             "Stay within the range where your torso stays neutral. Excessive extension "
             "is not better — it shifts the movement from the shoulder to the lumbar spine, "
             "which is not the intent."),
            ("Elbow bending significantly during the movement",
             "A very slight elbow bend is fine for comfort, but the arm should remain "
             "essentially straight. Bending the elbow converts the exercise into a "
             "tricep kickback rather than a shoulder extension."),
        ],
        "rep_guidance": (
            "Perform 10-15 repetitions per arm (unilateral) or both arms together "
            "(bilateral). Shoulder Extension is underrated as a posture corrector — "
            "two to three sets daily can help undo the rounded-shoulder effect of "
            "prolonged sitting and screen time."
        ),
    },
]

SESSION_DESIGN = """
Building a complete functional movement session does not require complicated programming.
A straightforward structure that works well for this exercise set is:

WARM-UP (5-8 minutes)
Spend five to eight minutes raising your core temperature and mobilising your joints
before loading any movement. Light jogging or brisk walking in place, arm circles,
leg swings, hip circles, and ankle rolls all qualify. A few easy reps of the Sit to
Stand and Standing Leg Raise make excellent final warm-up movements before moving
into the main session.

MAIN SESSION STRUCTURE
For most people, two to three sets of each exercise with 45-90 seconds rest between
sets is the right starting point. Do not rush the rest — the quality of each rep
matters more than the pace of the workout. A sensible session order:
  1. Deep Squat (lower body, bilateral)
  2. Hurdle Step (lower body, unilateral, hip mobility)
  3. Inline Lunge (lower body, unilateral, sagittal plane)
  4. Side Lunge (lower body, lateral plane)
  5. Sit to Stand (functional lower body)
  6. Standing Leg Raise (balance and hip flexor)
  7. Shoulder Abduction (upper body, lateral chain)
  8. Shoulder Scaption (upper body, scapular plane)
  9. Shoulder Extension (upper body, posterior chain)

COOL-DOWN (5-10 minutes)
End every session with gentle stretching of the muscles you have worked. Priority
stretches for this programme: hip flexor lunge stretch, pigeon pose or figure-four
stretch for the glutes, doorway chest stretch, and a simple cross-body stretch for
the posterior shoulder. Hold each stretch for 30-60 seconds and breathe slowly.

PROGRESSION PRINCIPLES
Add load (resistance bands, dumbbells, bodyweight progressions) only after you can
perform all reps of an exercise with consistent, clean technique. When you add load,
reduce your rep count initially and build back up. A two-week rule is useful: if you
cannot perform an exercise correctly for two sessions in a row, go back to the
previous progression level and spend more time there.

FREQUENCY
Three sessions per week with at least one rest day between sessions is ideal for
most people starting out. Two sessions per week is still highly effective. Daily
functional movement practice (bodyweight only, lower intensity) is also beneficial
and appropriate for most people as long as they listen to their body and do not
train through pain.
"""

CORRECTIONS_QUICK_REFERENCE = {
    "Knees caving inward in any lower body exercise": (
        "Push knees out over toes. Strengthen glute medius with lateral band walks "
        "and clamshells."
    ),
    "Heels lifting from floor": (
        "Work on ankle dorsiflexion. Stretch calves daily. Temporarily elevate heels "
        "as a workaround."
    ),
    "Shoulder shrugging": (
        "Retract and depress shoulder blades before initiating any shoulder movement. "
        "Reduce load if shrug persists."
    ),
    "Trunk leaning or rotating to assist limb movement": (
        "Core is not engaging or limb mobility is limited. Brace core more deliberately. "
        "Reduce range of motion until control improves."
    ),
    "Loss of balance in single-leg exercises": (
        "Use fingertip wall support temporarily. Practice static single-leg balance "
        "for 20-30 seconds daily. Build to dynamic single-leg work."
    ),
    "Arms going above shoulder height in shoulder exercises": (
        "Shoulder height is the ceiling. Mark the position with a reference point "
        "at eye level if helpful."
    ),
    "Not reaching full depth in squat movements": (
        "Work on hip flexor and ankle mobility. Use counterbalance (arms forward) "
        "to help maintain balance at depth."
    ),
    "One side significantly weaker or stiffer than the other": (
        "Give the weaker side extra unilateral practice. Do not load bilateral "
        "exercises heavily until the asymmetry is reduced."
    ),
}


# ---------------------------------------------------------------------------
# PDF Builder
# ---------------------------------------------------------------------------

def _safe(text: str) -> str:
    """Replace Unicode characters that Latin-1 (fpdf built-in fonts) cannot handle."""
    replacements = {
        "—": "--",   # em dash
        "–": "-",    # en dash
        "‘": "'",    # left single quote
        "’": "'",    # right single quote
        "“": '"',    # left double quote
        "”": '"',    # right double quote
        "…": "...",  # ellipsis
        "•": "-",    # bullet
        "→": "->",   # arrow
        "✓": "OK",   # checkmark
        "✗": "X",    # cross
        "↓": "v",    # down arrow
        "↑": "^",    # up arrow
        "é": "e",    # e acute
        "‹": "<",
        "›": ">",
    }
    for ch, repl in replacements.items():
        text = text.replace(ch, repl)
    return text


class Manual(FPDF):
    def header(self):
        pass

    def footer(self):
        self.set_y(-12)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(150, 150, 150)
        self.cell(0, 6, f"Functional Movement Conditioning Manual  --  Page {self.page_no()}", align="C")

    def _reset_x(self):
        self.set_x(self.l_margin)

    def chapter_title(self, text: str):
        self._reset_x()
        self.set_font("Helvetica", "B", 16)
        self.set_text_color(30, 90, 160)
        self.ln(6)
        self._reset_x()
        self.multi_cell(180, 10, _safe(text), align="L")
        self.set_draw_color(30, 90, 160)
        self.set_line_width(0.5)
        self.line(self.l_margin, self.get_y(), self.l_margin + 180, self.get_y())
        self.ln(3)
        self.set_text_color(0, 0, 0)

    def section_title(self, text: str):
        self._reset_x()
        self.set_font("Helvetica", "B", 12)
        self.set_text_color(60, 60, 60)
        self.ln(4)
        self._reset_x()
        self.multi_cell(180, 8, _safe(text), align="L")
        self.set_text_color(0, 0, 0)

    def subsection(self, label: str, body: str):
        self._reset_x()
        self.set_font("Helvetica", "B", 10)
        self.set_text_color(80, 80, 80)
        self.multi_cell(180, 6, _safe(label), align="L")
        self._reset_x()
        self.set_font("Helvetica", "", 10)
        self.set_text_color(0, 0, 0)
        self.multi_cell(180, 5.5, _safe(body.strip()), align="J")
        self.ln(2)

    def body(self, text: str, size: int = 10):
        self._reset_x()
        self.set_font("Helvetica", "", size)
        self.multi_cell(180, 5.5, _safe(text.strip()), align="J")
        self.ln(3)

    def bullet_list(self, items: list[str]):
        self.set_font("Helvetica", "", 10)
        for item in items:
            self.set_x(self.l_margin + 4)
            self.multi_cell(176, 5.5, f"- {_safe(item.strip())}", align="L")

    def error_block(self, errors: list[tuple[str, str]]):
        for label, correction in errors:
            self.set_font("Helvetica", "BI", 10)
            self.set_text_color(180, 50, 50)
            self.set_x(self.l_margin + 2)
            self.multi_cell(178, 5.5, f"X  {_safe(label)}", align="L")
            self._reset_x()
            self.set_font("Helvetica", "I", 10)
            self.set_text_color(50, 120, 50)
            self.set_x(self.l_margin + 10)
            self.multi_cell(170, 5.5, f"->  {_safe(correction.strip())}", align="L")
            self.set_text_color(0, 0, 0)
            self._reset_x()
            self.ln(1)


def build_pdf():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    pdf = Manual(orientation="P", unit="mm", format="A4")
    pdf.set_auto_page_break(auto=True, margin=18)
    pdf.set_margins(left=15, top=15, right=15)

    # ── Cover page ────────────────────────────────────────────────────────────
    pdf.add_page()
    pdf.ln(40)
    pdf.set_font("Helvetica", "B", 26)
    pdf.set_text_color(30, 90, 160)
    pdf.multi_cell(0, 12, TITLE, align="C")
    pdf.ln(6)
    pdf.set_font("Helvetica", "", 13)
    pdf.set_text_color(80, 80, 80)
    pdf.multi_cell(0, 7, SUBTITLE, align="C")
    pdf.ln(20)
    pdf.set_font("Helvetica", "I", 10)
    pdf.set_text_color(120, 120, 120)
    pdf.multi_cell(0, 6, _safe(
        "This guide combines functional movement principles drawn from US Army "
        "Physical Readiness Training doctrine (FM 7-22) with the specific form "
        "criteria and error patterns used in building and labelling this system's "
        "training dataset. It is written for everyday athletes and rehabilitation "
        "clients -- not drill instructors."
    ), align="C")

    # ── Introduction ─────────────────────────────────────────────────────────
    pdf.add_page()
    pdf.chapter_title("Introduction")
    pdf.body(INTRO)

    # ── Foundation Principles ─────────────────────────────────────────────────
    pdf.chapter_title("Foundation Principles")
    pdf.body(
        "Before diving into the individual exercises, it is worth spending a few "
        "minutes on the principles that underpin all of them. These are not abstract "
        "ideas — they are practical tools you can use on every single rep."
    )
    for principle, text in FOUNDATION_PRINCIPLES.items():
        pdf.subsection(principle, text)

    # ── Exercises ─────────────────────────────────────────────────────────────
    for i, ex in enumerate(EXERCISES):
        pdf.add_page()
        pdf.chapter_title(f"Exercise {i + 1}: {ex['name']}")

        pdf.section_title("Overview")
        pdf.body(ex["overview"])

        pdf.section_title("Setup")
        pdf.body(ex["setup"])

        pdf.section_title("Coaching Cues")
        pdf.bullet_list(ex["cues"])
        pdf.ln(2)

        pdf.section_title("Common Errors and Corrections")
        pdf.error_block(ex["errors"])

        pdf.section_title("Rep Guidance")
        pdf.body(ex["rep_guidance"])

    # ── Session Design ─────────────────────────────────────────────────────────
    pdf.add_page()
    pdf.chapter_title("Putting It Together: Session Design")
    pdf.body(SESSION_DESIGN)

    # ── Quick Reference ────────────────────────────────────────────────────────
    pdf.add_page()
    pdf.chapter_title("Common Corrections: Quick Reference")
    pdf.body(
        "Use this table when you are in the middle of a session and need a fast "
        "reminder without rereading the full exercise section."
    )
    for problem, fix in CORRECTIONS_QUICK_REFERENCE.items():
        pdf.subsection(problem, fix)

    # ── Save ──────────────────────────────────────────────────────────────────
    pdf.output(str(OUT_PATH))
    print(f"Written: {OUT_PATH}  ({OUT_PATH.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    build_pdf()
