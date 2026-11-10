"""BlazePose landmark indices (33 points). Left/right are the climber's own."""

NOSE = 0
L_EAR, R_EAR = 7, 8
L_SHOULDER, R_SHOULDER = 11, 12
L_ELBOW, R_ELBOW = 13, 14
L_WRIST, R_WRIST = 15, 16
L_HIP, R_HIP = 23, 24
L_KNEE, R_KNEE = 25, 26
L_ANKLE, R_ANKLE = 27, 28
L_HEEL, R_HEEL = 29, 30
L_TOE, R_TOE = 31, 32

N_LANDMARKS = 33

# side -> (shoulder, elbow, wrist) / (hip, knee, ankle, toe)
ARM = {"left": (L_SHOULDER, L_ELBOW, L_WRIST), "right": (R_SHOULDER, R_ELBOW, R_WRIST)}
LEG = {"left": (L_HIP, L_KNEE, L_ANKLE, L_TOE), "right": (R_HIP, R_KNEE, R_ANKLE, R_TOE)}

# The 12 joints the pose solver moves; everything else rides along rigidly.
MAIN_JOINTS = [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]

LIMB_BONES = [(11, 13), (13, 15), (12, 14), (14, 16), (23, 25), (25, 27), (24, 26), (26, 28)]
TORSO_BONES = [(11, 12), (23, 24), (11, 23), (12, 24), (11, 24), (12, 23)]
BONES = LIMB_BONES + TORSO_BONES

# Lines to draw for a skeleton.
SKELETON = [
    (11, 13), (13, 15), (12, 14), (14, 16),
    (23, 25), (25, 27), (24, 26), (26, 28),
    (27, 31), (28, 32),
    (11, 12), (23, 24), (11, 23), (12, 24),
]

# Per limb index (see types.LIMBS): the joint that is pinned when the limb is on a hold.
LIMB_END_JOINT = [L_WRIST, R_WRIST, L_ANKLE, R_ANKLE]
LIMB_ROOT_JOINT = [L_SHOULDER, R_SHOULDER, L_HIP, R_HIP]
LIMB_SIDE = ["left", "right", "left", "right"]
