import torch


'''                                                      Up
         | Kubric | OpenCV | Pytorch3d                    |
X-axis   | Right  | Right  |   Left                       |________ Right
Y-axis   | Up     | Down   |   Up                        /
Z-axis   | Out    | In     |   In                    Out/
Kubric:    right-handed frame, default is the same as openGL / EG3D
OpenCV:    right-handed frame
pytorch3d: right-handed frame
COLMAP:    same as OpenCV convention
See: https://stackoverflow.com/questions/44375149/opencv-to-opengl-coordinate-system-transform

EG3D renderer requires extrinsics (camera to world transformation) defined in OpenCV coordinate convension
See https://github.com/NVlabs/eg3d/issues/65
'''
opengl_to_cv2 = torch.tensor([[1.0, 0.0, 0.0, 0.0],   # inverse y-axis and z-axis, no translation
                            [0.0, -1.0, 0.0, 0.0],
                            [0.0, 0.0, -1.0, 0.0],
                            [0.0, 0.0, 0.0, 1.0]])
cv2_to_torch3d = torch.tensor([[-1.0, 0.0, 0.0, 0.0], # inverse x-axis and y-axis, no translation
                                [0.0, -1.0, 0.0, 0.0],
                                [0.0, 0.0, 1.0, 0.0],
                                [0.0, 0.0, 0.0, 1.0]])
cv2_to_eg3d = torch.tensor([[1,0,0,0], 
                            [0,1,0,0], 
                            [0,0,1,0], 
                            [0,0,0,1]], dtype=torch.float32)
cv2_to_colmap = torch.tensor([[-1,0,0,0], 
                            [0,-1,0,0], 
                            [0,0,1,0], 
                            [0,0,0,1]], dtype=torch.float32)
torch3d_to_cv2 = torch.inverse(cv2_to_torch3d)
cv2_to_opengl = torch.inverse(opengl_to_cv2)
eg3d_to_cv2 = torch.inverse(cv2_to_eg3d)
colmap_to_cv2 = torch.inverse(cv2_to_colmap)

SCALE_TRIPLANE = 0.87    # raw scale of the triplane (object) of pretrained model [-0.87, 0.87]
SCALE_TRIPLANE_SAFE = 0.6   # tuned by hand, normalize the object in this range for security

# scene scale normalization params
SCALE_OMNIOBJ = 3.8
SCALE_SHAPENET = 3.0
SCALE_MVIMGNET = 2.5
SCALE_OBJAVERSE = 2.4 #2.8
SCALE_OBJAVERSE_BOTH = 3.0

# dataset sampling weights params
WEIGHT_OMNIOBJ = 0.05
WEIGHT_SHAPENET = 0.05
WEIGHT_MVIMGNET = 0.45
WEIGHT_OBJAVERSE = 0.45
WEIGHT_WILD = 1.0

# Shapenet categories
ShapeNet_ids = {
    'table': '04379243', 'car': '02958343', 'chair': '03001627', 'airplane': '02691156', 'sofa': '04256520',
    'rifle': '04090263', 'lamp': '03636649', 'watercraft': '04530566', 'bench': '02828884', 'loudspeaker': '03691459',
    'cabinet': '02933112', 'display': '03211117', 'telephone': '04401088', 'bus': '02924116', 'bathtub': '02808440',
    'guitar': '03467517', 'faucet': '03325088', 'clock': '03046257', 'flowerport': '03991062', 'jar': '03593526',
    'bottle': '02876657', 'bookshelf': '02871439', 'laptop': '03642806', 'knife': '03624134', 'train': '04468005',
    'trash bin': '02747177', 'motorbike': '03790512', 'pistol': '03948459', 'file cabinet': '03337140',
    'bed': '02818832', 'piano': '03928116', 'stove': '04330267', 'mug': '03797390', 'bowl': '02880940',
    'washer': '04554684', 'printer': '04004475', 'helmet': '03513137', 'microwaves': '03761084',
    'skateboard': '04225987', 'tower': '04460130', 'camera': '02942699', 'basket': '02801938', 'can': '02946921',
    'pillow': '03938244', 'mailbox': '03710193', 'dishwasher': '03207941', 'rocket': '04099429', 'bag': '02773838',
    'birdhouse': '02843684', 'earphone': '03261776', 'microphone': '03759954', 'remote': '04074963',
    'keyboard': '03085013', 'bicycle': '02834778', 'cap': '02954340'
}

ShapeNet_general_train = ['airplane', 'bench', 'cabinet', 'car', 'chair', 'display', 'lamp', 'loudspeaker', 'rifle',
    'sofa', 'table', 'telephone', 'watercraft']

ShapeNet_general_test_unseen = ['bus', 'guitar', 'clock', 'bottle', 'train', 'mug', 'washer', 'skateboard', 'dishwasher', 'pistol']

