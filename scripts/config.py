# ============================================
# 配置
# ============================================
NUM_BINS = 200  # Value 分成 200 个 bin
VALUE_MIN = -1.0  # Value 最小值
VALUE_MAX = 0.0   # Value 最大值


# 相机名称配置（支持鱼眼相机和深度相机两种类型）
CAMERA_CONFIGS = {
    "fisheye": {
        "left": "robot0_mid_fisheye_color",
        "right": "robot1_mid_fisheye_color",
        "description": "FisheyeCamera",
    },
    # "depth": {
    #     "left": "pikaDepthCamera_l",
    #     "right": "pikaDepthCamera_r",
    #     "description": "DepthCamera",
    # },
}