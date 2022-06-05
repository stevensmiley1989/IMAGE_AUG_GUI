import os
TRAIN_SPLIT=70 #70/30 train/val split only need to say 70
PYTHON_PATH='python3'
path_prefix_elements=r'/media/steven/Elements/'
path_prefix_mount_mac=r'/media/steven/Elements/'
path_prefix_volumes_one=r'/Volumes/One Touch/'
if os.path.exists(path_prefix_elements):
    path_prefix=path_prefix_elements
elif os.path.exists(path_prefix_mount_mac):
    path_prefix=path_prefix_mount_mac
else:
    path_prefix=path_prefix_volumes_one
print('path_prefix',path_prefix)
path_JPEGImages=r'{}Drone_Videos/Combined_transporter9_only_upto_5_18_2022/JPEGImages'.format(path_prefix)
path_Annotations=r'{}Drone_Videos/Combined_transporter9_only_upto_5_18_2022/Annotations'.format(path_prefix)
PREFIX='augmented'
root_background_img=r'misc/gradient_green.jpg' #path to background image for app
DEFAULT_ENCODING = 'utf-8'
XML_EXT = '.xml'
JPG_EXT = '.jpg'
COLOR='red'
root_bg='#000000'#'black'
root_fg='#b7f731'#'lime'
canvas_columnspan=50
canvas_rowspan=50
var_sometimes_INIT=1
var_sometimes_frac_INIT='0.5'
var_Fliplr_INIT=1
var_Fliplr_frac_INIT='0.5'
var_sometimes_Fliplr_frac_INIT='0.5'
var_Flipud_INIT=1
var_Flipud_frac_INIT='0.5'
var_sometimes_Flipud_frac_INIT='0.5'
var_Crop_INIT=0
var_Crop_frac1_INIT='0'
var_Crop_frac2_INIT='0.5'
var_sometimes_Crop_frac_INIT='0.5'
var_Affine_INIT=1
var_Affine_frac1_INIT='0.5'
var_Affine_frac2_INIT='1.5'
var_sometimes_Affine_frac_INIT='0.5'
var_GrayScale_INIT=1
var_GrayScale_frac1_INIT='1.0'
var_sometimes_GrayScale_frac_INIT='0.5'
var_ColorTemp_INIT=1
var_ColorTemp_frac1_INIT='1100'
var_ColorTemp_frac2_INIT='10000'
var_sometimes_ColorTemp_frac_INIT='0.5'
var_GaussianBlur_INIT=1
var_GaussianBlur_frac1_INIT='3.0'
var_sometimes_GaussianBlur_frac_INIT='0.5'
MAX_AUGS=500