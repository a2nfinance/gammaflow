##############################################################################################
## The source code is referenced from MoCoGAN                                               ##
## (see https://github.com/CarloP95/mocogan/tree/a71449c0b617265b8c5193449b8121267941bf4c). ##
##############################################################################################
'''
resize video data to 96x96
require ffmpeg
need to set [Walk, Run, ...] directories under raw_data/
    downloaded from http://www.wisdom.weizmann.ac.il/%7Evision/SpaceTimeActions.html
'''
import os
import glob
import ffmpeg
import subprocess
import shutil

current_path = os.path.dirname(__file__)
resized_path = os.path.join(current_path, 'resized_data')
dirs = glob.glob(os.path.join(current_path, 'raw_data/*/*'))
#files = [ glob.glob(dir + '/*') for dir in dirs ]
files = [ glob.glob(dir) for dir in dirs ]
files = sum(files, []) # flatten

''' script for cropping '''
#for file in files:

#    ## Additions to resize UCF-101 and maintain directory structure
#    dir = file.split(os.sep)
#    currentAction = dir[2]
#    currentName = dir[3]
#    saveDir = os.path.join(resized_path, currentAction)

#    if not os.path.exists(saveDir):
#        os.makedirs(saveDir)
    #### End of additions.

#    os.system("ffmpeg -i %s -pix_fmt yuv420p -vf crop=96:96:42:24 %s.mp4" %
#             (file, os.path.join(saveDir, currentName)[:-4]))

''' script for reducing size '''
# # resize to 96x96
for file in files:

    ## Additions to resize UCF-101 and maintain directory structure
    dir = file.split(os.sep)
    currentAction = dir[-2]
    currentName = dir[-1]
    saveDir = os.path.join(resized_path, currentAction)

    if not os.path.exists(saveDir):
        os.makedirs(saveDir)
    #### End of additions.

    #os.system("ffmpeg -i %s -pix_fmt yuv420p -vf scale=96:96 %s.mp4" %
              #(file, os.path.join(saveDir, currentName)[:-4]))
              
    output_path = os.path.join(saveDir, currentName)[:-4] + ".mp4"
    # Construct the ffmpeg command
    command = [
        "ffmpeg",
        "-i", file,
        "-pix_fmt", "yuv420p",
        "-vf", "scale=96:96",
        output_path
    ]

    # Check if ffmpeg is available
    ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path is None:
        print("FFmpeg is not installed or not found in the system PATH.")
    else:
        # Execute the command
        try:
            subprocess.run(command, check=True)
            print("FFmpeg command executed successfully.")
        except subprocess.CalledProcessError as e:
            print(f"An error occurred while executing the FFmpeg command: {e}")
    

