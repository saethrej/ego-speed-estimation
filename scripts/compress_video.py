"""

This script compresses and crops all 'filename' videos found in the data directory

"""

import os 

filename ="video.mp4"
outfilename = "video_comcro.mp4"

def main():
    this_dir = os.getcwd()
    for root, subFolders, files in os.walk(this_dir):
        if (not filename in files) or (outfilename in files):
            continue
        os.system("ffmpeg -i '" + os.path.join(root, filename) + "' -vf 'crop=iw:ih-400,scale=290:-2' -crf 5 '" + os.path.join(root, outfilename) + "'")
        # To delete original file, uncomment the following line
        # os.remove(os.path.join(root, filename))

main()