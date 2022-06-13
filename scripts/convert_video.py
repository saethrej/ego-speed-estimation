"""

This script converts all 'filename' videos found in the data directory from .hvec to .mp4

"""

import os 

filename ="video.hevc"
outfilename = "video.mp4"

def main():
    this_dir = os.getcwd()
    for root, subFolders, files in os.walk(this_dir):
        if (not filename in files) or (outfilename in files):
            continue
        os.system("ffmpeg -i '" + os.path.join(root, filename) + "' '" + os.path.join(root, outfilename) + "'")
        # To delete original file, uncomment the following line
        # os.remove(os.path.join(root, filename))

main()