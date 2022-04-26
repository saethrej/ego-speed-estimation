import os 

# filename = "runner_preprocessing.py" 
filename ="video.hevc"
outfilename = "video.mp4"

def main():
    this_dir = os.getcwd()
    num_videos = 0
    num_converted = 0
    for root, subFolders, files in os.walk(this_dir):
        if filename in files:
            num_videos += 1
        if outfilename in files:
            num_converted += 1

        if (not filename in files) or (outfilename in files):
            continue
        #print("This file: " + os.path.join(root, filename))
        #print("Output file: " + os.path.join(root, outfilename))
        os.system("ffmpeg -i '" + os.path.join(root, filename) + "' '" + os.path.join(root, outfilename) + "'")
        #os.remove(os.path.join(root, filename))
    
    print("Input Videos:", num_videos)
    print("Converted Videos:", num_converted)

main()