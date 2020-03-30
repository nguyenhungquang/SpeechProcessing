import argparse
import re
import nltk
import subprocess
import os
import tempfile
import queue
import sys
from pynput import keyboard
import sounddevice as sd
import soundfile as sf
import numpy  # Make sure NumPy is loaded before it is used in the callback
assert numpy  # avoid "imported but unused" message (W0611)
#nltk.download('all')
start=False
stop=False
directory=""
def int_or_str(text):
    """Helper function for argument parsing."""
    try:
        return int(text)
    except ValueError:
        return text


parser = argparse.ArgumentParser(add_help=False)
parser.add_argument(
    '-l', '--list-devices', action='store_true',
    help='show list of audio devices and exit')
args, remaining = parser.parse_known_args()
if args.list_devices:
    print(sd.query_devices())
    parser.exit(0)
parser = argparse.ArgumentParser(
    description=__doc__,
    formatter_class=argparse.RawDescriptionHelpFormatter,
    parents=[parser])
parser.add_argument(
    'dir',nargs='?',metavar='1',
    help='directory name')
parser.add_argument(
    'filename', nargs='?', metavar='FILENAME',
    help='audio file to store recording to')
parser.add_argument(
    '-d', '--device', type=int_or_str,
    help='input device (numeric ID or substring)')
parser.add_argument(
    '-r', '--samplerate', type=int, help='sampling rate')
parser.add_argument(
    '-c', '--channels', type=int, default=1, help='number of input channels')
parser.add_argument(
    '-t', '--subtype', type=str, help='sound file subtype (e.g. "PCM_24")')
args = parser.parse_args(remaining)

q = queue.Queue()


def callback(indata, frames, time, status):
    """This is called (from a separate thread) for each audio block."""
    if status:
        print(status, file=sys.stderr)
    q.put(indata.copy())
"""
def on_press(key):
    if key == keyboard.Key.space:
        print(1)
        start=True
def on_release(key):
    if key == keyboard.Key.tab:
        print(2)
        stop=True
    if key == keyboard.Key.esc:
        # Stop listener
        return False
"""
def start_recording(index):
    try:
        if args.samplerate is None:
            device_info = sd.query_devices(args.device, 'input')
            # soundfile expects an int, sounddevice provides a float:
            args.samplerate = int(device_info['default_samplerate'])
        if args.filename is None:
            args.filename = tempfile.mktemp(prefix=str(index)+'sentence',
                                            suffix='.wav', dir='')
        new_file= tempfile.mktemp(prefix=str(index)+'sentence',suffix='.wav', dir=args.dir)
        # Make sure the file is opened before recording anything:
        with sf.SoundFile(new_file, mode='x', samplerate=args.samplerate,
                          channels=args.channels, subtype=args.subtype) as file:
            with sd.InputStream(samplerate=args.samplerate, device=args.device,
                                channels=args.channels, callback=callback):
                print('#' * 80)
                print('press Ctrl+C to stop the recording')
                print('#' * 80)
                while True:
                    file.write(q.get())
    except KeyboardInterrupt:
        print('\nRecording finished: ' + repr(new_file))
        #break
        return new_file
        #parser.exit(0)
    except Exception as e:
        parser.exit(type(e).__name__ + ': ' + str(e))
def record_multi_files(array):
    index=0
    while True:
        is_rec=input("Start recording? y/n? ")
        if is_rec=='y':
            while len(re.split(r'[a-zA-Z0-9]',array[index]))==0:
                array.pop(index)
            print(array[index])
            wav_name=start_recording(index)
            with open("./"+args.dir+"/index.txt",'a',encoding="utf-8") as f:
                f.write(wav_name[wav_name.find("\\")+1:].split()[0]+"\n")
                f.write(array[index]+"\n")
        if is_rec=='n':
            return
        index+=1
if __name__=='__main__':
    txt_file="text.txt"
    subprocess.Popen("notepad.exe "+txt_file)
    if args.dir is None:
        args.dir="1"
    try:
        os.mkdir(args.dir)
    except OSError:
        print("Directory already exist")
        sys.exit()
    with open(txt_file,encoding="utf-8") as f:
        file_content=f.read()
    #sentence_array=re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s',file_content)
    sentence_array=nltk.tokenize.sent_tokenize(file_content)
    #print(sentence_array)
    #with keyboard.Listener(
    #        on_press=on_press,
    #        on_release=on_release) as listener:
    #    listener.join()
    record_multi_files(sentence_array)

