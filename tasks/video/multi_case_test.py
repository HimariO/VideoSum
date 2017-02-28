import os

step = 28600
videos = [
    'Ce7equ9zCxk_4_19',
    'IhwPQL9dFYc_130_136',
    'ACOmKiJDkA4_49_54',
]

for v in videos:
    command = "python3 ./single_case_test.py --checkpoint 'step-%d' --video './dataset/YouTubeClips/%s.avi'" % (step, v)
    os.system(command)
