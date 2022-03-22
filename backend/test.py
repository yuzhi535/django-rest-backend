from module import process, split_video, dynamic_time_warping, process_csv
import os
import paddlehub as hub

filename = 'test.mp4'
split_dir = 'test_split'

base_dir = '.'

path = os.path.join(base_dir, filename)
outputpath = os.path.join(base_dir, split_dir)


res = split_video(path, base_path='test_split')
# print(res)

model = hub.Module(name='openpose_body_estimation')
process(model, outputpath, output_path='test_output')

out = process_csv(base_path='test_output', output_path='test_user')
criterion = process_csv()
cost = dynamic_time_warping(criterion, out)
print(cost)