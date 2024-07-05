# Copyright (2024) Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from dataset.utils import get_visual_type, sample_frame_indices
from .processor import Processor
from tools.rw_utils import read_jsonlines

class MMDataset(object):
    def __init__(self, ann_path="", anns=None, processor:Processor=None):
        self.processor = processor
        if anns is None:
            self.anns = []
            if isinstance(ann_path, str):
                ann_path = [ann_path]
            for path in ann_path:
                self.anns.extend(read_jsonlines(path))
        else:
            self.anns = anns

    def __len__(self):
        return len(self.anns)
    
    def __getitem__(self, index):
        try:
            ann = self.anns[index]

            prompt = ann['text']['prompt']

            video_file = ann['video_file']
            visual_files = []
            start_time = ann.get("start_time", 0)
            end_time = ann.get("end_time", -1)
            if isinstance(video_file, list):
                # This is for MVBench/Episodic Reasoning
                # The video_file are a list of sorted frames extract from the target video
                for img_file in video_file:
                    if get_visual_type(img_file) == 'image':
                        visual_files.append(img_file)
                frame_indices = sample_frame_indices(start_frame=0, total_frames=len(visual_files), n_frames=min(len(visual_files), self.processor.max_n_frames))
                visual_files = [v for i,v in enumerate(visual_files) if i in frame_indices]    
            else:
                if get_visual_type(video_file) in ['image', 'video', 'gif']:
                    visual_files.append(video_file)
            assert len(visual_files) >= 0, f"Failed to load valid visual file from anns[{index}]!"
            images = []
            for v_f in visual_files:
                images.extend(self.processor.load_images(v_f, start_time=start_time, end_time=end_time))
            model_inputs = self.processor(prompt, images=images, edit_prompt=True, return_prompt=True)
        except Exception as e:
            print(f"Load data error: {e}")
            return ann, None
        return ann, model_inputs
