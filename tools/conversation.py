# Copyright (2024) Bytedance Ltd. and/or its affiliates

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# copy and modify from: https://github.com/OpenGVLab/Ask-Anything/blob/main/video_chat2/conversation.py
from PIL import Image
import torch
from transformers import StoppingCriteria, StoppingCriteriaList

from enum import auto, Enum
import os
from dataset.processor import Processor
import re


IMAGE_TOKEN = "<image>"
VIDEO_TOKEN = "<video>"

class SeparatorStyle(Enum):
    """Different separator style."""
    SINGLE = auto()
    TWO = auto()

def get_prompt(conv):
    ret = ""
    if conv.system:
        ret = conv.system + conv.sep1
    for i, (role, message) in enumerate(conv.messages):
        if message:
            # In current version, the image should be add at the first conversation round.
            # So we need to remove the special image tokens in following user input.
            if i > 0:
                message = re.sub(f"({IMAGE_TOKEN}|{VIDEO_TOKEN})\n*", "", message)
            ret += role + ": " + message
            if i % 2:
                ret += conv.sep2
            else:
                ret += conv.sep1
        else:
            ret += role + ":"
    return ret


class StoppingCriteriaSub(StoppingCriteria):
    def __init__(self, stops=[], encounters=1):
        super().__init__()
        self.stops = stops

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        for stop in self.stops:
            if torch.all((stop == input_ids[0][-len(stop):])).item():
                return True
        return False


class Chat:
    def __init__(self, model, processor: Processor, device='cuda', debug=False):
        self.model = model
        self.processor = processor
        self.device = device
        self.debug = debug
        stop_words_ids = [torch.tensor([self.processor.tokenizer.eos_token_id]).to(device)]
        self.stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=stop_words_ids)])

    def ask(self,text,conv):
        conv.messages.append([conv.roles[0], text])
        return conv

    def prepare_model_inputs(self, conv, visual_data_file=None, images=None, n_frames=None):
        conv.messages.append([conv.roles[1], None])
        print(conv.messages)
        conv.messages[0][1] = re.sub(f"({IMAGE_TOKEN}|{VIDEO_TOKEN})\n*", "", conv.messages[0][1])
        
        if images is None or isinstance(images, list) and len(images) == 0:
            if isinstance(visual_data_file, str) and os.path.exists(visual_data_file):
                images = self.processor.load_images(visual_data_file, n_frames)
            elif isinstance(visual_data_file, Image.Image):
                images = [visual_data_file]
            elif visual_data_file is None or visual_data_file == "":
                images = None
            else:
                raise NotImplementedError
    
        # os.system("rm tmp_images/*")    
        # for i, img in enumerate(images):
        #     img.save(f"tmp_images/{i+1}.jpg")
        
        if isinstance(images, list) and len(images) > 0:
            conv.messages[0][1] = IMAGE_TOKEN*len(images) + '\n' + conv.messages[0][1]

        prompt = get_prompt(conv)
        if self.debug:
            print(f"visual_data_file: {visual_data_file}")
            print(f"Prompt: {prompt}", flush=True)

        inputs = self.processor(prompt, images=images, edit_prompt=False, return_prompt=False)
        inputs = {k:v.to(self.device) for k,v in inputs.items() if v is not None}
        return inputs, conv, images

    def answer(self, conv, visual_data_file=None, images=None, n_frames=None, max_new_tokens=256, num_beams=1, min_length=1, top_p=1.0,
               repetition_penalty=1.0, length_penalty=1, temperature=0):
        inputs, conv, images = self.prepare_model_inputs(conv, visual_data_file, images, n_frames)
        if self.model is not None:
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                stopping_criteria=self.stopping_criteria,
                num_beams=num_beams,
                do_sample=True if temperature > 0 else False,
                min_length=min_length,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                length_penalty=length_penalty,
                temperature=temperature,
            )
            output_text = self.processor.tokenizer.decode(outputs[0][inputs['input_ids'][0].shape[0]:], skip_special_tokens=True)
        else:
            output_text = "Fake respone as launched in debug mode!"
        conv.messages[-1][1] = output_text
        return output_text, conv, images

class EasyDict(dict):
    """
    Get attributes

    >>> d = EasyDict({'foo':3})
    >>> d['foo']
    3
    >>> d.foo
    3
    >>> d.bar
    Traceback (most recent call last):
    ...
    AttributeError: 'EasyDict' object has no attribute 'bar'

    Works recursively

    >>> d = EasyDict({'foo':3, 'bar':{'x':1, 'y':2}})
    >>> isinstance(d.bar, dict)
    True
    >>> d.bar.x
    1
    """

    def __init__(self, d=None, **kwargs):
        if d is None:
            d = {}
        if kwargs:
            d.update(**kwargs)
        for k, v in d.items():
            setattr(self, k, v)
        # Class attributes
        for k in self.__class__.__dict__.keys():
            if not (k.startswith("__") and k.endswith("__")) and not k in ("update", "pop"):
                setattr(self, k, getattr(self, k))

    def __setattr__(self, name, value):
        if isinstance(value, (list, tuple)):
            value = [self.__class__(x) if isinstance(x, dict) else x for x in value]
        elif isinstance(value, dict) and not isinstance(value, self.__class__):
            value = self.__class__(value)
        super(EasyDict, self).__setattr__(name, value)
        super(EasyDict, self).__setitem__(name, value)

    __setitem__ = __setattr__

    def update(self, e=None, **f):
        d = e or dict()
        d.update(f)
        for k in d:
            setattr(self, k, d[k])

    def pop(self, k, d=None):
        if hasattr(self, k):
            delattr(self, k)
        return super(EasyDict, self).pop(k, d)

conv_tarsier = EasyDict({
    "system": "",
    "roles": ("USER", "ASSISTANT"),
    "messages": [],
    "sep1": " ",
    "sep2": "</s>",
}
)

conv_tarsier_yi = EasyDict({
    "system": "",
    "roles": ("USER", "ASSISTANT"),
    "messages": [],
    "sep1": " ",
    "sep2": "<|endoftext|>",
}
)

conv_tarsier_qwen2 = EasyDict({
    "system": "",
    "roles": ("USER", "ASSISTANT"),
    "messages": [],
    "sep1": " ",
    "sep2": "<|endoftext|>",
}
)

conv_templates = {
    "tarsier-7b": conv_tarsier,
    "tarsier-13b": conv_tarsier,
    "tarsier-34b": conv_tarsier_yi,
    "tarsier2-7b": conv_tarsier_qwen2
}
