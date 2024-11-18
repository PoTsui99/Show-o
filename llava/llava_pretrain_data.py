import copy
import json
import os
from functools import partial

import torch
from PIL import Image
from llava.llava import conversation as conversation_lib
from torch.utils.data import Dataset
from torch.utils.data.distributed import DistributedSampler
from transformers import CLIPImageProcessor

DEFAULT_IMAGE_TOKEN = "<image>"
IGNORE_INDEX = -100
conversation_lib.default_conversation = conversation_lib.conv_templates["plain"]

def preprocess_multimodal(sources):
    for source in sources:
        for sentence in source:
            if DEFAULT_IMAGE_TOKEN in sentence['value']:
                sentence['value'] = sentence['value'].replace(DEFAULT_IMAGE_TOKEN, '').strip()
                sentence['value'] = DEFAULT_IMAGE_TOKEN + '\n' + sentence['value']
                sentence['value'] = sentence['value'].strip()

                # Customized operation, get rid of <image> special token. Edited by Zechen
                sentence["value"] = sentence["value"].replace(DEFAULT_IMAGE_TOKEN, "")
                sentence['value'] = sentence['value'].strip()

    return sources


def preprocess_plain(sources, tokenizer):
    # add end signal and concatenate together
    conversations = []
    for source in sources:
        assert len(source) == 2
        # assert DEFAULT_IMAGE_TOKEN in source[0]['value']
        # source[0]['value'] = DEFAULT_IMAGE_TOKEN
        source[0]['value'] = ""
        conversation = source[0]['value'] + source[1]['value'] + conversation_lib.default_conversation.sep
        conversations.append(conversation)

    # tokenize conversations
    # input_ids = [tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations]
    input_ids = [tokenizer(prompt)["input_ids"] + [tokenizer.eos_token_id] for prompt in conversations]
    targets = copy.deepcopy(input_ids)

    for target, source in zip(targets, sources):
        # tokenized_len = len(tokenizer_image_token(source[0]['value'], tokenizer))
        tokenized_len = len(tokenizer(source[0]['value'])["input_ids"])
        if tokenized_len > 0:
            target[:tokenized_len] = IGNORE_INDEX

    return dict(input_ids=torch.tensor(input_ids), labels=torch.tensor(targets))


# Define constants
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def expand2square(pil_img, background_color):
    """
    Expand image to square with specified background color.
    background_color should be a tuple of RGB values (0-255).
    """
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result

class LLaVAPretrainCaptioningDataset(Dataset):
    def __init__(self, tokenizer, pad2square=False):
        super(LLaVAPretrainCaptioningDataset, self).__init__()
        
        self.tokenizer = tokenizer
        self.pad2square = pad2square

        # Convert IMAGENET_MEAN from [0,1] to [0,255] for padding
        self.background_color = tuple(int(x * 255) for x in IMAGENET_MEAN)

        data_file_path = "/mnt/bn/vgfm2/test_dit/blip_laion_cc_sbu_558k.json"
        self.image_root = "/mnt/bn/vgfm2/test_dit/pretraining_data"

        with open(data_file_path, 'r') as f:
            data = json.load(f)
        self.list_data_dict = []
        for item in data:
            if 'image' in item.keys():
                self.list_data_dict.append(item)

        self.processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14-336")
        
        # Verify that processor's crop size is square
        assert self.processor.crop_size['height'] == self.processor.crop_size['width'], \
            "Processor crop size must be square"

        print("Formatting llava captioning data")

    def __len__(self):
        return len(self.list_data_dict)

    def __getitem__(self, i):
        sources = self.list_data_dict[i]
        if isinstance(i, int):
            sources = [sources]
        assert len(sources) == 1, "Don't know why it is wrapped to a list"  # FIXME

        assert 'image' in sources[0]
        image_file = self.list_data_dict[i]['image']
        image_folder = self.image_root
        image = Image.open(os.path.join(image_folder, image_file)).convert('RGB')
        
        # Resize image to target size
        target_size = self.processor.crop_size['height']  # We've asserted height == width
        
        if self.pad2square:
            # Expand to square with ImageNet mean values as background
            image = expand2square(image, self.background_color)
        
        # Resize to target size
        image = image.resize((target_size, target_size), Image.Resampling.BICUBIC)
        
        # Apply CLIP preprocessing
        image = self.processor.preprocess(image, return_tensors='pt')['pixel_values'][0]

        sources = preprocess_multimodal(copy.deepcopy([e["conversations"] for e in sources]))
        data_dict = preprocess_plain(sources, self.tokenizer)

        if isinstance(i, int):
            data_dict = dict(input_ids=data_dict["input_ids"][0],
                           labels=data_dict["labels"][0])

        # image exist in the data
        if 'image' in self.list_data_dict[i]:
            data_dict['image'] = image
        else:
            # image does not exist in the data, but the model is multimodal
            data_dict['image'] = torch.zeros(3, target_size, target_size)

        return data_dict
    

def collate_fn(
        instances,
        tokenizer=None,
        max_length=77,
):
    input_ids, labels = tuple([instance[key] for instance in instances]
                              for key in ("input_ids", "labels"))
    input_ids = torch.nn.utils.rnn.pad_sequence(
        input_ids,
        batch_first=True,
        padding_value=tokenizer.pad_token_id)
    labels = torch.nn.utils.rnn.pad_sequence(labels,
                                             batch_first=True,
                                             padding_value=IGNORE_INDEX)

    if input_ids.shape[-1] < max_length:
        offset = max_length - input_ids.shape[-1]
        pad_tube = torch.ones(size=(input_ids.shape[0], offset), dtype=input_ids.dtype) * tokenizer.pad_token_id
        input_ids = torch.cat([input_ids, pad_tube], dim=1)

        offset = max_length - labels.shape[-1]
        pad_tube = torch.ones(size=(labels.shape[0], offset), dtype=labels.dtype) * IGNORE_INDEX
        labels = torch.cat([labels, pad_tube], dim=1)

    min_max_len = min(max_length, tokenizer.model_max_length)

    input_ids = input_ids[:, :min_max_len]
    labels = labels[:, :min_max_len]
    batch = dict(
        input_ids=input_ids,
        labels=labels,
        attention_mask=input_ids.ne(tokenizer.pad_token_id),
    )

    if 'image' in instances[0]:
        images = [instance['image'] for instance in instances]
        if all(x is not None and x.shape == images[0].shape for x in images):
            batch['images'] = torch.stack(images)
        else:
            batch['images'] = images

    return batch


def get_plain_data_loader(
        tokenizer,
        batch_size,
        num_workers,
        world_size,
        local_rank,
        max_length,
):
    train_dataset = LLaVAPretrainCaptioningDataset(tokenizer)
    datasampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=local_rank)
    dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=partial(
            collate_fn,
            tokenizer=tokenizer,
            max_length=max_length,
        ),
        sampler=datasampler
    )

    return dataloader


if __name__ == '__main__':
    import transformers
    pretrained_model_path = '/mnt/bn/vgfm2/test_mlx/xavier/pretrained_weights/phi-1_5'
    tokenizer = transformers.AutoTokenizer.from_pretrained(pretrained_model_path,
                                                           padding_side="left")
    special_tokens = ("soi", "eoi", "sovi", "eovi", "t2i", "mmu", "t2v", "v2v", "lvg")
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    tokenizer.add_tokens(list(special_tokens))

    dataset = LLaVAPretrainCaptioningDataset(tokenizer)

    dataset.__getitem__(0)

