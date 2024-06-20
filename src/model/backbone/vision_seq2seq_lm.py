import torch
from torch import nn
from typing import Any, Optional, Tuple, Union
from transformers.models.t5.modeling_t5 import *
from torch.nn import CrossEntropyLoss
from transformers import AutoModel, AutoConfig, AutoModelForSeq2SeqLM,PreTrainedModel, AutoModelForCausalLM
from transformers.utils import ModelOutput

from transformers.utils.model_parallel_utils import assert_device_map, get_device_map

from collections import defaultdict, OrderedDict
from typing import Optional, Union, Tuple
import torch.nn.functional as F
from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_int8_training,
    set_peft_model_state_dict, PrefixTuningConfig,
    PeftModel
)
from dataclasses import dataclass

@dataclass
class VisionLMForConditionalGenerationModelOutput(ModelOutput):
    """
    Class defining the outputs of [`Blip2ForConditionalGeneration`].

    Args:
        loss (`torch.FloatTensor`, *optional*, returned when `labels` is provided, `torch.FloatTensor` of shape `(1,)`):
            Language modeling loss from the language model.
        logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head of the language model.
        vision_outputs (`BaseModelOutputWithPooling`):
            Outputs of the vision encoder.
        query_output (`BaseModelOutputWithPoolingAndCrossAttentions`):
            Outputs of the Q-Former (Querying Transformer).
        language_model_outputs (`CausalLMOutputWithPast` or `Seq2SeqLMOutput`):
            Outputs of the language model.
    """

    loss: Optional[Tuple[torch.FloatTensor]] = None
    logits: Optional[Tuple[torch.FloatTensor]] = None
    vision_outputs: Optional[torch.FloatTensor] = None
    language_model_outputs: Optional[Tuple[torch.FloatTensor]] = None

    def to_tuple(self) -> Tuple[Any]:
        return tuple(
            self[k]
            if k not in ["vision_outputs","language_model_outputs"]
            else getattr(self, k).to_tuple()
            for k in self.keys()
        )

class Vision_Seq2Seq_LM(PreTrainedModel):
    def __init__(self,
                 vit_pretrained="google/vit-base-patch16-224-in21k",
                 lm_pretrained="luqh/ClinicalT5-base",
                 cast_dtype=torch.float32,
                 freeze_lm=True,
                 use_lora=False,
                 lora_alpha=16,
                 lora_r=8,
                 lora_dropout=0.05,
                 lora_bias="none",
                 prefix_tokens=32,
                 ):
        vision_config = AutoConfig.from_pretrained(vit_pretrained)
        lm_config = AutoConfig.from_pretrained(lm_pretrained)
        super().__init__(lm_config)
        try:
            self.language_model = AutoModelForSeq2SeqLM.from_pretrained(lm_pretrained,config=lm_config)
        except:
            self.language_model = AutoModelForSeq2SeqLM.from_pretrained(lm_pretrained,config=lm_config,from_flax=True)
        lm_hidden_size = lm_config.d_model
        lm_config.bos_token_id = lm_config.decoder_start_token_id

        self.vision_projection=nn.Linear(vision_config.hidden_size, lm_hidden_size)
        self.language_projection = nn.Linear(lm_hidden_size, lm_hidden_size)
        self.llm_cast_dtype = cast_dtype
        print("LM loaded")
        if freeze_lm:
            print("Freeze LM")
            for param in self.language_model.parameters():
                param.requires_grad = False

        if use_lora==True and freeze_lm==False:
            print("Using LoRA")
            target_modules = ["q", "v"]
            task = "SEQ_2_SEQ_LM"
            if isinstance(use_lora, bool) or use_lora=="lora":
                config = LoraConfig(
                    r=lora_r,
                    lora_alpha=lora_alpha,
                    target_modules=target_modules,
                    lora_dropout=lora_dropout,
                    bias=lora_bias,
                    task_type=task
                )
            elif use_lora == "lora_all":
                if "mt0" in lm_pretrained.lower() or "t5" in lm_pretrained.lower():
                    target_modules = ["q", ".k", "v", ".o", "wi_0", "wi_1", "wo"]
                config = LoraConfig(
                    r=lora_r,
                    lora_alpha=lora_alpha,
                    target_modules=target_modules,
                    lora_dropout=lora_dropout,
                    bias=lora_bias,
                    task_type=task
                )
            elif use_lora == "prefix":
                config = PrefixTuningConfig(
                    task_type=task,
                    num_virtual_tokens=prefix_tokens,
                )
            self.language_model = get_peft_model(self.language_model, config)

    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.language_model.set_input_embeddings(value)

    def set_output_embeddings(self, new_embeddings):
        self.language_model.set_output_embeddings(new_embeddings)

    def get_output_embeddings(self) -> nn.Module:
        return self.language_model.get_output_embeddings()

    def forward(
            self,
            visual_features: torch.FloatTensor,
            input_ids: Optional[torch.FloatTensor]=None,
            attention_mask: Optional[torch.LongTensor] = None,
            decoder_input_ids: Optional[torch.LongTensor] = None,
            decoder_attention_mask: Optional[torch.LongTensor] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            labels: Optional[torch.LongTensor] = None,
            return_dict: Optional[bool] = None,
            **kwargs
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        image_embeds = self.vision_projection(visual_features)
        orig_batch_size = visual_features.shape[0]
        image_attention_mask = torch.ones(image_embeds.size()[:-1], dtype=torch.long, device=image_embeds.device)

        with torch.cuda.amp.autocast(dtype=self.llm_cast_dtype):
            if input_ids is not None:
                lm_embedding = self.language_model.get_input_embeddings()
                inputs_embeds = lm_embedding(input_ids)
                inputs_embeds = torch.cat([image_embeds,inputs_embeds],dim=1)
                if attention_mask is None:
                    attention_mask = torch.ones_like(input_ids).to(image_embeds.device)
                attention_mask = torch.cat([image_attention_mask, attention_mask], dim=1)
            else:
                lm_embedding = self.language_model.get_input_embeddings()
                input_ids = (
                    torch.LongTensor([[self.config.bos_token_id]])
                    .repeat(orig_batch_size, 1)
                    .to(image_embeds.device)
                )
                inputs_embeds = lm_embedding(input_ids)
                inputs_embeds = torch.cat([image_embeds, inputs_embeds], dim=1)

                if attention_mask is None:
                    attention_mask = torch.ones_like(input_ids)
                attention_mask = torch.cat([image_attention_mask, attention_mask],dim=1)


            outputs = self.language_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                decoder_input_ids=decoder_input_ids,
                decoder_attention_mask=decoder_attention_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                labels=labels,
            )
            loss = outputs.loss if return_dict else outputs[0]
            logits = outputs.logits if return_dict else outputs[1]

        if not return_dict:
            output = (logits, image_embeds, outputs)
            return ((loss,) + output) if loss is not None else output

        return VisionLMForConditionalGenerationModelOutput(
            loss=loss,
            logits=logits,
            vision_outputs=image_embeds,
            language_model_outputs=outputs,
        )
    
    @torch.no_grad()
    def generate(
        self,
        visual_features: torch.FloatTensor,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        **generate_kwargs,
    ) -> torch.LongTensor:
        """
        Overrides `generate` function to be able to use the model as a conditional generator.
        Args:
            visual_features (`torch.FloatTensor` of shape (batch_size, num_channels, height, width)):
                Input images to be processed.
            input_ids (`torch.LongTensor` of shape (batch_size, sequence_length), *optional*):
                The sequence used as a prompt for the generation.
            attention_mask (`torch.LongTensor` of shape (batch_size, sequence_length), *optional*):
                Mask to avoid performing attention on padding token indices
        Returns:
            captions (list): A list of strings of length batch_size * num_captions.
        """
        if hasattr(self, "hf_device_map"):
            # preprocess for `accelerate`
            self._preprocess_accelerate()

        orig_batch_size = visual_features.shape[0]
        image_embeds = self.vision_projection(visual_features)
        image_attention_mask = torch.ones(image_embeds.size()[:-1], dtype=torch.long, device=image_embeds.device)

        with torch.cuda.amp.autocast(dtype=self.llm_cast_dtype):
            if input_ids is not None:
                lm_embedding = self.language_model.get_input_embeddings()
                inputs_embeds = lm_embedding(input_ids)
                inputs_embeds = torch.cat([image_embeds,inputs_embeds],dim=1)
                if attention_mask is None:
                    attention_mask = torch.ones_like(input_ids).to(image_embeds.device)
                attention_mask = torch.cat([image_attention_mask, attention_mask], dim=1)
            else:
                lm_embedding = self.language_model.get_input_embeddings()
                input_ids = (
                    torch.LongTensor([[self.config.bos_token_id]])
                    .repeat(orig_batch_size, 1)
                    .to(image_embeds.device)
                )
                inputs_embeds = lm_embedding(input_ids)
                inputs_embeds = torch.cat([image_embeds, inputs_embeds], dim=1)

                if attention_mask is None:
                    attention_mask = torch.ones_like(input_ids)
                attention_mask = torch.cat([image_attention_mask, attention_mask],dim=1)
                

            outputs = self.language_model.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                **generate_kwargs,
            )

        return outputs

    def _preprocess_accelerate(self):
        r"""
        Some pre-processing hacks to make the model `accelerate` compatible. Check
        https://github.com/huggingface/transformers/pull/21707 for more details.
        """
        hf_device_map = self.hf_device_map

        if len(hf_device_map) > 1 and "language_model" not in hf_device_map and torch.cuda.device_count() > 1:
            # warn users about unexpected behavior when using multi-GPU + BLIP-2 + `accelerate`.
            print(
                "The `language_model` is not in the `hf_device_map` dictionary and you are running your script"
                " in a multi-GPU environment. this may lead to unexpected behavior when using `accelerate`."
                " Please pass a `device_map` that contains `language_model` to remove this warning."
                " Please refer to https://github.com/huggingface/blog/blob/main/accelerate-large-models.md for",
                " more details on creating a `device_map` for large models.",
            )

        if hasattr(self.language_model, "_hf_hook"):
            self.language_model._hf_hook.io_same_device = True  # For `generate` compatibility