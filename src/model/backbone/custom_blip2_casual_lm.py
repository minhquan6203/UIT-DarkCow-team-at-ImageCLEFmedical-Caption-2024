
import torch
from torch import nn
from typing import Any, Optional, Tuple, Union
from transformers.models.t5.modeling_t5 import *
from torch.nn import CrossEntropyLoss
from transformers import AutoModel, AutoConfig,PreTrainedModel, AutoModelForCausalLM, Blip2QFormerModel,Blip2Config
from transformers.utils import ModelOutput
from transformers.models.blip_2.modeling_blip_2 import Blip2ForConditionalGenerationModelOutput
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

class Custom_BLIP2_Casual_LM(PreTrainedModel):
    def __init__(self,
                 vit_pretrained="google/vit-base-patch16-224-in21k",
                 qformer_pretrained="Salesforce/blip2-opt-2.7b",
                 lm_pretrained="VietAI/vit5-base",
                 freeze_qformer=False,
                 num_query_token=64,
                 freeze_lm=True,
                 use_lora=False,
                 cast_dtype=torch.float32,
                 lora_alpha=16,
                 lora_r=8,
                 lora_dropout=0.05,
                 lora_bias="none",
                 prefix_tokens=32,
                 lora_checkpoint=None
                 ):
        config = Blip2Config.from_pretrained(qformer_pretrained)
        config.num_query_tokens=num_query_token
        vision_config = AutoConfig.from_pretrained(vit_pretrained)
        lm_config = AutoConfig.from_pretrained(lm_pretrained)
        super().__init__(config)
        
        self.language_model = AutoModelForCausalLM.from_pretrained(lm_pretrained,config=lm_config)
        lm_hidden_size = lm_config.hidden_size
        
        config.text_config=lm_config
        config.vision_config=vision_config

        self.vision_projection=nn.Linear(vision_config.hidden_size, config.qformer_config.encoder_hidden_size)
        self.qformer = Blip2QFormerModel.from_pretrained(qformer_pretrained,config=config.qformer_config)
        self.query_tokens = nn.Parameter(torch.zeros(1, config.num_query_tokens, config.qformer_config.hidden_size))
        self.language_projection = nn.Linear(config.qformer_config.hidden_size, lm_hidden_size)
        self.llm_cast_dtype = cast_dtype
        print("LM loaded")

        if freeze_lm:
            print("Freeze LLM")
            for param in self.language_model.parameters():
                param.requires_grad = False

        if freeze_qformer:
            print("Freeze QFormer")
            self.query_tokens.requires_grad = False
            for param in self.qformer.parameters():
                param.requires_grad = False

        if use_lora==True and freeze_lm==False:
            print("Using LoRA")
            self.language_model = prepare_model_for_int8_training(self.language_model, use_gradient_checkpointing=True)
            if lora_checkpoint:
                print("Loading LoRA adapter ", lora_checkpoint)
                self.language_model = PeftModel.from_pretrained(self.language_model, lora_checkpoint)
            else:
                target_modules = ["q", "v"]
                task = "CAUSAL_LM" 
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
                    if "poly" in lm_pretrained.lower():
                        target_modules = ["c_attn", "c_proj", "c_fc"]
                    elif "bloom" in lm_pretrained.lower():
                        target_modules = ["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"]
                    elif "mistral" in lm_pretrained.lower():
                        target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj","lm_head"]
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
            input_ids: Optional[torch.FloatTensor] = None,
            decoder_input_ids: Optional[torch.LongTensor] = None,
            decoder_attention_mask: Optional[torch.LongTensor] = None,
            attention_mask: Optional[torch.LongTensor] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            labels: Optional[torch.LongTensor] = None,
            return_dict: Optional[bool] = None,
            **kwargs
    ) -> Union[Tuple, Blip2ForConditionalGenerationModelOutput]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict


        image_embeds = self.vision_projection(visual_features)
        num_images = orig_batch_size = visual_features.shape[0]
        image_attention_mask = torch.ones(image_embeds.size()[:-1], dtype=torch.long, device=image_embeds.device)

        query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
        query_outputs = self.qformer(
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        query_output = query_outputs[0]
        if num_images > 1:
            query_output = query_output.view(orig_batch_size, -1, query_output.shape[2])
        language_model_inputs = self.language_projection(query_output)
        language_attention_mask = torch.ones(
            language_model_inputs.size()[:-1], dtype=torch.long, device=language_model_inputs.device
        )

        with torch.cuda.amp.autocast(dtype=self.llm_cast_dtype):
            if input_ids is not None:
                lm_embedding = self.language_model.get_input_embeddings()
                inputs_embeds = lm_embedding(input_ids)
                inputs_embeds = torch.cat([language_model_inputs, inputs_embeds.to(language_model_inputs.device)], dim=1)
                
                if attention_mask is None:
                    attention_mask = torch.ones_like(input_ids)
                attention_mask = torch.cat([language_attention_mask, attention_mask], dim=1)
            else:
                lm_embedding = self.language_model.get_input_embeddings()
                input_ids = (
                    torch.LongTensor([[self.config.text_config.bos_token_id]])
                    .repeat(orig_batch_size, 1)
                    .to(image_embeds.device)
                )
                inputs_embeds = lm_embedding(input_ids)
                inputs_embeds = torch.cat([language_model_inputs, inputs_embeds], dim=1)

                if attention_mask is None:
                    attention_mask = torch.ones_like(input_ids)
                attention_mask = torch.cat([language_attention_mask, attention_mask],dim=1)

            outputs = self.language_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            logits = outputs.logits if return_dict else outputs[0]
            loss = None
            # we compute the loss here since we need to take into account the sequence length of the query embeds
            if labels is not None:
                labels = labels.to(logits.device)
                logits = logits[:, -labels.size(1):, :]
                # Shift so that tokens < n predict n
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous().to(logits.device)
                # Flatten the tokens
                loss_fct = CrossEntropyLoss(reduction="mean")

                loss = loss_fct(shift_logits.view(-1, self.config.text_config.vocab_size), shift_labels.view(-1))

        if not return_dict:
            output = (logits, image_embeds, query_outputs, outputs)
            return ((loss,) + output) if loss is not None else output

        return Blip2ForConditionalGenerationModelOutput(
            loss=loss,
            logits=logits,
            vision_outputs=image_embeds,
            qformer_outputs=query_outputs,
            language_model_outputs=outputs,
        )
    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, use_cache=None, **kwargs
    ):
        # cut decoder_input_ids if past_key_values is used
        if past_key_values is not None:
            input_ids = input_ids[:, -1:]

        # if model is used as a decoder in encoder-decoder model, the decoder attention mask is created on the fly
        input_shape = input_ids.shape
        if attention_mask is None:
            attention_mask = input_ids.new_ones(input_shape)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "visual_features": kwargs.get("visual_features", None),
            "past_key_values": past_key_values,
            "use_cache": use_cache,
        }

    def _reorder_cache(self, past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past),
            )
        return reordered_past

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