# Copyright 2024 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
""" Testing suite for the Mindspore Prophetnet model. """

import numpy as np

from mindnlp.transformers import ProphetNetConfig
from mindnlp.utils import is_mindspore_available

from ...test_modeling_common import floats_tensor, ids_tensor


if is_mindspore_available():
    import mindspore as ms
    from mindspore import ops

    from mindnlp.transformers import ProphetNetDecoder


class ProphetNetStandaloneDecoderModelTester:
    def __init__(
        self,
        parent,
        vocab_size=99,
        batch_size=13,
        hidden_size=16,
        encoder_seq_length=7,
        decoder_seq_length=7,
        # For common tests
        is_training=True,
        is_decoder=True,
        use_attention_mask=True,
        add_cross_attention=False,
        use_cache=False,
        use_labels=True,
        decoder_start_token_id=0,
        encoder_ffn_dim=32,
        num_encoder_layers=2,
        num_encoder_attention_heads=4,
        decoder_ffn_dim=32,
        num_decoder_layers=2,
        num_decoder_attention_heads=4,
        max_position_embeddings=30,
        is_encoder_decoder=False,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        ngram=2,
        num_buckets=32,
        relative_max_distance=128,
        disable_ngram_loss=False,
        scope=None,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.encoder_seq_length = encoder_seq_length
        self.decoder_seq_length = decoder_seq_length
        # For common tests
        self.seq_length = self.decoder_seq_length
        self.is_training = is_training
        self.use_attention_mask = use_attention_mask
        self.use_labels = use_labels

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_decoder_layers
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.decoder_ffn_dim = decoder_ffn_dim
        self.encoder_ffn_dim = encoder_ffn_dim
        self.num_attention_heads = num_decoder_attention_heads
        self.num_encoder_attention_heads = num_encoder_attention_heads
        self.num_decoder_attention_heads = num_decoder_attention_heads
        self.eos_token_id = eos_token_id
        self.bos_token_id = bos_token_id
        self.pad_token_id = pad_token_id
        self.decoder_start_token_id = decoder_start_token_id
        self.ngram = ngram
        self.num_buckets = num_buckets
        self.relative_max_distance = relative_max_distance
        self.use_cache = use_cache
        self.disable_ngram_loss = disable_ngram_loss
        self.max_position_embeddings = max_position_embeddings
        self.add_cross_attention = add_cross_attention
        self.is_encoder_decoder = is_encoder_decoder

        self.scope = None
        self.decoder_key_length = decoder_seq_length
        self.base_model_out_len = 2
        self.num_hidden_states_types = 2  # decoder_main, decoder_ngram
        self.decoder_attention_idx = 1

    def prepare_config_and_inputs(self):
        input_ids = ids_tensor(
            [self.batch_size, self.encoder_seq_length], self.vocab_size)

        attention_mask = None
        if self.use_attention_mask:
            attention_mask = ids_tensor(
                [self.batch_size, self.encoder_seq_length], vocab_size=2)

        lm_labels = None
        if self.use_labels:
            lm_labels = ids_tensor(
                [self.batch_size, self.encoder_seq_length], self.vocab_size)

        config = ProphetNetConfig(
            vocab_size=self.vocab_size,
            hidden_size=self.hidden_size,
            num_encoder_layers=self.num_encoder_layers,
            num_decoder_layers=self.num_decoder_layers,
            decoder_ffn_dim=self.decoder_ffn_dim,
            encoder_ffn_dim=self.encoder_ffn_dim,
            num_encoder_attention_heads=self.num_encoder_attention_heads,
            num_decoder_attention_heads=self.num_decoder_attention_heads,
            eos_token_id=self.eos_token_id,
            bos_token_id=self.bos_token_id,
            use_cache=self.use_cache,
            pad_token_id=self.pad_token_id,
            decoder_start_token_id=self.decoder_start_token_id,
            ngram=self.ngram,
            num_buckets=self.num_buckets,
            relative_max_distance=self.relative_max_distance,
            disable_ngram_loss=self.disable_ngram_loss,
            max_position_embeddings=self.max_position_embeddings,
            add_cross_attention=self.add_cross_attention,
            is_encoder_decoder=self.is_encoder_decoder,
        )

        return (
            config,
            input_ids,
            attention_mask,
            lm_labels,
        )

    def prepare_config_and_inputs_for_decoder(self):
        (
            config,
            input_ids,
            attention_mask,
            lm_labels,
        ) = self.prepare_config_and_inputs()

        encoder_hidden_states = floats_tensor(
            [self.batch_size, self.encoder_seq_length, self.hidden_size])
        encoder_attention_mask = ids_tensor(
            [self.batch_size, self.encoder_seq_length], vocab_size=2)

        return (
            config,
            input_ids,
            attention_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            lm_labels,
        )

    def create_and_check_decoder_model_past(
        self,
        config,
        input_ids,
        attention_mask,
        lm_labels,
    ):
        config.use_cache = True
        model = ProphetNetDecoder(config=config)
        model.set_train(False)
        # first forward pass
        outputs = model(input_ids, use_cache=True)
        outputs_use_cache_conf = model(input_ids)
        outputs_no_past = model(input_ids, use_cache=False)

        self.parent.assertTrue(len(outputs) == len(outputs_use_cache_conf))
        self.parent.assertTrue(len(outputs) == len(outputs_no_past) + 1)

        past_key_values = outputs["past_key_values"]

        # create hypothetical next token and extent to next_input_ids
        next_tokens = ids_tensor((self.batch_size, 1), config.vocab_size)

        # append to next input_ids and
        next_input_ids = ops.cat([input_ids, next_tokens], axis=-1)

        output_from_no_past = model(next_input_ids)["last_hidden_state"]
        output_from_past = model(next_tokens, past_key_values=past_key_values)[
            "last_hidden_state"]

        # select random slice
        random_slice_idx = ids_tensor((1,), output_from_past.shape[-1]).item()
        output_from_no_past_slice = output_from_no_past[:,
                                                        next_input_ids.shape[-1] - 1, random_slice_idx].detach()
        output_from_past_slice = output_from_past[:,
                                                  0, random_slice_idx].detach()

        # test that outputs are equal for slice
        assert np.allclose(output_from_past_slice.asnumpy(),
                           output_from_no_past_slice.asnumpy(), atol=1e-3)

    def create_and_check_decoder_model_attention_mask_past(
        self,
        config,
        input_ids,
        attention_mask,
        lm_labels,
    ):
        model = ProphetNetDecoder(config=config)
        model.set_train(False)

        # create attention mask
        attn_mask = ops.ones(input_ids.shape, dtype=ms.int64)

        half_seq_length = input_ids.shape[-1] // 2
        attn_mask[:, half_seq_length:] = 0

        # first forward pass
        past_key_values = model(input_ids, attention_mask=attn_mask, use_cache=True)[
            "past_key_values"]

        # create hypothetical next token and extent to next_input_ids
        next_tokens = ids_tensor((self.batch_size, 1), config.vocab_size)

        # change a random masked slice from input_ids
        random_seq_idx_to_change = ids_tensor((1,), half_seq_length).item() + 1
        random_other_next_tokens = ids_tensor(
            (self.batch_size, 1), config.vocab_size).squeeze(-1)
        input_ids[:, -random_seq_idx_to_change] = random_other_next_tokens

        # append to next input_ids and attn_mask
        next_input_ids = ops.cat([input_ids, next_tokens], axis=-1)
        attn_mask = ops.cat(
            [attn_mask, ops.ones((attn_mask.shape[0], 1), dtype=ms.int64)],
            axis=1,
        )

        # get two different outputs
        output_from_no_past = model(next_input_ids)["last_hidden_state"]
        output_from_past = model(next_tokens, past_key_values=past_key_values)[
            "last_hidden_state"]

        # select random slice
        random_slice_idx = ids_tensor((1,), output_from_past.shape[-1]).item()
        output_from_no_past_slice = output_from_no_past[:,
                                                        next_input_ids.shape[-1] - 1, random_slice_idx].detach()
        output_from_past_slice = output_from_past[:,
                                                  0, random_slice_idx].detach()

        # test that outputs are equal for slice
        assert np.allclose(output_from_past_slice.asnumpy(),
                           output_from_no_past_slice.asnumpy(), atol=1e-2)

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        (
            config,
            input_ids,
            attention_mask,
            lm_labels,
        ) = config_and_inputs

        inputs_dict = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        return config, inputs_dict
