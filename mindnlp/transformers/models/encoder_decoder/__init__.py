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
"""
EncoderDecoder Model init
"""

from .import configuration_encoder_decoder, modeling_encoder_decoder
from .configuration_encoder_decoder import *
from .modeling_encoder_decoder import *

__all__ = []
__all__.extend(configuration_encoder_decoder.__all__)
__all__.extend(modeling_encoder_decoder.__all__)
