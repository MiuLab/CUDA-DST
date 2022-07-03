"""
 Copyright (c) 2020, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
from torch.nn import BCEWithLogitsLoss
from transformers import BertPreTrainedModel, BertModel

TARGET_LABEL = ["Hotel-Internet"]

class BertForMultiLabelValueClassification(BertPreTrainedModel):
    """BERT model for classification.
    This module is composed of the BERT model with a linear layer on top of
    the pooled output.
    """
    def __init__(self, config):
        super().__init__(config)
        self.boolean_slot_list = config.boolean_target_slot_list
        self.dontcare_slot_list = config.dontcare_target_slot_list
        self.bert = BertModel(config)
        self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)
        self.dropout_heads = nn.Dropout(config.heads_dropout_rate)
        for slot in self.boolean_slot_list:
            self.add_module("boolean_" + slot, nn.Linear(config.hidden_size, 4))
        for slot in self.dontcare_slot_list:
            self.add_module("dontcare_" + slot, nn.Linear(config.hidden_size, 3))
        self.init_weights()
        

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):
        _, pooled_output = self.bert(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        pooled_output = self.dropout(pooled_output)

        class_loss_fct = nn.CrossEntropyLoss()
        total_loss = 0
        logits = {}
        
        for slot in self.boolean_slot_list:
            class_logits = self.dropout_heads(getattr(self, 'boolean_' + slot)(pooled_output))
            
            if labels is not None:
                batch = labels[slot].shape[0]
                for i in range(batch):
                    if labels[slot][i] != 0:
                        total_loss += class_loss_fct(class_logits[i:i+1], labels[slot][i:i+1])
            
            _, pred = torch.max(class_logits, 1)
            logits[slot] = pred

        for slot in self.dontcare_slot_list:
            class_logits = self.dropout_heads(getattr(self, 'dontcare_' + slot)(pooled_output))

            if labels is not None:
                batch = labels[slot].shape[0]
                for i in range(batch):
                    if labels[slot][i] != 0:
                        total_loss += class_loss_fct(class_logits[i:i+1], labels[slot][i:i+1])
            
            _, pred = torch.max(class_logits, 1)
            logits[slot] = pred

        if labels is not None:
            return total_loss
        else:
            return logits

