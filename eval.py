# Copyright 2021 Huawei Technologies Co., Ltd
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

import argparse
import os
import random
import numpy as np
import mindspore as ms
import mindspore.nn as nn
from mindspore import context
from mindspore.ops import operations as ops
from mindspore import Tensor
from mindspore import load_checkpoint, load_param_into_net
import src.dataset as dt
from src.config import relationnet_cfg as cfg
import scipy.stats
import scipy.special as sc


parser = argparse.ArgumentParser(description="One Shot Visual Recognition")
parser.add_argument("-f", "--feature_dim", type=int, default=64)
parser.add_argument("-r", "--relation_dim", type=int, default=8)
parser.add_argument("-w", "--class_num", type=int, default=5)                       #N-way
parser.add_argument("-s", "--sample_num_per_class", type=int, default=1)            #K-shot
parser.add_argument("-b", "--batch_num_per_class", type=int, default=19)            #batch_size = b*N(K)
parser.add_argument("-e", "--episode", type=int, default=10)
parser.add_argument("-t", "--test_episode", type=int, default=1000)
parser.add_argument("-l", "--learning_rate", type=float, default=0.001)
parser.add_argument("-g", "--gpu", type=int, default=0)
parser.add_argument("-u", "--hidden_unit", type=int, default=10)
parser.add_argument("--data_path", default='/home/jialing/FSL/data/omniglot_resized/',
                    help="Path where the dataset is saved")
parser.add_argument("-dt", "--device_target", type=str, default='Ascend', choices=("Ascend", "GPU", "CPU"),
                    help="Device target, support Ascend, GPU and CPU.")
parser.add_argument("-di", "--device_id", type=int, default=5, help='device id of GPU or Ascend. (Default: 0)')
parser.add_argument("--ckpts_dir", default='./output/')
args = parser.parse_args()

context.set_context(mode=context.GRAPH_MODE, device_target=args.device_target, device_id=args.device_id)


# init operators
concat0dim = ops.Concat(axis=0)


def mean_confidence_interval(data, confidence=0.95):
    a = 1.0*np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * sc.stdtrit(n-1, (1+confidence)/2.)
    return m, h


class Encoder_Relation(nn.Cell):
    """docstring for ClassName"""

    def __init__(self, input_size, hidden_size):
        super(Encoder_Relation, self).__init__()

        #init operations
        self.tile = ops.Tile()
        self.expand_dims = ops.ExpandDims()
        self.transpose = ops.Transpose()
        self.concat2dim = ops.Concat(axis=2)
        self.relu = ops.ReLU()
        self.sigmoid = ops.Sigmoid()
        self.reshape = ops.Reshape()
        self.stack = ops.Stack(0)
        self.feature_dim = cfg.feature_dim
        self.class_num = cfg.class_num

        #CNNEncoder-Network
        self.Encoderlayer1 = nn.SequentialCell(
            nn.Conv2d(1, 64, kernel_size=3, pad_mode='pad', padding=0, has_bias=True),              #pad_mode用pad还是valid
            nn.BatchNorm2d(64, momentum=1, affine=True),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2))
        self.Encoderlayer2 = nn.SequentialCell(
            nn.Conv2d(64, 64, kernel_size=3, pad_mode='pad', padding=0, has_bias=True),
            nn.BatchNorm2d(64, momentum=1, affine=True),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2))
        self.Encoderlayer3 = nn.SequentialCell(
            nn.Conv2d(64, 64, kernel_size=3, pad_mode='pad', padding=1, has_bias=True),
            nn.BatchNorm2d(64, momentum=1, affine=True),
            nn.ReLU())
        self.Encoderlayer4 = nn.SequentialCell(
            nn.Conv2d(64, 64, kernel_size=3, pad_mode='pad', padding=1, has_bias=True),
            nn.BatchNorm2d(64, momentum=1, affine=True),
            nn.ReLU())

        #Relation-Network
        self.Relationlayer1 = nn.SequentialCell(
            nn.Conv2d(128, 64, kernel_size=3, pad_mode='pad', padding=1, has_bias=True),
            nn.BatchNorm2d(64, momentum=1, affine=True),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2))
        self.Relationlayer2 = nn.SequentialCell(
            nn.Conv2d(64, 64, kernel_size=3, pad_mode='pad', padding=1, has_bias=True),
            nn.BatchNorm2d(64, momentum=1, affine=True),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2))
        self.fc1 = nn.Dense(input_size, hidden_size)  # input_size = feature_dim=64,hidden_size = relation_dim=8
        self.fc2 = nn.Dense(hidden_size, 1)

    def construct(self, x):  # modified forward->construct
        sample = x[:5, :, :, :]  # [5,64,5,5]
        batch = x[5:, :, :, :]  # [95,64,5,5]
        s_out = self.Encoderlayer1(sample)
        s_out = self.Encoderlayer2(s_out)
        s_out = self.Encoderlayer3(s_out)
        s_out = self.Encoderlayer4(s_out)
        b_out = self.Encoderlayer1(batch)
        b_out = self.Encoderlayer2(b_out)
        b_out = self.Encoderlayer3(b_out)
        b_out = self.Encoderlayer4(b_out)
        sample_features, batch_features = s_out, b_out        #[5,64,5,5]     [95,64,5,5]

        if batch_features.shape[0] == 95:
            sample_features_ext_list1 = []
            sample_features_ext_list2 = []
            for _ in range(45):
                sample_features_ext_list1.append(sample_features)
            sample_features_ext1 = self.stack(sample_features_ext_list1)  # [1,5,64,5,5]
            for _ in range(50):
                sample_features_ext_list2.append(sample_features)
            sample_features_ext2 = self.stack(sample_features_ext_list2)
            sample_features_ext = concat0dim((sample_features_ext1, sample_features_ext2))   #[95,5,64,5,5]
            batch_features_ext_list = []
            for _ in range(5):
                batch_features_ext_list.append(batch_features)
            batch_features_ext = self.stack(batch_features_ext_list)  # [5,95,64,5,5]
            batch_features_ext = self.transpose(batch_features_ext, (1, 0, 2, 3, 4))  # [95,5,64,5,5]

        else:
            sample_features_ext_list = []
            batch_features_ext_list = []
            for _ in range(5):
                sample_features_ext_list.append(sample_features)
                batch_features_ext_list.append(batch_features)
            sample_features_ext = self.stack(sample_features_ext_list)
            batch_features_ext = self.stack(batch_features_ext_list)
            batch_features_ext = self.transpose(batch_features_ext, (1, 0, 2, 3, 4))

        relation_pairs = self.concat2dim((sample_features_ext, batch_features_ext))     #[95,5,128,5,5]
        relation_pairs = self.reshape(relation_pairs, (-1, self.feature_dim*2, 5, 5))     #[475,128,5,5]


        #put relation pairs into relation network
        x = relation_pairs
        out = self.Relationlayer1(x)
        out = self.Relationlayer2(out)
        out = self.reshape(out, (out.shape[0], -1))
        out = self.relu(self.fc1(out))
        out = self.sigmoid(self.fc2(out))
        out = self.reshape(out, (-1, self.class_num))
        return out

def weight_init(custom_cell):

    for _, m in custom_cell.cells_and_names():
        if isinstance(m, nn.Conv2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.set_data(Tensor(np.random.normal(0, np.sqrt(2. / n),
                                                      m.weight.data.shape).astype("float32")))
            if m.bias is not None:
                m.bias.set_data(
                    Tensor(np.zeros(m.bias.data.shape, dtype="float32")))
        elif isinstance(m, nn.BatchNorm2d):
            m.gamma.set_data(
                Tensor(np.ones(m.gamma.data.shape, dtype="float32")))
            m.beta.set_data(
                Tensor(np.zeros(m.beta.data.shape, dtype="float32")))
        elif isinstance(m, nn.Dense):
            m.weight.set_data(Tensor(np.random.normal(
                0, 0.01, m.weight.data.shape).astype("float32")))
            if m.bias is not None:
                m.bias.set_data(
                    Tensor(np.ones(m.bias.data.shape, dtype="float32")))



def main():
    #Step 1 : create ckpts_dir
    if not os.path.exists(args.ckpts_dir):
        os.makedirs(args.ckpts_dir)

    # Step 1: init data folders
    print("init data folders")
    _, metatest_character_folders = dt.omniglot_character_folders(data_path=args.data_path)

    #Step 4 : init networks
    print("init neural networks")
    encoder_relation = Encoder_Relation(cfg.feature_dim, cfg.relation_dim)
    encoder_relation.set_train(False)
    weight_init(encoder_relation)

    #load parameters
    if os.path.exists(os.path.join(args.ckpts_dir,
                                   str("omniglot_encoder_relation_network" + str(cfg.class_num) + "way_" +
                                       str(cfg.sample_num_per_class) + "shot.ckpt"))):
        param_dict = load_checkpoint(os.path.join(args.ckpts_dir,
                                                  str("omniglot_encoder_relation_network" + str(cfg.class_num)
                                                      + "way_" + str(cfg.sample_num_per_class) + "shot.ckpt")))
        load_param_into_net(encoder_relation, param_dict)
        print("successfully load parameters")
    else:
        print("Error:can not load checkpoint")




    total_accuracy = 0.0
    print("=" * 10 + "Testing" + "=" * 10)
    for episode in range(args.episode):



        total_rewards = 0
        accuracies = []
        for _ in range(cfg.test_episode):
            degrees = random.choice([0, 90, 180, 270])
            flip = random.choice([True, False])
            task = dt.OmniglotTask(metatest_character_folders, cfg.class_num, cfg.sample_num_per_class,
                                   cfg.sample_num_per_class)
            sample_dataloader = dt.get_data_loader(task, num_per_class=cfg.sample_num_per_class, split="train",
                                                   shuffle=False, rotation=degrees, flip=flip)
            test_dataloader = dt.get_data_loader(task, num_per_class=cfg.sample_num_per_class, split="test",
                                                 shuffle=True, rotation=degrees, flip=flip)
            test_samples, _ = sample_dataloader.__iter__().__next__()
            test_batches, test_batch_labels = test_dataloader.__iter__().__next__()

            # concat samples and batches
            test_input = concat0dim((test_samples, test_batches))
            test_relations = encoder_relation(test_input)

            predict_labels = ops.Argmax(axis=1, output_type=ms.int32)(test_relations).asnumpy()
            test_batch_labels = test_batch_labels.asnumpy().astype(np.int32)
            rewards = [1 if predict_labels[j] == test_batch_labels[j] else 0 for j in range(cfg.class_num)]     #报错
            total_rewards += np.sum(rewards)
            accuracy = np.sum(rewards) / 1.0 / cfg.class_num / cfg.sample_num_per_class
            accuracies.append(accuracy)

        test_accuracy, h = mean_confidence_interval(accuracies)
        total_accuracy += test_accuracy
        print('-' * 5 + 'Episode {}/{}'.format(episode + 1, args.episode) + '-' * 5)
        print("test accuracy: %.4f or %.4f%%  h: %f" % (test_accuracy, test_accuracy*100, h))

    print("aver_accuracy : %.2f" % (total_accuracy/args.episode*100))


if __name__ == '__main__':
    main()
