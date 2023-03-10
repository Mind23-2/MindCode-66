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
#import moxing as mox
import mindspore as ms
import mindspore.nn as nn
from mindspore import context
from mindspore.context import ParallelMode
import mindspore.ops as ops
from mindspore import Tensor
from mindspore.communication.management import init, get_rank
import src.dataset as dt
from src.lr_generator import _generate_steps_lr
from src.config import relationnet_cfg as cfg
from src.relationnet import Encoder_Relation, weight_init, TrainOneStepCell
from src.net_train import train

parser = argparse.ArgumentParser(description="One Shot Visual Recognition")
parser.add_argument("-g", "--gpu", type=int, default=0)
parser.add_argument("-u", "--hidden_unit", type=int, default=10)
parser.add_argument("-dt", "--device_target", type=str, default='Ascend', choices=("Ascend", "GPU", "CPU"),
                    help="Device target, support Ascend, GPU and CPU.")
parser.add_argument("-di", "--device_id", type=int, default=6, help='device id of GPU or Ascend. (Default: 0)')
parser.add_argument("--ckpts_dir", default='./output/', help='the path of output')
parser.add_argument("--data_path", default='/home/jialing/FSL/data/omniglot_resized/',
                    help="Path where the dataset is saved")
parser.add_argument("--data_url", default=None)
parser.add_argument("--train_url", default=None)
parser.add_argument("--cloud", default=None, help='if run on cloud')
args = parser.parse_args()

# init operators
scatter = ops.ScatterNd()
concat0dim = ops.Concat(axis=0)


def main():
    local_data_url = args.data_path
    local_train_url = args.ckpts_dir
    # if run on the cloud
    if args.cloud:
        import moxing as mox
        local_data_url = './cache/data'
        local_train_url = './cache/ckpt'
        device_target = args.device_target
        device_num = int(os.getenv("RANK_SIZE"))
        device_id = int(os.getenv("DEVICE_ID"))
        context.set_context(mode=context.GRAPH_MODE, device_target=args.device_target)
        context.set_context(save_graphs=False)
        if device_target == "Ascend":
            context.set_context(device_id=device_id)

            if device_num > 1:
                cfg.episode = int(cfg.episode / 2)
                cfg.learning_rate = cfg.learning_rate*2
                context.reset_auto_parallel_context()
                context.set_auto_parallel_context(device_num=device_num, parallel_mode=ParallelMode.DATA_PARALLEL,
                                                  gradients_mean=True)
                init()
                local_data_url = os.path.join(local_data_url, str(device_id))
                local_train_url = os.path.join(local_train_url, "_" + str(get_rank()))
        elif device_target == "GPU":
            if device_num > 1:
                init()
                context.reset_auto_parallel_context()
                context.set_auto_parallel_context(device_num=device_num, parallel_mode=ParallelMode.DATA_PARALLEL,
                                                  gradients_mean=True)
        else:
            raise ValueError("Unsupported platform.")
        import moxing as mox
        mox.file.copy_parallel(src_url=args.data_url, dst_url=local_data_url)
    else:
        # run on the local server
        context.set_context(mode=context.GRAPH_MODE, device_target=args.device_target, device_id=args.device_id)
        context.set_context(save_graphs=False)

    # Step 1 : create output dir
    if not os.path.exists(local_train_url):
        os.makedirs(local_train_url)

    # Step 2 : init operators

    # Step 3 : init data folders
    print("init data folders")
    metatrain_character_folders, metatest_character_folders = dt.omniglot_character_folders(data_path=local_data_url)

    # Step 4 : init networks
    print("init neural networks")
    encoder_relation = Encoder_Relation(cfg.feature_dim, cfg.relation_dim)
    weight_init(encoder_relation)

    # Step 5 : load parameters
    load_ckpts = False
    # if os.path.exists(os.path.join(local_train_url,
    #                                str("omniglot_encoder_relation_network" + str(cfg.class_num) + "way_" +
    #                                    str(cfg.sample_num_per_class) + "shot.ckpt"))):
    #     param_dict = load_checkpoint(os.path.join(args.ckpts_dir,
    #                                               str("omniglot_encoder_relation_network" + str(cfg.class_num) + "way_"
    #                                                   + str(cfg.sample_num_per_class) + "shot.ckpt")))
    #     load_param_into_net(encoder_relation, param_dict)
    #     print("successfully load the param")
    #     load_ckpts = True

    print("init optim, loss")
    if load_ckpts:
        lr = Tensor(nn.piecewise_constant_lr(milestone=[50000 * i for i in range(1, 21)],
                                             learning_rates=[0.0001 * 0.5 ** i for i in range(0, 20)]),
                    dtype=ms.float32)
    else:
        lr = _generate_steps_lr(lr_init=0.0005, lr_max=cfg.learning_rate, total_steps=1000000, warmup_steps=100)
        #lr = generate_periodic_lr(lr_init=0.0005, decay_steps=40000, period_steps=200000,
        #                          total_steps=1000000, warmup_steps=100)
    optim = nn.Adam(encoder_relation.trainable_params(), learning_rate=lr)

    print("init loss function and grads")
    criterion = nn.MSELoss()
    netloss = nn.WithLossCell(encoder_relation, criterion)
    net_g = TrainOneStepCell(netloss, optim)


    # train
    train(metatrain_character_folders=metatrain_character_folders, metatest_character_folders=metatest_character_folders
          , netloss=netloss, net_g=net_g, encoder_relation=encoder_relation, local_train_url=local_train_url, args=args)


if __name__ == '__main__':
    main()
