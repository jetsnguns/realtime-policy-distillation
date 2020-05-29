import math
from typing import Sequence

import ray
from ray.rllib.evaluation import collect_metrics
from ray.rllib.evaluation.worker_set import WorkerSet
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models import ModelCatalog
from ray.rllib.utils.annotations import override
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.torch_ops import huber_loss
import torch as t
from torch import nn


class BasicArch(nn.Module):
    def __init__(self, conv_channels: Sequence[int], fc_dims: int, out_dims: int):
        super().__init__()

        self.feature_extractor = nn.Sequential(
            *[nn.Sequential(nn.Conv2d(in_ch, out_ch, 3, stride=2), nn.ReLU())
              for in_ch, out_ch in zip(conv_channels[:-1], conv_channels[1:])]
        )

        self.avg_pooling = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Linear(conv_channels[-1], fc_dims),
            nn.ReLU(),
        )
        self.out = nn.Linear(fc_dims, out_dims)

        self.conv_channels = conv_channels

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.avg_pooling(x)
        x = x.reshape(-1, self.conv_channels[-1])
        x = self.fc(x)
        x = self.out(x)
        return x


class CustomModel(TorchModelV2, nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name, **kwargs):
        assert num_outputs == action_space.n

        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        self.trainer = BasicArch([4, 8, 16], 32, num_outputs)
        self.student = BasicArch([4, 4, 4], 32, num_outputs)

        self.nets = {
            "trainer": self.trainer,
            "student": self.student,
        }
        self.custom_net = None


    @override(TorchModelV2)
    def forward(self, input_dict, state, seq_lens):
        x = input_dict['obs'].permute((0, 3, 1, 2)).float()
        if self.custom_net is not None:
            x = self.nets[self.custom_net](x)
        else:
            x = self.trainer(x)
        return x, state

    def get_advantages_or_q_values(self, model_outputs):
        return model_outputs


ModelCatalog.register_custom_model("custom_model", CustomModel)


class Loss:
    @staticmethod
    def choose(q, a):
        return t.gather(q, 1, a).reshape(-1)

    @staticmethod
    def calc_q(model, obs):
        return model(obs.permute((0, 3, 1, 2)).float())

    @staticmethod
    def calc_prev_next_q(model, train_batch):
        return {
            'prev': Loss.calc_q(model, train_batch[SampleBatch.CUR_OBS]),
            'next': Loss.calc_q(model, train_batch[SampleBatch.NEXT_OBS]),
        }

    @staticmethod
    def q_loss(q_prev, q_next, next_action_q, train_batch, gamma):
        lhs = Loss.choose(q_prev, train_batch[SampleBatch.ACTIONS].reshape(-1, 1))
        a_next = t.argmax(next_action_q, 1, keepdim=True)

        rhs = (
            train_batch[SampleBatch.REWARDS] +
            gamma * Loss.choose(q_next, a_next) * (1.0 - train_batch[SampleBatch.DONES].float())
        )

        td_error = lhs - rhs.detach()
        huber = huber_loss(td_error)
        loss = huber * train_batch['weights']

        return loss.mean(), td_error

    @staticmethod
    def kl_loss(q1, q2, tau):
        log_p1 = t.log_softmax(q1 / tau, 1)
        log_p2 = t.log_softmax(q2 / tau, 1)

        # print(log_p1.shape, log_p2.shape)

        return (t.exp(log_p1) * (log_p1 - log_p2)).mean()

    def __init__(self, policy, model, target_model, train_batch, gamma, tau):
        trainer = self.calc_prev_next_q(model.trainer, train_batch)
        trainer_target = self.calc_prev_next_q(target_model.trainer, train_batch)
        student = self.calc_prev_next_q(model.student, train_batch)

        self.trainer_q, self.trainer_td_error = self.q_loss(
            trainer['prev'], trainer['next'], trainer_target['next'], train_batch, gamma,
        )
        self.student_q, self.student_td_error = self.q_loss(
            student['prev'], student['next'], trainer_target['next'], train_batch, gamma,
        )
        self.student_kl = self.kl_loss(trainer['prev'].detach(), student['prev'], tau)

        self.loss = self.trainer_q + self.student_q + self.student_kl
        self.td_error = full_detach(self.trainer_td_error)

        self.stats = {
            "our_td_error": self.td_error,
            "our_trainer_q": full_detach(self.trainer_q).item(),
            "our_student_q": full_detach(self.student_q).item(),
            "our_student_kl": full_detach(self.student_kl).item(),
        }

        policy.q_loss = self


def loss_callback(policy, model, _, train_batch):
    loss = Loss(
        policy,
        policy.q_model,
        policy.target_q_model,
        train_batch,
        policy.config['gamma'],
        policy.config['model']['custom_options']['tau'],
    )
    return loss.loss


def custom_eval_fn(trainer, workers: WorkerSet):
    def change_q_func(net_id):
        def foo(policy, _):
            policy.model.custom_net = net_id
        return foo

    result = {}

    for net_id in trainer.get_policy().q_model.nets:
        print(net_id)

        workers.foreach_policy(change_q_func(net_id))

        if trainer.config["evaluation_num_workers"] == 0:
            for _ in range(trainer.config["evaluation_num_episodes"]):
                trainer.evaluation_workers.local_worker().sample()
        else:
            num_rounds = int(
                math.ceil(trainer.config["evaluation_num_episodes"] /
                          trainer.config["evaluation_num_workers"]))
            num_workers = len(trainer.evaluation_workers.remote_workers())
            num_episodes = num_rounds * num_workers
            for i in range(num_rounds):
                ray.get([
                    w.sample.remote()
                    for w in trainer.evaluation_workers.remote_workers()
                ])

        metrics = collect_metrics(trainer.evaluation_workers.local_worker(),
                                  trainer.evaluation_workers.remote_workers())

        result[net_id] = metrics

    workers.foreach_policy(change_q_func(None))

    return result


def full_detach(tensor):
    return tensor.cpu().detach().numpy()