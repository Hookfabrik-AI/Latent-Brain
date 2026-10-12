"""Verified-reward experiments for a tiny *simulated* debugging workflow.

Not connected to a private coding agent.  All toy outcomes originate from the
independent environment.  Consumers must supply their own trusted verifier.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import hmac
import json
import secrets

import torch
from torch import nn
import torch.nn.functional as F

from .fly import FlyBrain, BUDGETS, INPUT_SIZE

ACTIONS = ('inspect', 'test_a', 'test_b', 'report')
EVENT_COOKIES = {'inspected': 1, 'hypothesis_confirmed': 3, 'case_resolved': 5}


@dataclass(frozen=True)
class VerifiedEvent:
    """Evidence receipt produced by a trusted test controller, not a model."""
    case_id: str
    event_id: str
    kind: str
    source: str
    evidence: str
    signature: str


class CookieLedger:
    """Fail-closed award ledger. A signing key belongs ONLY to the verifier.

    HMAC protects the toy-example receipt from client-side forgery; it does not
    authenticate the actual behavior of an external tool/agent. A real host
    must mint receipts only after validating revision-bound execution evidence.
    """

    def __init__(self, verifier_key: bytes):
        if not verifier_key or len(verifier_key) < 16:
            raise ValueError('verifier key must contain at least 16 bytes')
        self.__key = verifier_key
        self.__awarded: set[str] = set()
        self.total = 0

    @staticmethod
    def _message(event: VerifiedEvent) -> bytes:
        return json.dumps([event.case_id, event.event_id, event.kind,
                           event.source, event.evidence], separators=(',', ':')).encode()

    def award(self, event: VerifiedEvent) -> int:
        if not isinstance(event, VerifiedEvent):
            return 0
        if (event.kind not in EVENT_COOKIES or event.source != 'verified-toy-runner'
                or not event.case_id or not event.event_id or not event.evidence):
            return 0
        expected = hmac.new(self.__key, self._message(event), hashlib.sha256).hexdigest()
        if not hmac.compare_digest(expected, event.signature):
            return 0
        identity = f'{event.case_id}\x00{event.event_id}'
        if identity in self.__awarded:
            return 0
        self.__awarded.add(identity)
        reward = EVENT_COOKIES[event.kind]
        self.total += reward
        return reward


class ToyDebugLab:
    """Small synthetic workflow; never executes code or touches real files.

    Correct hypothesis is hidden until `inspect`; tests can confirm or refute.
    The same event cannot earn cookies twice. Random case IDs avoid cross-run
    reuse of receipts. The model never receives the HMAC key or reward control.
    """

    def __init__(self, *, seed: int, instance: int, key: bytes | None = None):
        import random
        rng = random.Random(seed)
        self.case_id = f'toy-{seed}-{instance}'
        self.answer = rng.randrange(2)
        self.symptom = rng.randrange(4)
        self.noise = (rng.randrange(2), rng.randrange(2))
        self.__key = key if key is not None else secrets.token_bytes(32)
        self.ledger = CookieLedger(self.__key)
        self.inspected = False
        self.tested: set[int] = set()
        self.confirmed = False
        self.done = False
        self.success = False
        self.steps = 0
        self.previous_action = None
        self.previous_success = False

    def observation(self) -> torch.Tensor:
        sym = self.symptom
        v = [float(self.inspected),
             float(self.inspected and self.answer == 0),
             float(self.inspected and self.answer == 1),
             float(0 in self.tested), float(1 in self.tested),
             float(self.confirmed), float(bool(self.tested) and not self.confirmed),
             min(self.steps / 8.0, 1.0), float(self.done),
             float(sym % 2 == 0), float(sym % 2 == 1)]
        v += [float(self.previous_action == a) for a in range(4)]
        v += [float(self.previous_success), *map(float, self.noise)]
        assert len(v) == INPUT_SIZE
        return torch.tensor([v], dtype=torch.float32)

    def _receipt(self, kind: str) -> VerifiedEvent:
        evidence = hashlib.sha256(f'{self.case_id}|{kind}|{self.answer}'.encode()).hexdigest()
        event = VerifiedEvent(self.case_id, f'{kind}:{self.case_id}', kind,
                              'verified-toy-runner', evidence, '')
        signature = hmac.new(self.__key, CookieLedger._message(event), hashlib.sha256).hexdigest()
        return VerifiedEvent(event.case_id, event.event_id, kind, event.source, evidence, signature)

    def step(self, action: int) -> tuple[float, bool]:
        if self.done:
            raise ValueError('episode is finished')
        if type(action) is not int or not 0 <= action < 4:
            raise ValueError('unsupported action')
        reward, executed_success = -0.15, False  # every action has a cost
        if action == 0:
            self.inspected = True
            reward += self.ledger.award(self._receipt('inspected'))
            executed_success = True
        elif action in (1, 2):
            if not self.inspected:
                # The controller refuses blind tests. The model cannot earn
                # cookies by guessing a hypothesis before gathering evidence.
                reward -= 0.75
            else:
                candidate = action - 1
                already_tested = candidate in self.tested
                self.tested.add(candidate)
                if candidate == self.answer:
                    self.confirmed = True
                    if not already_tested:
                        reward += self.ledger.award(self._receipt('hypothesis_confirmed'))
                    executed_success = True
        else:
            self.success = self.confirmed and self.inspected
            if self.success:
                self.done = True
                reward += self.ledger.award(self._receipt('case_resolved'))
                executed_success = True
            else:
                # Controller refuses an unsupported completion. The task remains
                # open, so the learner can gather evidence before retrying.
                reward -= 0.75
        self.steps += 1
        if self.steps >= 8:
            self.done = True
        self.previous_action = action
        self.previous_success = executed_success
        return reward, self.done


class RewardFlyBrain(nn.Module):
    """Reusable GRU FlyBrain with a learned state-value head.

    18 input dimensions here encode debug workflow observations, NOT the maze
    observations of v0.3. Its checkpoint is intentionally incompatible.
    """

    def __init__(self, hidden_size: int = 32):
        super().__init__()
        self.backbone = FlyBrain(hidden_size)
        self.value = nn.Linear(hidden_size, 1)

    def initial_state(self):
        return self.backbone.initial_state()

    def forward(self, observation, hidden):
        action_logits, budget_logits, next_hidden = self.backbone.step(observation, hidden)
        return action_logits.squeeze(0), budget_logits.squeeze(0), self.value(next_hidden).squeeze(), next_hidden

    def imagined_scores(self, hidden: torch.Tensor, logits: torch.Tensor, depth: int) -> torch.Tensor:
        """Bounded learned lookahead. Imagined states are NOT real evidence."""
        if depth not in BUDGETS:
            raise ValueError('invalid thinking depth')
        scores = []
        for a in range(4):
            imagined = self.backbone.imagine(hidden, torch.tensor([a]))
            score = logits[a]
            # Subsequent steps follow the learned policy in imagined states.
            for k in range(depth):
                score = score + (0.12 * (0.80 ** k)) * self.value(imagined)[0, 0]
                if k + 1 < depth:
                    follow = int(self.backbone.policy(imagined).argmax(-1).item())
                    imagined = self.backbone.imagine(imagined, torch.tensor([follow]))
            scores.append(score)
        return torch.stack(scores)


@dataclass(frozen=True)
class RewardTrainConfig:
    seed: int = 42
    episodes: int = 450
    hidden_size: int = 32
    learning_rate: float = 0.003
    max_steps: int = 8


def episode(brain: RewardFlyBrain, env: ToyDebugLab, *, train: bool, forced_budget: int | None = None):
    """Run through the toy verifier. Train mode returns differentiable losses."""
    hidden = brain.initial_state()
    records = []
    total = 0.0
    choices = []
    for _ in range(8):
        obs = env.observation()
        logits, budget_logits, value, next_hidden = brain(obs, hidden)
        budget_dist = torch.distributions.Categorical(logits=budget_logits)
        budget_ix = budget_dist.sample() if train and forced_budget is None else torch.tensor(
            BUDGETS.index(forced_budget) if forced_budget is not None else int(budget_logits.argmax().item()))
        depth = BUDGETS[int(budget_ix.item())]
        scores = brain.imagined_scores(next_hidden, logits, depth)
        action_dist = torch.distributions.Categorical(logits=scores)
        action_tensor = action_dist.sample() if train else scores.argmax()
        action = int(action_tensor.item())
        reward, done = env.step(action)
        # Real observed successor; used ONLY for training imagination consistency.
        if train:
            future_obs = env.observation()
            _, _, _, real_next_hidden = brain(future_obs, next_hidden)
            imagined = brain.backbone.imagine(next_hidden, action_tensor.reshape(1))
            imagination_loss = F.mse_loss(imagined, real_next_hidden.detach())
            logprob = action_dist.log_prob(action_tensor) + budget_dist.log_prob(budget_ix)
            entropy = action_dist.entropy() + 0.1 * budget_dist.entropy()
            records.append((reward - 0.01 * depth, logprob, value, entropy, imagination_loss))
        choices.append((action, depth))
        total += reward
        hidden = next_hidden.detach() if not train else next_hidden
        if done:
            break
    if not train:
        return {'solved': env.success, 'cookies': env.ledger.total,
                'reward': total, 'steps': env.steps, 'choices': choices,
                'verification': 'toy_execution_verified'}
    returns, running = [], 0.0
    for reward, *_ in reversed(records):
        running = reward + 0.97 * running
        returns.append(running)
    returns.reverse()
    policy_loss = 0.0
    critic_loss = 0.0
    entropy_total = 0.0
    future_loss = 0.0
    for (_, logp, v, ent, imagined_loss), outcome in zip(records, returns):
        target = v.new_tensor(outcome)
        policy_loss = policy_loss - logp * (target - v.detach())
        critic_loss = critic_loss + F.smooth_l1_loss(v, target)
        entropy_total = entropy_total + ent
        future_loss = future_loss + imagined_loss
    n = len(records)
    loss = (policy_loss / n + 0.30 * critic_loss / n
            + 0.10 * future_loss / n - 0.02 * entropy_total / n)
    return loss, env.success, total


def train_reward(config: RewardTrainConfig):
    """Policy-gradient learning from environment-produced reward, no teacher labels."""
    if config.episodes < 1 or config.max_steps != 8:
        raise ValueError('episodes must be positive; supported horizon is 8')
    torch.manual_seed(config.seed)
    torch.set_num_threads(min(4, torch.get_num_threads()))
    brain = RewardFlyBrain(config.hidden_size)
    opt = torch.optim.Adam(brain.parameters(), lr=config.learning_rate)
    running_success = []
    for i in range(config.episodes):
        # Train seeds are distinct from evaluation namespace, by construction.
        env = ToyDebugLab(seed=config.seed * 100000 + i, instance=i)
        loss, solved, _ = episode(brain, env, train=True)
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(brain.parameters(), 2.0)
        opt.step()
        running_success.append(int(solved))
    return brain.eval(), {'training_episodes': config.episodes,
                          'recent_training_success': sum(running_success[-50:]) / min(50, len(running_success)),
                          'seed': config.seed, 'objective': 'verified toy rewards, policy-gradient + value + latent-consistency'}


def evaluate_reward(brain: RewardFlyBrain, *, eval_seed: int = 9001,
                    cases: int = 100, forced_budget: int | None = None):
    if cases <= 0 or eval_seed < 0:
        raise ValueError('invalid evaluation request')
    brain.eval()
    successes = cookies = steps = 0
    depths: dict[int, int] = {1: 0, 2: 0, 4: 0}
    with torch.inference_mode():
        for i in range(cases):
            # Disjoint case IDs and deterministic worlds across compared agents.
            env = ToyDebugLab(seed=10**10 + eval_seed * 100000 + i, instance=i)
            result = episode(brain, env, train=False, forced_budget=forced_budget)
            successes += int(result['solved'])
            cookies += result['cookies']
            steps += result['steps']
            for _, depth in result['choices']:
                depths[depth] += 1
    return {'solved': successes, 'cases': cases, 'cookies': cookies,
            'total_steps': steps, 'depths': depths, 'real_toy_replays': cases,
            'verification': 'toy_execution_verified'}


def save_reward(brain: RewardFlyBrain, filename: str, meta: dict):
    torch.save({'format': 'reward-fly-v1', 'weights': brain.state_dict(),
                'hidden_size': brain.backbone.hidden_size, 'training': meta}, filename)


def load_reward(filename: str):
    data = torch.load(filename, map_location='cpu', weights_only=True)
    if data['format'] != 'reward-fly-v1':
        raise ValueError('incompatible checkpoint')
    brain = RewardFlyBrain(int(data['hidden_size']))
    brain.load_state_dict(data['weights'], strict=True)
    return brain.eval(), data['training']


class DebugShadowAdapter:
    """Non-executing neutral adapter: An external agent supplies validated 18D observations.

    No workspace access, no private agent imports and NO reward ingestion. A
    trusted host must separately own execution, evidence and replay records.
    """

    def __init__(self, brain: RewardFlyBrain):
        self.brain = brain.eval()
        self.hidden = brain.initial_state()

    def reset(self):
        self.hidden = self.brain.initial_state()

    @torch.inference_mode()
    def suggest(self, observation: list[float]) -> dict:
        if len(observation) != INPUT_SIZE or any(type(v) not in (float, int) or not 0 <= v <= 1 for v in observation):
            raise ValueError('18 bounded numeric observation values required')
        x = torch.tensor([observation], dtype=torch.float32)
        logits, budget, _, self.hidden = self.brain(x, self.hidden)
        depth = BUDGETS[int(budget.argmax().item())]
        action = int(self.brain.imagined_scores(self.hidden, logits, depth).argmax().item())
        return {'proposal': ACTIONS[action], 'thinking_budget': depth,
                'authority': 'advisory_only', 'verification': 'not_executed'}


def evaluate_rule_baseline(*, eval_seed: int = 9001, cases: int = 100) -> dict:
    """Transparent non-neural ceiling for this easy toy observation contract."""
    if cases <= 0 or eval_seed < 0:
        raise ValueError('invalid evaluation request')
    successes = cookies = steps = 0
    for i in range(cases):
        env = ToyDebugLab(seed=10**10 + eval_seed * 100000 + i, instance=i)
        while not env.done:
            observation = env.observation()[0]
            if observation[0] == 0:  # no investigation yet
                action = 0
            elif not env.confirmed:
                action = 1 if observation[1] == 1 else 2
            else:
                action = 3
            env.step(action)
        successes += int(env.success)
        cookies += env.ledger.total
        steps += env.steps
    return {'solved': successes, 'cases': cases, 'cookies': cookies,
            'total_steps': steps, 'verification': 'toy_execution_verified'}
