"""Regression coverage for verified toy rewards and policy-learning interface."""
import dataclasses
import math
import tempfile
import unittest

import torch

from latent_brain.cookies import (
    ACTIONS, CookieLedger, DebugShadowAdapter, RewardFlyBrain, RewardTrainConfig,
    ToyDebugLab, VerifiedEvent, episode, evaluate_reward, load_reward,
    save_reward, train_reward, evaluate_rule_baseline,
)


class CookieSafety(unittest.TestCase):
    def test_unverified_event_never_mints_cookies(self):
        env = ToyDebugLab(seed=12, instance=1)
        fake = VerifiedEvent('toy-12-1', 'fake', 'case_resolved', 'verified-toy-runner', 'claimed', 'wrong')
        self.assertEqual(env.ledger.award(fake), 0)
        self.assertEqual(env.ledger.total, 0)

    def test_no_duplicate_rewards_for_same_evidence(self):
        env = ToyDebugLab(seed=7, instance=99)
        env.step(0)
        self.assertEqual(env.ledger.total, 1)
        env.step(0)
        self.assertEqual(env.ledger.total, 1)
        correct_action = 1 + env.answer
        env.step(correct_action)
        self.assertEqual(env.ledger.total, 4)
        env.step(correct_action)
        self.assertEqual(env.ledger.total, 4)
        env.step(3)
        self.assertTrue(env.success)
        self.assertEqual(env.ledger.total, 9)

    def test_early_report_never_produces_success_cookies(self):
        env = ToyDebugLab(seed=22, instance=1)
        env.step(3)
        self.assertFalse(env.success)
        self.assertEqual(env.ledger.total, 0)
        self.assertFalse(env.done)
        env.step(0)
        self.assertEqual(env.ledger.total, 1)

    def test_report_requires_inspection_and_verification(self):
        env = ToyDebugLab(seed=22, instance=2)
        env.step(env.answer+1)  # rejected: no prior inspection
        env.step(3)
        self.assertFalse(env.success)
        self.assertEqual(env.ledger.total, 0)
        env.step(0)
        env.step(env.answer+1)
        env.step(3)
        self.assertTrue(env.success)
        self.assertEqual(env.ledger.total, 9)

    def test_blind_hypothesis_is_denied_without_rewards(self):
        env = ToyDebugLab(seed=84, instance=3)
        actual_reward, done = env.step(env.answer+1)
        self.assertLess(actual_reward, 0)
        self.assertFalse(done)
        self.assertFalse(env.confirmed)
        self.assertFalse(env.tested)
        self.assertEqual(env.ledger.total, 0)

    def test_failed_test_does_not_forge_confirmation(self):
        env = ToyDebugLab(seed=1, instance=1)
        env.step(0)
        env.step(1 + (1-env.answer))
        self.assertFalse(env.confirmed)
        self.assertEqual(env.ledger.total, 1)
        env.step(3)
        self.assertFalse(env.success)

    def test_ledger_rejects_false_and_duplicate_receipt(self):
        key = b'a' * 32
        env = ToyDebugLab(seed=1, instance=5, key=key)
        receipt = env._receipt('inspected')
        self.assertEqual(env.ledger.award(receipt), 1)
        self.assertEqual(env.ledger.award(receipt), 0)
        self.assertEqual(env.ledger.award(dataclasses.replace(receipt, kind='case_resolved')), 0)
        self.assertEqual(env.ledger.award(dataclasses.replace(receipt, source='agent-self-report')), 0)

    def test_bad_values_rejected(self):
        with self.assertRaises(ValueError):
            CookieLedger(b'short')
        env = ToyDebugLab(seed=1, instance=5)
        for bad in (-1, 4, 'inspect', True):
            with self.assertRaises(ValueError):
                env.step(bad)


class LearningInterface(unittest.TestCase):
    def test_shadow_never_executes_and_can_reset(self):
        brain = RewardFlyBrain()
        adapter = DebugShadowAdapter(brain)
        observation = ToyDebugLab(seed=3, instance=4).observation()[0].tolist()
        first = adapter.suggest(observation)
        self.assertEqual(first['verification'], 'not_executed')
        self.assertEqual(first['authority'], 'advisory_only')
        self.assertIn(first['proposal'], ACTIONS)
        adapter.reset()
        self.assertEqual(first, adapter.suggest(observation))
        for bad in ([1], ['x']*18, [float('nan')]*18, [True]*18):
            with self.assertRaises(ValueError):
                adapter.suggest(bad)

    def test_budget_changes_number_of_imagined_steps(self):
        brain = RewardFlyBrain()
        observation = ToyDebugLab(seed=3, instance=5).observation()
        with torch.no_grad():
            logits, _, _, hidden = brain(observation, brain.initial_state())
            for depth in (1,2,4):
                scores = brain.imagined_scores(hidden, logits, depth)
                self.assertEqual(scores.shape, (4,))
                self.assertTrue(torch.isfinite(scores).all())

    def test_toy_eval_never_claims_external_code_execution(self):
        brain = RewardFlyBrain()
        result = evaluate_reward(brain, eval_seed=777, cases=5)
        self.assertEqual(result['verification'], 'toy_execution_verified')
        self.assertEqual(result['real_toy_replays'], 5)

    def test_strong_handwritten_baseline_solves_toy_lab(self):
        result = evaluate_rule_baseline(eval_seed=100, cases=20)
        self.assertEqual(result['solved'], 20)
        self.assertEqual(result['total_steps'], 60)
        self.assertEqual(result['cookies'], 180)

    def test_checkpoint_roundtrip(self):
        brain = RewardFlyBrain()
        with tempfile.TemporaryDirectory() as td:
            file = f'{td}/reward.pt'
            save_reward(brain, file, {'sample': 'tested'})
            restored, metadata = load_reward(file)
            self.assertEqual(metadata, {'sample':'tested'})
            for a,b in zip(brain.parameters(),restored.parameters()):
                self.assertTrue(torch.equal(a,b))

    def test_small_learning_cycle_changes_parameters(self):
        from copy import deepcopy
        config=RewardTrainConfig(seed=4, episodes=3)
        torch.manual_seed(4)
        prior=RewardFlyBrain()
        trained, info=train_reward(config)
        self.assertEqual(info['training_episodes'], 3)
        self.assertTrue(any(not torch.equal(x,y) for x,y in zip(prior.parameters(),trained.parameters())))

