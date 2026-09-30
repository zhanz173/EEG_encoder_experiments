import json
from pathlib import Path
import sys
import tempfile
import unittest

from experiments.sweep_temporal import build_plan, parse_args, run_queue


FAKE_TRAIN = '''
import json, os, sys, time
from pathlib import Path
root, name, lam, fail = sys.argv[1:]
root = Path(root)
root.mkdir(parents=True, exist_ok=True)
start = time.time()
time.sleep(.3)
(root / 'worker.json').write_text(json.dumps(dict(gpu=os.environ['CUDA_VISIBLE_DEVICES'],
    start=start, end=time.time())))
if fail == 'yes':
    raise SystemExit(3)
(root / 'best.pt').write_text('fake checkpoint')
(root / 'config.json').write_text(json.dumps(dict(architecture='slow-fast',lambda_rate=float(lam),seed=42)))
metrics = dict(huber=.2,nmse=.3,estimated_bits_per_channel_sample=.1)
(root / 'results.json').write_text(json.dumps(dict(val=metrics,test=metrics,best_epoch=1)))
'''


class QueueTests(unittest.TestCase):
    def test_plan_defaults_and_reserved_overrides(self):
        args = parse_args(['--manifest', 'data.csv', '--shards-dir', 'shards'])
        plan = build_plan(args)
        self.assertEqual(args.gpus, ['0', '1'])
        self.assertEqual(len(plan['jobs']), 9)
        for job in plan['jobs']:
            command = job['command']
            self.assertEqual(command[command.index('--slow-dim') + 1], '32')
            self.assertEqual(command[command.index('--fast-dim') + 1], '32')
            self.assertEqual(command[command.index('--channel-mode') + 1], 'joint')
        for extra in (['--gpus', '0', '0'], ['--lambdas', 'nan'],
                      ['--extra', '--lambda-rate=.8']):
            with self.assertRaises(SystemExit):
                parse_args(['--manifest', 'data.csv', '--shards-dir', 'shards', *extra])

    def test_dual_slots_failure_continuation_and_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            jobs = []
            for i in range(4):
                dest = root / f'job{i}'
                jobs.append(dict(name=f'job{i}', architecture='slow-fast', lambda_rate=.1,
                                 seed=42, run_dir=str(dest), command=[sys.executable, '-c',
                                 FAKE_TRAIN, str(dest), f'job{i}', '.1', 'yes' if i == 1 else 'no']))
            plan = dict(threads=1, jobs=jobs)
            self.assertFalse(run_queue(plan, root, ['0', '1'], poll_seconds=.01))
            status = json.loads((root / 'queue_status.json').read_text())
            self.assertEqual([s['status'] for s in status], ['complete', 'failed', 'complete', 'complete'])
            workers = [json.loads((root / f'job{i}/worker.json').read_text()) for i in range(4)]
            self.assertLess(max(w['start'] for w in workers[:2]), min(w['end'] for w in workers[:2]))
            for gpu in ['0', '1']:
                ordered = sorted([w for w in workers if w['gpu'] == gpu], key=lambda w: w['start'])
                for previous, following in zip(ordered, ordered[1:]):
                    self.assertLessEqual(previous['end'], following['start'])
            first_start = workers[0]['start']
            jobs[1]['command'][-1] = 'no'
            self.assertTrue(run_queue(plan, root, ['0', '1'], resume=True, poll_seconds=.01))
            self.assertEqual(json.loads((root / 'job0/worker.json').read_text())['start'], first_start)
            self.assertEqual(len((root / 'summary.csv').read_text().splitlines()), 5)


if __name__ == '__main__':
    unittest.main()
