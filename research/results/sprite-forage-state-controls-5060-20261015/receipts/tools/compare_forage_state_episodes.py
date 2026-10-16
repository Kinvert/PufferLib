"""Offline paired episode effects and repeat checks for the state pilot."""
import json
from pathlib import Path
import statistics

from forage_state_ablations import WORLD_KEYS, rows
from run_forage_attention_budget import ROOT, sha

CASES = ROOT / 'build/sprite-forage-state-ablations-5060-20261015'
HOT = ROOT / 'build/sprite-forage-state-onehot-5060-20261015'
REL = ROOT / 'build/sprite-forage-state-relative-5060-20261015'
REPEAT = ROOT / 'build/sprite-forage-state-relative-repeat-5060-20261015'
OUT = ROOT / 'research/results/sprite-forage-state-controls-5060-20261015'


def load(packet, name):
    assert (packet / 'full/COMPLETE').is_file()
    evaluations = json.loads((packet / 'full/evaluations.json').read_text())
    data = {}
    for ev in evaluations:
        if ev['name'] != name:
            continue
        assert sha(Path(ev['csv_path'])) == ev['csv_sha256']
        for row in rows(Path(ev['csv_path'])):
            key = (ev['world'], int(row['episode_id']))
            assert key not in data
            data[key] = row
    assert len(data) == 256
    return data


def compare(a_packet, a_name, b_packet, b_name):
    a, b = load(a_packet, a_name), load(b_packet, b_name)
    assert a.keys() == b.keys()
    diffs = []
    for key in sorted(a):
        assert all(a[key][k] == b[key][k] for k in WORLD_KEYS)
        diffs.append(float.fromhex(b[key]['perf']) - float.fromhex(a[key]['perf']))
    return dict(reference=a_name, candidate=b_name, episodes=256,
        collection_difference_points=100 * statistics.fmean(diffs),
        candidate_higher=sum(d > 0 for d in diffs), tied=sum(d == 0 for d in diffs),
        candidate_lower=sum(d < 0 for d in diffs),
        scope='Paired evaluation worlds for one trained seed; not independent training evidence')


if __name__ == '__main__':
    OUT.mkdir(exist_ok=True)
    comparisons = [
        compare(CASES, 'quiet-otter-1', HOT, 'gentle-maple-1'),
        compare(CASES, 'happy-heron-3', HOT, 'mystic-sparrow-2'),
        compare(CASES, 'gentle-fern-5', HOT, 'happy-fern-3'),
        compare(HOT, 'happy-fern-3', REL, 'mossy-robin-4'),
        compare(CASES, 'quiet-otter-1', REL, 'mossy-robin-4')]
    result = dict(comparisons=comparisons, fixed_training_seed=7, independent_training_n=1,
        selected_on_development_data=True, statistical_or_pareto_claim=False)
    if (REPEAT / 'full/COMPLETE').is_file():
        a = json.loads((REL / 'full/training.json').read_text())[0]
        b = json.loads((REPEAT / 'full/training.json').read_text())[0]
        original = {Path(p).name: digest for p, digest in a['checkpoints'].items()}
        repeated = {Path(p).name: digest for p, digest in b['checkpoints'].items()}
        assert len(original) == 8 and original == repeated
        assert load(REL, 'mossy-robin-4') == load(REPEAT, 'mossy-robin-4')
        result['repeat'] = dict(all_eight_checkpoint_hashes_identical=True,
            all_256_episode_rows_identical=True, training_seed=7, independent_training_n=1)
    (OUT / 'paired.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))
