import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock
import numpy as np
import pandas as pd
from pressure_pipeline import Database, analyze, clean_readings, demo_readings, hourly_window, llm_summary, main
from leak_toolkit_working import features, greedy_placement, leak_shape_from_cube

class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.clean, _, _ = clean_readings(demo_readings(90))

    def test_known_signals(self):
        rows = {r['sensor_id']: r for r in analyze(self.clean)}
        self.assertEqual(rows['synthetic_decline']['status'], 'possible_leak_indicator')
        self.assertEqual(rows['synthetic_stable']['status'], 'no_sustained_decline_indicator')
        self.assertEqual(rows['synthetic_valve_step']['status'], 'abrupt_change_review')

    def test_validation_duplicates_and_timezone(self):
        raw = pd.DataFrame([
            ['2026-01-01T00:00:00Z', 'a', 50], ['2025-12-31T18:00:00-06:00', 'a', 54],
            ['bad', 'a', 50], ['2026-01-02', None, 60], ['2026-01-02', 'a', float('inf')],
            ['2026-01-02', 'a', -1], ['2026-01-02', 'a', 'bad'],
        ], columns=['ts', 'sensor_id', 'pressure_psi'])
        clean, rejected, audit = clean_readings(raw)
        self.assertEqual(len(rejected), 5)
        self.assertEqual(audit['duplicate_rows_averaged'], 1)
        self.assertEqual(clean.pressure_psi.iloc[0], 52)

    def test_gaps_and_staleness(self):
        group = self.clean[self.clean.sensor_id == 'synthetic_decline'].reset_index(drop=True)
        small, quality = hourly_window(group.drop(index=[100, 101]), days=90)
        self.assertFalse(small.isna().any())
        self.assertEqual(quality['filled_hours'], 2)
        large, quality = hourly_window(group.drop(index=range(100, 110)), days=90)
        self.assertEqual(int(large.isna().sum()), 10)
        stale = self.clean.drop(self.clean[(self.clean.sensor_id == 'synthetic_decline') & (self.clean.ts > self.clean.ts.max() - pd.Timedelta(days=2))].index)
        result = {r['sensor_id']: r for r in analyze(stale)}
        self.assertEqual(result['synthetic_decline']['status'], 'insufficient_data')

    def test_short_data_and_future_end(self):
        rows = analyze(self.clean[self.clean.ts < self.clean.ts.min() + pd.Timedelta(days=7)])
        self.assertTrue(all(r['status'] == 'insufficient_data' for r in rows))
        rows = analyze(self.clean, end=self.clean.ts.max() + pd.Timedelta(days=2))
        self.assertTrue(all(r['status'] == 'insufficient_data' for r in rows))

    def test_database_and_identifiers(self):
        for target in [':memory:', 'sqlite:///:memory:']:
            with Database(target) as db:
                db.check()
                db.write(demo_readings(14), 'pressure_readings')
                self.assertEqual(len(db.read('pressure_readings')), 14 * 24 * 3)
                with self.assertRaises(ValueError):
                    db.read('pressure_readings; DROP TABLE pressure_readings')

    def test_llm_success_failure_and_disabled(self):
        client = Mock()
        client.responses.create.return_value = SimpleNamespace(status='completed', output_text='Review evidence.', id='test')
        result = llm_summary({'sensors': []}, 'openai', client, 'test-model')
        self.assertEqual(result['status'], 'completed')
        self.assertFalse(client.responses.create.call_args.kwargs['store'])
        client.responses.create.side_effect = TimeoutError('secret should not appear')
        result = llm_summary({}, 'openai', client, 'test-model')
        self.assertEqual(result['status'], 'error')
        self.assertNotIn('secret', result['text'])
        self.assertEqual(llm_summary({})['status'], 'disabled')

    def test_end_to_end_and_append_only_results(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = str(Path(tmp) / 'demo.sqlite')
            out = str(Path(tmp) / 'results')
            self.assertEqual(main(['demo', '--db', db, '--out', out, '--llm', 'off']), 0)
            report = json.loads((Path(out) / 'report.json').read_text())
            self.assertEqual(report['source'], 'synthetic_demo')
            self.assertEqual(len(report['sensors']), 3)
            self.assertEqual(main(['analyze', '--db', db, '--out', out, '--llm', 'off']), 0)
            with Database(db) as connection:
                self.assertEqual(connection.conn.execute('SELECT COUNT(*) FROM pressure_analysis_results').fetchone()[0], 6)
            with self.assertRaises(ValueError):
                main(['demo', '--db', db, '--out', out])

    def test_all_invalid_sensor_is_visible(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = str(Path(tmp) / 'bad.sqlite')
            with Database(db) as connection:
                connection.write(pd.DataFrame([{'ts': 'bad', 'sensor_id': 'broken', 'pressure_psi': -9}]), 'pressure_readings')
            main(['analyze', '--db', db, '--out', tmp, '--llm', 'off'])
            report = json.loads((Path(tmp) / 'report.json').read_text())
            self.assertEqual(report['sensors'][0]['status'], 'insufficient_data')

    def test_features_are_finite(self):
        self.assertTrue(np.isfinite(features(np.full(24*30, 60.0), 30)).all())
        with self.assertRaises(ValueError):
            features([1, 2], 30)

    def test_placement_does_not_use_held_out_rows(self):
        first = pd.DataFrame({'old': [np.inf, np.inf], 'a': [1, np.inf], 'b': [np.inf, 0]}, index=['train', 'test'])
        self.assertEqual(greedy_placement(first, ['train'], ['old'], ['a', 'b'], 1, 12), ['a'])
        first.loc['test', 'a'] = 0
        self.assertEqual(greedy_placement(first, ['train'], ['old'], ['a', 'b'], 1, 12), ['a'])
        curve, _, _ = leak_shape_from_cube({'train': pd.DataFrame({'a': [-1, -2]})}, first.loc[['train']], ['a'])
        self.assertEqual(curve.tolist(), [0.5, 1.0])

if __name__ == '__main__':
    unittest.main()
