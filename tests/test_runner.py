"""Tests for Runner execution, callback progress, and error handling."""

from exetasiz.core.models import Dataset, TestCase
from exetasiz.evaluators.exact_match import ExactMatchEvaluator
from exetasiz.providers.mock import MockProvider
from exetasiz.runner.runner import Runner


def test_runner_happy_path():
    tc1 = TestCase(id="t1", category="general", prompt="What is 1+1?", expected_answer="2")
    tc2 = TestCase(id="t2", category="general", prompt="Capital of France?", expected_answer="Paris")
    dataset = Dataset(name="test_ds", test_cases=[tc1, tc2])

    p1 = MockProvider(name="m1", responses={"What is 1+1?": "2", "Capital of France?": "Paris"})
    p2 = MockProvider(name="m2", responses={"What is 1+1?": "2", "Capital of France?": "London"})

    evaluator = ExactMatchEvaluator()
    runner = Runner(dataset=dataset, providers=[p1, p2], evaluators=[evaluator], experiment_name="demo")

    progress_events = []
    def on_step(curr, total, model, cid):
        progress_events.append((curr, total, model, cid))

    run_result = runner.run(progress_callback=on_step)

    assert run_result.experiment_name == "demo"
    assert len(run_result.results) == 4  # 2 models * 2 test cases * 1 evaluator
    assert len(progress_events) == 4

    # Check model 1 results
    m1_results = [r for r in run_result.results if r.model == "m1"]
    assert all(r.passed for r in m1_results)

    # Check model 2 results
    m2_results = [r for r in run_result.results if r.model == "m2"]
    assert sum(r.passed for r in m2_results) == 1


def test_runner_provider_error_does_not_halt_run():
    tc1 = TestCase(id="t1", category="c", prompt="p1", expected_answer="a1")
    tc2 = TestCase(id="t2", category="c", prompt="p2", expected_answer="a2")
    dataset = Dataset(name="ds", test_cases=[tc1, tc2])

    failing_provider = MockProvider(name="failing_model", simulate_error=True, error_message="API Quota Exceeded")
    working_provider = MockProvider(name="good_model", responses={"p1": "a1", "p2": "a2"})

    runner = Runner(
        dataset=dataset,
        providers=[failing_provider, working_provider],
        evaluators=[ExactMatchEvaluator()],
    )

    run_result = runner.run()
    assert len(run_result.results) == 4

    failed_results = [r for r in run_result.results if r.model == "failing_model"]
    assert len(failed_results) == 2
    for r in failed_results:
        assert r.passed is False
        assert r.score == 0.0
        assert r.error is not None
        assert "API Quota Exceeded" in r.error

    good_results = [r for r in run_result.results if r.model == "good_model"]
    assert len(good_results) == 2
    for r in good_results:
        assert r.passed is True
        assert r.error is None
