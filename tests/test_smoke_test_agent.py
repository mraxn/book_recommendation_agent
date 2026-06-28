from scripts.smoke_test_agent import SMOKE_SCENARIOS


def test_smoke_suite_contains_ten_scenarios() -> None:
    assert len(SMOKE_SCENARIOS) == 10


def test_smoke_suite_contains_three_follow_ups() -> None:
    assert sum(1 for scenario in SMOKE_SCENARIOS if scenario.follow_up) == 3


def test_smoke_suite_names_are_unique() -> None:
    names = [scenario.name for scenario in SMOKE_SCENARIOS]

    assert len(names) == len(set(names))


def test_smoke_suite_prompts_are_complicated() -> None:
    assert all(len(scenario.prompt.split()) >= 10 for scenario in SMOKE_SCENARIOS)
