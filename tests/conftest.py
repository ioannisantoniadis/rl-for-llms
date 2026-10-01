import numpy as np
import pytest

from rl4llm.toy_language import default_testbed, sequence_logprobs


@pytest.fixture(scope="session")
def testbed():
    return default_testbed()


@pytest.fixture(scope="session")
def lang(testbed):
    return testbed[0]


@pytest.fixture(scope="session")
def ref_logits(testbed):
    return testbed[1]


@pytest.fixture(scope="session")
def ref_logp(lang, ref_logits):
    return sequence_logprobs(lang, ref_logits)


@pytest.fixture(scope="session")
def reward(testbed):
    return testbed[2]


@pytest.fixture
def rng():
    return np.random.default_rng(1234)
