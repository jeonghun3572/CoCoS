import contextlib
import faulthandler
import io
import multiprocessing
import os
import platform
import signal
import tempfile
import itertools
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np

import evaluate


def check_correctness(check_program, timeout, task_id, completion_id):
    """
    Evaluates the functional correctness of a completion by running the test
    suite provided in the problem.

    :param completion_id: an optional completion ID so we can match
        the results later even if execution finishes asynchronously.
    """
    manager = multiprocessing.Manager()
    result = manager.list()

    p = multiprocessing.Process(target=unsafe_execute, args=(check_program, result, timeout))
    p.start()
    p.join(timeout=timeout + 1)
    if p.is_alive():
        p.kill()

    if not result:
        result.append("timed out")

    return dict(
        task_id=task_id,
        passed=result[0] == "passed",
        result=result[0],
        completion_id=completion_id,
    )


def unsafe_execute(check_program, result, timeout):

    with create_tempdir():

        # These system calls are needed when cleaning up tempdir.
        import os
        import shutil

        rmtree = shutil.rmtree
        rmdir = os.rmdir
        chdir = os.chdir

        # Disable functionalities that can make destructive changes to the test.
        reliability_guard()

        # Run program.
        try:
            exec_globals = {}
            with swallow_io():
                with time_limit(timeout):
                    exec(check_program, exec_globals)
            result.append("passed")
        except TimeoutException:
            result.append("timed out")
        except BaseException as e:
            result.append(f"failed: {e}")

        # Needed for cleaning up.
        shutil.rmtree = rmtree
        os.rmdir = rmdir
        os.chdir = chdir


@contextlib.contextmanager
def time_limit(seconds):
    def signal_handler(signum, frame):
        raise TimeoutException("Timed out!")

    signal.setitimer(signal.ITIMER_REAL, seconds)
    signal.signal(signal.SIGALRM, signal_handler)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)


@contextlib.contextmanager
def swallow_io():
    stream = WriteOnlyStringIO()
    with contextlib.redirect_stdout(stream):
        with contextlib.redirect_stderr(stream):
            with redirect_stdin(stream):
                yield


@contextlib.contextmanager
def create_tempdir():
    with tempfile.TemporaryDirectory() as dirname:
        with chdir(dirname):
            yield dirname


class TimeoutException(Exception):
    pass


class WriteOnlyStringIO(io.StringIO):
    """StringIO that throws an exception when it's read from"""

    def read(self, *args, **kwargs):
        raise OSError

    def readline(self, *args, **kwargs):
        raise OSError

    def readlines(self, *args, **kwargs):
        raise OSError

    def readable(self, *args, **kwargs):
        """Returns True if the IO object can be read."""
        return False


class redirect_stdin(contextlib._RedirectStream):  # type: ignore
    _stream = "stdin"


@contextlib.contextmanager
def chdir(root):
    if root == ".":
        yield
        return
    cwd = os.getcwd()
    os.chdir(root)
    try:
        yield
    except BaseException as exc:
        raise exc
    finally:
        os.chdir(cwd)


def reliability_guard(maximum_memory_bytes=None):
    """
    This disables various destructive functions and prevents the generated code
    from interfering with the test (e.g. fork bomb, killing other processes,
    removing filesystem files, etc.)

    WARNING
    This function is NOT a security sandbox. Untrusted code, including, model-
    generated code, should not be blindly executed outside of one. See the
    Codex paper for more information about OpenAI's code sandbox, and proceed
    with caution.
    """

    if maximum_memory_bytes is not None:
        import resource

        resource.setrlimit(resource.RLIMIT_AS, (maximum_memory_bytes, maximum_memory_bytes))
        resource.setrlimit(resource.RLIMIT_DATA, (maximum_memory_bytes, maximum_memory_bytes))
        if not platform.uname().system == "Darwin":
            resource.setrlimit(resource.RLIMIT_STACK, (maximum_memory_bytes, maximum_memory_bytes))

    faulthandler.disable()

    import builtins

    builtins.exit = None
    builtins.quit = None

    import os

    os.environ["OMP_NUM_THREADS"] = "1"

    os.kill = None
    os.system = None
    os.putenv = None
    os.remove = None
    os.removedirs = None
    os.rmdir = None
    os.fchdir = None
    os.setuid = None
    os.fork = None
    os.forkpty = None
    os.killpg = None
    os.rename = None
    os.renames = None
    os.truncate = None
    os.replace = None
    os.unlink = None
    os.fchmod = None
    os.fchown = None
    os.chmod = None
    os.chown = None
    os.chroot = None
    os.fchdir = None
    os.lchflags = None
    os.lchmod = None
    os.lchown = None
    os.getcwd = None
    os.chdir = None

    import shutil

    shutil.rmtree = None
    shutil.move = None
    shutil.chown = None

    import subprocess

    subprocess.Popen = None  # type: ignore

    __builtins__["help"] = None

    import sys

    sys.modules["ipdb"] = None
    sys.modules["joblib"] = None
    sys.modules["resource"] = None
    sys.modules["psutil"] = None
    sys.modules["tkinter"] = None

_CITATION = """\
@misc{chen2021evaluating,
      title={Evaluating Large Language Models Trained on Code},
      author={Mark Chen and Jerry Tworek and Heewoo Jun and Qiming Yuan \
and Henrique Ponde de Oliveira Pinto and Jared Kaplan and Harri Edwards \
and Yuri Burda and Nicholas Joseph and Greg Brockman and Alex Ray \
and Raul Puri and Gretchen Krueger and Michael Petrov and Heidy Khlaaf \
and Girish Sastry and Pamela Mishkin and Brooke Chan and Scott Gray \
and Nick Ryder and Mikhail Pavlov and Alethea Power and Lukasz Kaiser \
and Mohammad Bavarian and Clemens Winter and Philippe Tillet \
and Felipe Petroski Such and Dave Cummings and Matthias Plappert \
and Fotios Chantzis and Elizabeth Barnes and Ariel Herbert-Voss \
and William Hebgen Guss and Alex Nichol and Alex Paino and Nikolas Tezak \
and Jie Tang and Igor Babuschkin and Suchir Balaji and Shantanu Jain \
and William Saunders and Christopher Hesse and Andrew N. Carr \
and Jan Leike and Josh Achiam and Vedant Misra and Evan Morikawa \
and Alec Radford and Matthew Knight and Miles Brundage and Mira Murati \
and Katie Mayer and Peter Welinder and Bob McGrew and Dario Amodei \
and Sam McCandlish and Ilya Sutskever and Wojciech Zaremba},
      year={2021},
      eprint={2107.03374},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
"""

_DESCRIPTION = """\
This metric implements the evaluation harness for the HumanEval problem solving dataset
described in the paper "Evaluating Large Language Models Trained on Code"
(https://arxiv.org/abs/2107.03374).
"""


_KWARGS_DESCRIPTION = """
Calculates how good are predictions given some references, using certain scores
Args:
    predictions: list of candidates to evaluate. Each candidates should be a list
        of strings with several code candidates to solve the problem.
    references: a list with a test for each prediction. Each test should evaluate the
        correctness of a code candidate.
    k: number of code candidates to consider in the evaluation (Default: [1, 10, 100])
    num_workers: number of workers used to evaluate the canidate programs (Default: 4).
    timeout:
Returns:
    pass_at_k: dict with pass rates for each k
    results: dict with granular results of each unittest
Examples:
    >>> code_eval = evaluate.load("code_eval")
    >>> test_cases = ["assert add(2,3)==5"]
    >>> candidates = [["def add(a,b): return a*b", "def add(a, b): return a+b"]]
    >>> pass_at_k, results = code_eval.compute(references=test_cases, predictions=candidates, k=[1, 2])
    >>> print(pass_at_k)
    {'pass@1': 0.5, 'pass@2': 1.0}
"""


_WARNING = """
################################################################################
                                  !!!WARNING!!!
################################################################################
The "code_eval" metric executes untrusted model-generated code in Python.
Although it is highly unlikely that model-generated code will do something
overtly malicious in response to this test suite, model-generated code may act
destructively due to a lack of model capability or alignment.
Users are strongly encouraged to sandbox this evaluation suite so that it
does not perform destructive actions on their host or network. For more
information on how OpenAI sandboxes its code, see the paper "Evaluating Large
Language Models Trained on Code" (https://arxiv.org/abs/2107.03374).

Once you have read this disclaimer and taken appropriate precautions,
set the environment variable HF_ALLOW_CODE_EVAL="1". Within Python you can to this
with:

>>> import os
>>> os.environ["HF_ALLOW_CODE_EVAL"] = "1"

################################################################################\
"""

_LICENSE = """The MIT License

Copyright (c) OpenAI (https://openai.com)

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE."""


@evaluate.utils.file_utils.add_start_docstrings(_DESCRIPTION, _KWARGS_DESCRIPTION)
class CodeEval(evaluate.Metric):
    def compute(predictions, references, k=[1, 10, 100], num_workers=4, timeout=3.0):
        """Returns the scores"""

        if os.getenv("HF_ALLOW_CODE_EVAL", 0) != "1":
            raise ValueError(_WARNING)

        if os.name == "nt":
            raise NotImplementedError("This metric is currently not supported on Windows.")

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = []
            completion_id = Counter()
            n_samples = 0
            results = defaultdict(list)

            for task_id, (candidates, test_case) in enumerate(zip(predictions, references)):
                for candidate in candidates:
                    test_program = candidate + "\n" + test_case
                    args = (test_program, timeout, task_id, completion_id[task_id])
                    future = executor.submit(check_correctness, *args)
                    futures.append(future)
                    completion_id[task_id] += 1
                    n_samples += 1

            for future in as_completed(futures):
                result = future.result()
                results[result["task_id"]].append((result["completion_id"], result))

        total, correct = [], []
        for result in results.values():
            result.sort()
            passed = [r[1]["passed"] for r in result]
            total.append(len(passed))
            correct.append(sum(passed))
        total = np.array(total)
        correct = np.array(correct)

        ks = k
        pass_at_k = {f"pass@{k}": estimate_pass_at_k(total, correct, k).mean() for k in ks if (total >= k).all()}

        return pass_at_k, results


def estimate_pass_at_k(num_samples, num_correct, k):
    """Estimates pass@k of each problem and returns them in an array."""

    def estimator(n: int, c: int, k: int) -> float:
        """Calculates 1 - comb(n - c, k) / comb(n, k)."""
        if n - c < k:
            return 1.0
        return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))

    if isinstance(num_samples, int):
        num_samples_it = itertools.repeat(num_samples, len(num_correct))
    else:
        assert len(num_samples) == len(num_correct)
        num_samples_it = iter(num_samples)

    return np.array([estimator(int(n), int(c), k) for n, c in zip(num_samples_it, num_correct)])