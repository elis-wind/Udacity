# Unit testing tools

To install `pytest`, run `pip install -U pytest` in your terminal. You can see more information on getting started [here](https://docs.pytest.org/en/latest/getting-started.html).

1. Create a test file starting with `test_`.
2. Define unit test functions that start with `test_` inside the test file.
3. Enter `pytest` into your terminal in the directory of your test file and it detects these tests for you.

`test_` is the default; if you wish to change this, you can learn how with this `pytest` [Examples and Customizations](https://docs.pytest.org/en/latest/example/index.html?highlight=customize) link.

In the test output, periods represent successful unit tests and Fs represent failed unit tests. Since all you see is which test functions failed, it's wise to have only one `assert` statement per test. Otherwise, you won't know exactly how many tests failed or which tests failed.

Your test won't be stopped by failed `assert` statements, but it will stop if you have syntax errors.
