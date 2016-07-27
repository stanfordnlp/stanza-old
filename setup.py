__author__ = 'victor, wmonroe4, kelvinguu'

from distutils.core import setup, Command


class UnitTest(Command):
    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        import subprocess
        errno = subprocess.call(['py.test', '--doctest-modules',
                                 '--ignore=stanza/research/pick_gpu.py',
                                 '--ignore=stanza/research/progress.py',
                                 '--ignore=stanza/research/summary.py',
                                 '--ignore=stanza/research/templates/third-party',
                                 'stanza', 'test/unit_tests'])
        raise SystemExit(errno)


class SlowTest(Command):
    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        import subprocess
        errno = subprocess.call(['py.test', '--doctest-modules', 'test/slow_tests'])
        raise SystemExit(errno)


class AllTest(Command):
    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        import subprocess
        errno = subprocess.call(['py.test', '--doctest-modules'])
        raise SystemExit(errno)


setup(
    name='stanza',
    version='0.1',
    packages=['stanza', 'stanza.text', 'stanza.monitoring', 'stanza.util'],
    url='https://github.com/stanfordnlp/stanza',
    license='MIT',
    author='Stanford NLP',
    author_email='victor@victorzhong.com',
    description='NLP library for Python',
    cmdclass={'test': UnitTest, 'slow_test': SlowTest, 'all_test': AllTest},
    download_url='https://github.com/stanfordnlp/stanza/tarball/0.1',
    keywords=['nlp', 'neural networks', 'machine learning'],
)
