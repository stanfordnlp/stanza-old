__author__ = 'victor, wmonroe4, kelvinguu'

from setuptools import setup, Command, find_packages


class UnitTest2(Command):
    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        import subprocess
        errno = subprocess.call(['python2', '-m', 'pytest', '--doctest-modules',
                                 '--ignore=stanza/research/pick_gpu.py',
                                 '--ignore=stanza/research/progress.py',
                                 '--ignore=stanza/research/summary.py',
                                 '--ignore=stanza/research/templates/third-party',
                                 'stanza', 'test/unit_tests'])
        raise SystemExit(errno)

class UnitTest3(Command):
    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        import subprocess
        errno = subprocess.call(['python3', '-m', 'pytest', '--doctest-modules',
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
    version='0.3',
    packages=find_packages(exclude=['docs', 'test']),
    url='https://github.com/stanfordnlp/stanza',
    license='MIT',
    author='Stanford NLP',
    author_email='victor@victorzhong.com',
    description='NLP library for Python',
    cmdclass={'test': UnitTest2, 'test3' : UnitTest3, 'slow_test': SlowTest, 'all_test': AllTest},
    download_url='https://github.com/stanfordnlp/stanza/tarball/0.1',
    keywords=['nlp', 'neural networks', 'machine learning'],
)
