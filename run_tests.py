import unittest
import argparse

def main(include_benchmarks: bool):
    # Create a test loader
    loader = unittest.TestLoader()
    
    # Always discover tests in the 'tests' directory
    combined_suite = unittest.TestSuite()
    test_suite = loader.discover(start_dir='tests', pattern='test_*.py')
    combined_suite.addTests(test_suite)
    
    # Optionally discover benchmarks
    if include_benchmarks:
        benchmark_suite = loader.discover(start_dir='tests', pattern='benchmark_*.py')
        combined_suite.addTests(benchmark_suite)

    # Create a test runner and run the combined suite
    runner = unittest.TextTestRunner()
    runner.run(combined_suite)

if __name__ == "__main__":
    # Set up command line argument parsing
    parser = argparse.ArgumentParser(description='Run tests and optionally benchmarks.')
    parser.add_argument('--no-benchmarks', action='store_false', dest='include_benchmarks',
                        help='Exclude benchmarks from the test run')

    args = parser.parse_args()

    # Run the main function with or without benchmarks based on the command line argument
    main(args.include_benchmarks)
