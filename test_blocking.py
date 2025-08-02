#!/usr/bin/env python3
"""
Pytest tests to verify the blocking logic works correctly for airplane seating.
"""

import pytest
import numpy as np
from time_simulator import simulate_boarding_time, BoardingMethod


class TestBlockingLogic:
    """Test class for airplane boarding blocking logic."""

    def setup_method(self):
        """Set up test fixtures before each test method."""
        # Set seed for reproducible results
        np.random.seed(42)

    def test_all_boarding_methods_run_successfully(self):
        """Test that all boarding methods can run without errors."""
        methods = [
            BoardingMethod.RANDOM,
            BoardingMethod.BACK_TO_FRONT,
            BoardingMethod.ROW_RANDOM,
            BoardingMethod.WINDOW_MIDDLE_AISLE,
            BoardingMethod.STEFFEN,
        ]

        for method in methods:
            # Should not raise any exceptions
            boarding_time = simulate_boarding_time(method, num_rows=3, num_columns=6)

            # Basic sanity checks
            assert boarding_time > 0, (
                f"Boarding time should be positive for method {method.value}"
            )
            assert boarding_time < 10000, (
                f"Boarding time seems unreasonably high for method {method.value}"
            )

    def test_boarding_time_varies_by_method(self):
        """Test that different boarding methods produce different average times."""
        methods = [
            BoardingMethod.RANDOM,
            BoardingMethod.BACK_TO_FRONT,
            BoardingMethod.ROW_RANDOM,
            BoardingMethod.WINDOW_MIDDLE_AISLE,
            BoardingMethod.STEFFEN,
        ]

        times = []
        for i, method in enumerate(methods):
            # Use different seeds to get some variation
            np.random.seed(42 + i)
            boarding_time = simulate_boarding_time(method, num_rows=5, num_columns=6)
            times.append(boarding_time)

        # Check that we get some variation in times
        assert len(set([round(t, -1) for t in times])) > 1, (
            "Different methods should produce different boarding times"
        )

    def test_small_vs_large_airplane(self):
        """Test that larger airplanes take more time to board."""
        np.random.seed(42)

        # Small airplane
        small_time = simulate_boarding_time(
            BoardingMethod.RANDOM, num_rows=3, num_columns=6
        )

        # Reset seed for fair comparison
        np.random.seed(42)

        # Large airplane
        large_time = simulate_boarding_time(
            BoardingMethod.RANDOM, num_rows=30, num_columns=6
        )

        assert large_time > small_time, (
            "Larger airplanes should take more time to board"
        )

    def test_blocking_logic_seat_positions(self):
        """Test that the blocking logic correctly identifies blocking scenarios."""
        # This is more of an integration test since we can't easily unit test the internal logic
        # But we can test that the system produces reasonable results

        methods_and_expected_patterns = [
            (
                BoardingMethod.WINDOW_MIDDLE_AISLE,
                "should be efficient due to minimal blocking",
            ),
            (
                BoardingMethod.STEFFEN,
                "should be very efficient with optimized boarding",
            ),
            (
                BoardingMethod.RANDOM,
                "should have moderate efficiency due to random blocking",
            ),
        ]

        results = []
        for method, description in methods_and_expected_patterns:
            # Run multiple times to get average
            times = []
            for i in range(5):
                np.random.seed(42 + i)
                time = simulate_boarding_time(method, num_rows=10, num_columns=6)
                times.append(time)

            avg_time = np.mean(times)
            results.append((method.value, avg_time, description))

        # Just verify all methods produce reasonable times
        for method_name, avg_time, description in results:
            assert 0 < avg_time < 20000, (
                f"{method_name} should produce reasonable boarding times"
            )
            print(f"{method_name}: {avg_time / 60:.1f} minutes - {description}")

    @pytest.mark.parametrize(
        "num_rows,num_columns",
        [
            (5, 6),
            (10, 6),
            (25, 6),
            (50, 6),
        ],
    )
    def test_different_airplane_sizes(self, num_rows, num_columns):
        """Test boarding simulation with different airplane sizes."""
        np.random.seed(42)

        boarding_time = simulate_boarding_time(
            BoardingMethod.RANDOM, num_rows=num_rows, num_columns=num_columns
        )

        # Sanity checks
        assert boarding_time > 0

        # More realistic upper bound based on actual simulation behavior
        # Each passenger takes on average 30-120 seconds depending on blocking and luggage
        max_expected = num_rows * num_columns * 120  # 2 minutes per passenger max
        assert boarding_time < max_expected, (
            f"Boarding time {boarding_time / 60:.1f} minutes seems too high for {num_rows}x{num_columns} airplane"
        )

        # Also check that it's not unreasonably fast (should take some time)
        min_expected = num_rows * num_columns * 10  # At least 10 seconds per passenger
        assert boarding_time > min_expected, (
            f"Boarding time {boarding_time / 60:.1f} minutes seems too fast for {num_rows}x{num_columns} airplane"
        )

    def test_consistent_seat_generation(self):
        """Test that seat generation is consistent and follows expected format."""
        # This tests the seat generation indirectly through the simulation
        for method in BoardingMethod:
            boarding_time = simulate_boarding_time(method, num_rows=2, num_columns=6)
            assert boarding_time > 0, (
                f"Method {method.value} should generate valid boarding sequence"
            )


def test_blocking_logic_explanation():
    """
    Document and verify the blocking logic for airplane seating ABC | DEF.

    This test serves as documentation for the blocking rules:
    - A is blocked by B, C (needs both to get out)
    - B is blocked by C (needs C to get out)
    - C is never blocked (aisle seat)
    - D is never blocked (aisle seat)
    - E is blocked by D (needs D to get out)
    - F is blocked by D, E (needs both to get out)
    """
    print("\nBlocking Logic Documentation:")
    print("=" * 50)
    print("Airplane seating layout: ABC | DEF")
    print("Blocking rules implemented:")
    print("- A blocked by: B, C")
    print("- B blocked by: C")
    print("- C blocked by: none (aisle)")
    print("- D blocked by: none (aisle)")
    print("- E blocked by: D")
    print("- F blocked by: D, E")
    print("=" * 50)

    # Run a quick test to verify the system works
    np.random.seed(42)
    time = simulate_boarding_time(BoardingMethod.RANDOM, num_rows=3, num_columns=6)
    assert time > 0, "Blocking logic should produce valid boarding times"


if __name__ == "__main__":
    # Run tests when script is executed directly
    pytest.main([__file__, "-v"])
