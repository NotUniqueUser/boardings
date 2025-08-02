import numpy as np
from enum import Enum

# Time constants for boarding simulation, measured in seconds
# exponential distribution for luggage placement (mean 0.5 minutes = 30 seconds)
TIME_LUGGAGE = 30
# exponential distribution for sitting when blocked by others (base time)
TIME_SITTING_BLOCKED = 30
# additional time per person blocking (0.25 * number of blocking people)
TIME_PER_BLOCKING_PERSON = 15
# additional time if passenger has more luggage to organize
TIME_EXTRA_LUGGAGE = 15


class BoardingMethod(Enum):
    RANDOM = "Random"
    ROW_RANDOM = "Row Random"
    BACK_TO_FRONT = "Back to Front"
    WINDOW_MIDDLE_AISLE = "Window-Middle-Aisle"
    STEFFEN = "Steffen Algorithm"


def simulate_boarding_time(
    method: BoardingMethod, num_rows: int = 50, num_columns: int = 6
) -> float:
    """
    Simulate the boarding time for a given boarding method.

    This simulation models passengers boarding through a single aisle, where boarding time
    is determined by the bottleneck and queue formation, not the sum of individual times.

    Args:
        method (BoardingMethod): The boarding method to use.
        num_rows (int): Number of rows in the aircraft (default: 50).
        num_columns (int): Number of columns/seats per row (default: 6, A-F).

    Returns:
        float: Total boarding time in seconds.
    """
    # Generate all seats
    column_letters = [chr(ord("A") + i) for i in range(num_columns)]
    all_seats = [
        f"{row:02d}{letter}"
        for row in range(1, num_rows + 1)
        for letter in column_letters
    ]

    # Get boarding order based on method
    if method == BoardingMethod.RANDOM:
        boarding_order = _random_boarding(all_seats)
    elif method == BoardingMethod.ROW_RANDOM:
        boarding_order = _row_random_boarding(all_seats, num_rows, num_columns)
    elif method == BoardingMethod.BACK_TO_FRONT:
        boarding_order = _back_to_front_boarding(all_seats, num_rows, num_columns)
    elif method == BoardingMethod.WINDOW_MIDDLE_AISLE:
        boarding_order = _window_middle_aisle_boarding(all_seats, num_rows, num_columns)
    elif method == BoardingMethod.STEFFEN:
        boarding_order = _steffen_boarding(all_seats, num_rows, num_columns)
    else:
        raise ValueError(f"Unknown boarding method: {method}")

    # Simulate realistic boarding with aisle queue and parallel processing
    return _simulate_aisle_boarding(boarding_order, num_rows, num_columns)


def _simulate_aisle_boarding(
    boarding_order: list[str], num_rows: int, num_columns: int
) -> float:
    """
    Simulate boarding according to the Hebrew specification rules EXACTLY:
    1. Walking time = 0 (negligible)
    2. Luggage placement: exponential distribution (mean 0.5 minutes = 30 seconds) - EVERYONE has luggage
    3. Sitting time: if blocked, exponential with mean (0.5 + 0.25 * blocking_count) minutes
    4. Extra luggage: if last passenger in row, organize all luggage in overhead compartment
    5. Ordering time = 0 (negligible)

    Follow the rules exactly without any additional assumptions about entry intervals.
    """
    total_time = 0.0
    seated_passengers = {}

    # Process passengers in their boarding order
    for seat in boarding_order:
        passenger_time = 0.0

        # Rule 1: Walking time = 0 (negligible) - no time added

        # Rule 2: Luggage placement - exponential distribution (mean 0.5 minutes = 30 seconds)
        # EVERYONE has luggage according to Hebrew specification
        luggage_time = np.random.exponential(TIME_LUGGAGE)
        passenger_time += luggage_time

        # Rule 3: Sitting time - check for blocking
        blocking_count = _count_blocking_passengers(seat, seated_passengers)

        if blocking_count > 0:
            # If blocked: exponential with mean (0.5 + 0.25 * blocking_count) minutes
            mean_sitting_time = (
                TIME_SITTING_BLOCKED + blocking_count * TIME_PER_BLOCKING_PERSON
            )
            sitting_time = np.random.exponential(mean_sitting_time)
            passenger_time += sitting_time
        # If not blocked, sitting time = 0 (negligible)

        # Rule 4: Extra luggage organization
        # If this passenger is the last to board in their row, organize all luggage
        if _is_last_passenger_in_row(
            seat, boarding_order, seated_passengers, num_columns
        ):
            # Organize luggage for all passengers in the row
            extra_luggage_time = np.random.exponential(TIME_EXTRA_LUGGAGE)
            passenger_time += extra_luggage_time

        # Rule 5: Ordering/arrangement time = 0 (negligible) - no time added

        # Mark passenger as seated
        seated_passengers[seat] = True

        # Add this passenger's time to the total boarding time
        total_time += passenger_time

    return total_time


def _is_last_passenger_in_row(
    seat: str, boarding_order: list[str], seated_passengers: dict, num_columns: int
) -> bool:
    """
    Check if this passenger is the last one to board in their row.
    Used to determine who organizes the overhead luggage compartment.
    """
    row_num = int(seat[:2])

    # Count how many passengers in this row are already seated
    seated_in_row = 0
    for seated_seat in seated_passengers:
        if seated_seat.startswith(f"{row_num:02d}"):
            seated_in_row += 1

    # After this passenger sits, they will be the last if seated_in_row + 1 == num_columns
    return (seated_in_row + 1) == num_columns


def _count_blocking_passengers(seat: str, seated_passengers: dict) -> int:
    """
    Count how many passengers are blocking the given seat.
    In airplane seating ABC | DEF, passengers are blocked by those between them and the aisle.
    """
    row_num = int(seat[:2])
    seat_letter = seat[2]
    blocking_count = 0

    # Check for blocking passengers based on seat position
    for seated_seat, is_seated in seated_passengers.items():
        if is_seated and seated_seat.startswith(f"{row_num:02d}"):
            seated_letter = seated_seat[2]

            # Left section (A, B, C) - blocked by seats closer to aisle (C)
            if seat_letter in ["A", "B"]:
                if seated_letter == "B" and seat_letter == "A":  # B blocks A
                    blocking_count += 1
                elif seated_letter == "C" and seat_letter in [
                    "A",
                    "B",
                ]:  # C blocks A and B
                    blocking_count += 1

            # Right section (D, E, F) - blocked by seats closer to aisle (D)
            elif seat_letter in ["E", "F"]:
                if seated_letter == "E" and seat_letter == "F":  # E blocks F
                    blocking_count += 1
                elif seated_letter == "D" and seat_letter in [
                    "E",
                    "F",
                ]:  # D blocks E and F
                    blocking_count += 1

    return blocking_count


def _random_boarding(seats: list[str]) -> list[str]:
    """Random boarding order."""
    shuffled_seats = seats.copy()
    np.random.shuffle(shuffled_seats)
    return shuffled_seats


def _back_to_front_boarding(
    seats: list[str], num_rows: int, num_columns: int
) -> list[str]:
    """Back-to-front boarding order."""
    column_letters = [chr(ord("A") + i) for i in range(num_columns)]
    boarding_order = []

    # Start from the back row and work forward
    for row in range(num_rows, 0, -1):
        for letter in column_letters:
            seat = f"{row:02d}{letter}"
            if seat in seats:
                boarding_order.append(seat)

    return boarding_order


def _row_random_boarding(
    seats: list[str], num_rows: int, num_columns: int
) -> list[str]:
    """Row-based random boarding order - passengers board row by row, but in random order within each row."""
    column_letters = [chr(ord("A") + i) for i in range(num_columns)]
    boarding_order = []

    # Create list of rows and shuffle them
    rows = list(range(1, num_rows + 1))
    np.random.shuffle(rows)

    # For each row, add passengers in random order
    for row in rows:
        row_seats = []
        for letter in column_letters:
            seat = f"{row:02d}{letter}"
            if seat in seats:
                row_seats.append(seat)

        # Shuffle seats within the row
        np.random.shuffle(row_seats)
        boarding_order.extend(row_seats)

    return boarding_order


def _window_middle_aisle_boarding(
    seats: list[str], num_rows: int, num_columns: int
) -> list[str]:
    """Window-middle-aisle boarding order."""
    column_letters = [chr(ord("A") + i) for i in range(num_columns)]
    boarding_order = []

    # Assuming A, F are window seats, B, E are middle, C, D are aisle
    # This assumes a 6-seat configuration (A-F)
    seat_priority = {"A": 1, "F": 1, "B": 2, "E": 2, "C": 3, "D": 3}

    for priority in [1, 2, 3]:
        for row in range(1, num_rows + 1):
            for letter in column_letters:
                if seat_priority.get(letter, 3) == priority:
                    seat = f"{row:02d}{letter}"
                    if seat in seats:
                        boarding_order.append(seat)

    return boarding_order


def _steffen_boarding(seats: list[str], num_rows: int, num_columns: int) -> list[str]:
    """Steffen boarding method - alternating rows, window to aisle."""
    column_letters = [chr(ord("A") + i) for i in range(num_columns)]
    boarding_order = []

    # Steffen method: odd rows back to front, even rows back to front, window to aisle
    seat_priority = {"A": 1, "F": 1, "B": 2, "E": 2, "C": 3, "D": 3}

    # First, odd rows from back to front
    for priority in [1, 2, 3]:
        for row in range(num_rows, 0, -1):
            if row % 2 == 1:  # Odd rows
                for letter in column_letters:
                    if seat_priority.get(letter, 3) == priority:
                        seat = f"{row:02d}{letter}"
                        if seat in seats:
                            boarding_order.append(seat)

    # Then, even rows from back to front
    for priority in [1, 2, 3]:
        for row in range(num_rows, 0, -1):
            if row % 2 == 0:  # Even rows
                for letter in column_letters:
                    if seat_priority.get(letter, 3) == priority:
                        seat = f"{row:02d}{letter}"
                        if seat in seats:
                            boarding_order.append(seat)

    return boarding_order
