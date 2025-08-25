import numpy as np
from enum import Enum

# Time constants for boarding simulation, measured in seconds
# exponential distribution for luggage placement
TIME_LUGGAGE = 0.5
# exponential distribution for sitting when blocked by others (base time)
TIME_SITTING_BLOCKED = 0.5
# additional time per person blocking (0.25 * number of blocking people)
TIME_PER_BLOCKING_PERSON = 0.25

RAND = np.random.Generator(np.random.MT19937())


class BoardingMethod(Enum):
    RANDOM = "Random"
    BACK_TO_FRONT = "Back to Front"
    FRONT_TO_BACK = "Front to Back"
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
    elif method == BoardingMethod.BACK_TO_FRONT:
        boarding_order = _back_to_front_boarding(all_seats, num_rows, num_columns)
    elif method == BoardingMethod.FRONT_TO_BACK:
        boarding_order = _front_to_back_boarding(all_seats, num_rows, num_columns)
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
    Simulate boarding according to the specification rules
    """
    last_passenger = None
    last_time = 0.0

    seated_passengers = {}
    current_time = 0.0

    for i, seat in enumerate(boarding_order):
        current_row = int(seat[:2])
        passenger_time = 0.0

        # Rule 4: Seating completion time
        last_row = int(last_passenger[:2]) if last_passenger else 0
        if current_row >= last_row:
            passenger_time += last_time

        # Rule 2: Add luggage
        passenger_time += RAND.exponential(TIME_LUGGAGE)

        # Rule 3: Sit in seat - check for blocking within the row
        blocking_count = _count_blocking_passengers(seat, seated_passengers)
        if blocking_count > 0:
            mean_sitting_time = (
                TIME_SITTING_BLOCKED + blocking_count * TIME_PER_BLOCKING_PERSON
            )
            passenger_time += RAND.exponential(mean_sitting_time)

        # Mark this passenger as seated
        seated_passengers[seat] = True
        # Prevent stacking time
        last_time = max(passenger_time - last_time, 0)
        last_passenger = seat

        current_time += passenger_time

    return current_time


def _count_blocking_passengers(seat: str, seated_passengers: dict) -> int:
    """
    Count how many passengers are blocking the given seat.
    In airplane seating ABC | DEF, passengers are blocked by those between them and the aisle.
    """
    row_num = int(seat[:2])
    seat_letter = seat[2]
    blocking_count = 0

    seated_in_same_row = [
        s[2] for s in seated_passengers if s.startswith(f"{row_num:02d}")
    ]

    if seat_letter in "AB":
        # Aisle is on the right side, so check for B and C
        if "C" in seated_in_same_row:
            blocking_count += 1
        if "B" in seated_in_same_row:
            blocking_count += 1
    elif seat_letter in "EF":
        # Aisle is on the left side, so check for D and E
        if "D" in seated_in_same_row:
            blocking_count += 1
        if "E" in seated_in_same_row:
            blocking_count += 1

    return blocking_count


def _random_boarding(seats: list[str]) -> list[str]:
    """Random boarding order."""
    shuffled_seats = seats.copy()
    RAND.shuffle(shuffled_seats)
    return shuffled_seats


def _back_to_front_boarding(
    seats: list[str], num_rows: int, num_columns: int
) -> list[str]:
    """Back-to-front boarding order."""
    column_letters = [chr(ord("A") + i) for i in range(num_columns)]
    boarding_order = []

    # Start from the back row and work forward
    for row in range(num_rows, 0, -1):
        for letter in RAND.permutation(column_letters):
            seat = f"{row:02d}{letter}"
            if seat in seats:
                boarding_order.append(seat)

    return boarding_order


def _front_to_back_boarding(
    seats: list[str], num_rows: int, num_columns: int
) -> list[str]:
    """Front-to-back boarding order."""
    column_letters = [chr(ord("A") + i) for i in range(num_columns)]
    boarding_order = []

    # Start from the front row (row 1) and work toward the back
    for row in range(1, num_rows + 1):
        for letter in RAND.permutation(column_letters):
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
