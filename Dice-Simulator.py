import random
import sys

DICE_ART = {
    1: (
        "┌─────────┐",
        "│         │",
        "│    ●    │",
        "│         │",
        "│         │",
        "└─────────┘",
    ),
    2: (
        "┌─────────┐",
        "│ ●       │",
        "│         │",
        "│       ● │",
        "│         │",
        "└─────────┘",
    ),
    3: (
        "┌─────────┐",
        "│ ●       │",
        "│    ●    │",
        "│       ● │",
        "│         │",
        "└─────────┘",
    ),
    4: (
        "┌─────────┐",
        "│ ●     ● │",
        "│         │",
        "│ ●     ● │",
        "│         │",
        "└─────────┘",
    ),
    5: (
        "┌─────────┐",
        "│ ●     ● │",
        "│    ●    │",
        "│ ●     ● │",
        "│         │",
        "└─────────┘",
    ),
    6: (
        "┌─────────┐",
        "│ ●     ● │",
        "│ ●     ● │",
        "│ ●     ● │",
        "│         │",
        "└─────────┘",
    ),
}


def get_user_choice():
    """Ask user whether to roll or exit."""
    while True:
        choice = input("\nRoll the dice? (y/n): ").strip().lower()
        if choice in ("y", "n"):
            return choice
        print("Invalid input. Enter y or n only.")


def roll_dice(count=2):
    """Roll given number of dice."""
    return [random.randint(1, 6) for _ in range(count)]


def print_dice(dice_values):
    """Print dice art side by side."""
    dice_lines = [DICE_ART[value] for value in dice_values]

    for i in range(6):
        for dice in dice_lines:
            print(dice[i], end="  ")
        print()


def main():
    print("🎲 DICE ROLLER SIMULATOR 🎲")
    print("-" * 30)

    roll_count = 0

    while True:
        choice = get_user_choice()
        if choice == "n":
            print("\nThanks for playing.")
            print(f"Total rolls: {roll_count}")
            sys.exit()

        dice = roll_dice(2)
        roll_count += 1

        print(f"\nRoll #{roll_count}")
        print(f"Dice values: {dice[0]} and {dice[1]}")
        print_dice(dice)

        print(f"Total: {sum(dice)}")


if __name__ == "__main__":
    main()
