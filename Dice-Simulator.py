import random

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


def roll_dice(count):
    return [random.randint(1, 6) for _ in range(count)]


def print_dice(values):
    for i in range(6):
        for v in values:
            print(DICE_ART[v][i], end="  ")
        print()


def main():
    roll_count = 0

    while True:
        print("\n1. Roll dice")
        print("2. Exit")
        choice = input("Choose option: ").strip()

        if choice == "2":
            print("Total rolls:", roll_count)
            break

        if choice != "1":
            print("Invalid choice.")
            continue

        dice_choice = input("Roll 1 dice or 2 dice? (1/2): ").strip()
        if dice_choice not in ("1", "2"):
            print("Enter only 1 or 2.")
            continue

        dice_count = int(dice_choice)
        dice = roll_dice(dice_count)
        roll_count += 1

        print(f"\nRoll #{roll_count}")
        print("Dice values:", *dice)
        print_dice(dice)
        print("Total:", sum(dice))


if __name__ == "__main__":
    main()
