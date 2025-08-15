# No installation needed
import math

# Display the board
def print_board(board):
    for row in board:
        print(" | ".join(row))
        print("-" * 5)

# Check for winner or tie
def check_winner(board):
    # Rows
    for row in board:
        if row[0] == row[1] == row[2] != " ":
            return row[0]
    # Columns
    for col in range(3):
        if board[0][col] == board[1][col] == board[2][col] != " ":
            return board[0][col]
    # Diagonals
    if board[0][0] == board[1][1] == board[2][2] != " ":
        return board[0][0]
    if board[0][2] == board[1][1] == board[2][0] != " ":
        return board[0][2]
    # Tie
    for row in board:
        if " " in row:
            return None
    return "Tie"

# Minimax algorithm
def minimax(board, depth, is_maximizing):
    result = check_winner(board)
    if result == "O":
        return 1
    elif result == "X":
        return -1
    elif result == "Tie":
        return 0

    if is_maximizing:
        best_score = -math.inf
        for i in range(3):
            for j in range(3):
                if board[i][j] == " ":
                    board[i][j] = "O"
                    score = minimax(board, depth + 1, False)
                    board[i][j] = " "
                    best_score = max(score, best_score)
        return best_score
    else:
        best_score = math.inf
        for i in range(3):
            for j in range(3):
                if board[i][j] == " ":
                    board[i][j] = "X"
                    score = minimax(board, depth + 1, True)
                    board[i][j] = " "
                    best_score = min(score, best_score)
        return best_score

# AI move
def ai_move(board):
    best_score = -math.inf
    move = None
    for i in range(3):
        for j in range(3):
            if board[i][j] == " ":
                board[i][j] = "O"
                score = minimax(board, 0, False)
                board[i][j] = " "
                if score > best_score:
                    best_score = score
                    move = (i, j)
    board[move[0]][move[1]] = "O"

# Main game loop
if __name__ == "__main__":
    board = [[" " for _ in range(3)] for _ in range(3)]
    print("You are X, AI is O")

    while True:
        print_board(board)
        if check_winner(board):
            result = check_winner(board)
            if result == "Tie":
                print("It's a Tie!")
            else:
                print(f"{result} wins!")
            break

        # Human move
        try:
            row = int(input("Enter row (0-2): "))
            col = int(input("Enter col (0-2): "))
        except ValueError:
            print("Invalid input! Please enter numbers only.")
            continue

        if 0 <= row <= 2 and 0 <= col <= 2 and board[row][col] == " ":
            board[row][col] = "X"
        else:
            print("Invalid move, try again.")
            continue

        if check_winner(board):
            print_board(board)
            result = check_winner(board)
            if result == "Tie":
                print("It's a Tie!")
            else:
                print(f"{result} wins!")
            break

        # AI move
        ai_move(board)
