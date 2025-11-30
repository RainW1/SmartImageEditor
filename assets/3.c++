#include <iostream>
#include <string>
using namespace std;

void printBoard(char board[3][3]) {
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            cout << board[i][j];
            if (j < 2) cout << " | ";
        }
        cout << endl;
        if (i < 2) cout << "---------" << endl;
    }
}

bool checkWin(char board[3][3], char player) {
    // Check rows and columns
    for (int i = 0; i < 3; i++) {
        if ((board[i][0] == player && board[i][1] == player && board[i][2] == player) ||
            (board[0][i] == player && board[1][i] == player && board[2][i] == player)) {
            return true;
        }
    }
    // Check diagonals
    if ((board[0][0] == player && board[1][1] == player && board[2][2] == player) ||
        (board[0][2] == player && board[1][1] == player && board[2][0] == player)) {
        return true;
    }
    return false;
}

void ticTacToe() {
    cout << "=== Two-player Tic Tac Toe ===" << endl;
    char board[3][3] = {{' ', ' ', ' '}, {' ', ' ', ' '}, {' ', ' ', ' '}};
    char currentPlayer = 'X';
    int row, col;
    
    for (int turn = 0; turn < 9; turn++) {
        printBoard(board);
        while (true) {
            cout << "Player " << currentPlayer << ", enter row (1-3): ";
            while (!(cin >> row) || row < 1 || row > 3) {
                cin.clear();
                cin.ignore(numeric_limits<streamsize>::max(), '\n');
                cout << "Invalid input! Enter row (1-3): ";
            }
            cout << "Player " << currentPlayer << ", enter column (1-3): ";
            while (!(cin >> col) || col < 1 || col > 3) {
                cin.clear();
                cin.ignore(numeric_limits<streamsize>::max(), '\n');
                cout << "Invalid input! Enter column (1-3): ";
            }
            
            row--; col--;
            if (board[row][col] == ' ') break;
            else cout << "Position already taken! Try again." << endl;
        }
        
        board[row][col] = currentPlayer;
        
        if (checkWin(board, currentPlayer)) {
            printBoard(board);
            cout << "Player " << currentPlayer << " wins!" << endl;
            return;
        }
        
        currentPlayer = (currentPlayer == 'X') ? 'O' : 'X';
    }
    
    printBoard(board);
    cout << "It's a tie!" << endl;
}

int main() {
    ticTacToe();
    return 0;
}