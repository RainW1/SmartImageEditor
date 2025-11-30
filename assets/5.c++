#include <iostream>
#include <cstdlib>
#include <ctime>
#include <string>
using namespace std;

void rollDice() {
    cout << "=== Dice Rolling Game ===" << endl;
    srand(time(0));
    char playAgain;
    
    do {
        cout << "Press Enter to roll the dice...";
        cin.ignore(); // Wait for Enter
        
        int playerDice = rand() % 6 + 1;
        int compDice = rand() % 6 + 1;
        
        cout << "Your dice: " << playerDice << endl;
        cout << "Computer's dice: " << compDice << endl;
        
        if (playerDice > compDice) {
            cout << "You win!" << endl;
        } else if (playerDice < compDice) {
            cout << "You lose!" << endl;
        } else {
            cout << "It's a tie!" << endl;
        }
        
        cout << "Play again? (y/n): ";
        cin >> playAgain;
    } while (playAgain == 'y' || playAgain == 'Y');
    
    cout << "Game over!" << endl;
}

int main() {
    rollDice();
    return 0;
}