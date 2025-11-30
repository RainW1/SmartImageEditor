#include <iostream>
#include <cstdlib>
#include <ctime>
using namespace std;

void numberBomb() {
    cout << "=== Number Bomb Game ===" << endl;
    srand(time(0));
    int minNum = 1, maxNum = 100;
    int bomb = rand() % 100 + 1;
    int guess;
    
    while (true) {
        cout << "\nCurrent range: " << minNum << " - " << maxNum << endl;
        cout << "Guess a number: ";
        
        while (!(cin >> guess)) {
            cin.clear();
            cin.ignore(numeric_limits<streamsize>::max(), '\n');
            cout << "Invalid input! Please enter a number: ";
        }
        
        if (guess < minNum || guess > maxNum) {
            cout << "Please guess between " << minNum << "-" << maxNum << "!" << endl;
            continue;
        }
        
        if (guess == bomb) {
            cout << "Boom! You hit the bomb (the bomb was " << bomb << ")!" << endl;
            break;
        }
        
        if (guess < bomb) {
            cout << "Too small, the bomb is in a larger range!" << endl;
            minNum = guess + 1;
        } else {
            cout << "Too big, the bomb is in a smaller range!" << endl;
            maxNum = guess - 1;
        }
        
        if (minNum >= maxNum) {
            cout << "You win! The bomb was " << bomb << endl;
            break;
        }
    }
}

int main() {
    numberBomb();
    return 0;
}