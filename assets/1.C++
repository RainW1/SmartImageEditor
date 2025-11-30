#include <iostream>
#include <cstdlib>
#include <ctime>
using namespace std;

void guessNumber() {
    cout << "=== Guess the Number Game ===" << endl;
    srand(time(0));
    int target = rand() % 100 + 1;
    int guess, attempts = 0;
    
    while (true) {
        cout << "Guess a number between 1-100: ";
        while (!(cin >> guess)) {
            cin.clear();
            cin.ignore(numeric_limits<streamsize>::max(), '\n');
            cout << "Invalid input! Please enter a number: ";
        }
        attempts++;
        
        if (guess < target) {
            cout << "Too small!" << endl;
        } else if (guess > target) {
            cout << "Too big!" << endl;
        } else {
            cout << "Congratulations! You guessed it in " << attempts << " tries." << endl;
            break;
        }
    }
}

int main() {
    guessNumber();
    return 0;
}