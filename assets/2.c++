#include <iostream>
#include <cstdlib>
#include <ctime>
#include <string>
using namespace std;

void rockPaperScissors() {
    cout << "=== Rock Paper Scissors Game ===" << endl;
    srand(time(0));
    string choices[] = {"rock", "paper", "scissors"};
    string userChoice;
    
    while (true) {
        cout << "Enter (rock/paper/scissors) or 'q' to quit: ";
        cin >> userChoice;
        
        if (userChoice == "q") {
            cout << "Game over!" << endl;
            break;
        }
        
        bool valid = false;
        for (string c : choices) {
            if (userChoice == c) {
                valid = true;
                break;
            }
        }
        if (!valid) {
            cout << "Invalid input, please choose again!" << endl;
            continue;
        }
        
        int compIndex = rand() % 3;
        string compChoice = choices[compIndex];
        cout << "Computer chose: " << compChoice << endl;
        
        if (userChoice == compChoice) {
            cout << "It's a tie!" << endl;
        } else if ((userChoice == "rock" && compChoice == "scissors") ||
                   (userChoice == "scissors" && compChoice == "paper") ||
                   (userChoice == "paper" && compChoice == "rock")) {
            cout << "You win!" << endl;
        } else {
            cout << "You lose!" << endl;
        }
    }
}

int main() {
    rockPaperScissors();
    return 0;
}