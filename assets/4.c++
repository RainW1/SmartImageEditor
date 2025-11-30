#include <iostream>
#include <cstdlib>
#include <ctime>
#include <vector>
#include <string>
#include <set>
using namespace std;

void guessWord() {
    cout << "=== Word Guessing Game ===" << endl;
    srand(time(0));
    vector<string> words = {"python", "apple", "banana", "computer", "sunshine", "programming"};
    string targetWord = words[rand() % words.size()];
    set<char> guessedLetters;
    int attempts = 6;
    
    while (attempts > 0) {
        // Show current progress
        cout << "Current word: ";
        bool complete = true;
        for (char c : targetWord) {
            if (guessedLetters.count(c)) {
                cout << c << " ";
            } else {
                cout << "_ ";
                complete = false;
            }
        }
        cout << endl;
        
        if (complete) {
            cout << "Congratulations! You guessed the word!" << endl;
            return;
        }
        
        char guess;
        cout << "Guess a letter: ";
        cin >> guess;
        guess = tolower(guess);
        
        if (!isalpha(guess)) {
            cout << "Please enter a letter!" << endl;
            continue;
        }
        
        if (guessedLetters.count(guess)) {
            cout << "You already guessed that letter!" << endl;
            continue;
        }
        
        guessedLetters.insert(guess);
        
        if (targetWord.find(guess) == string::npos) {
            attempts--;
            cout << "Wrong guess! Remaining attempts: " << attempts << endl;
        }
    }
    
    cout << "Game over! The word was: " << targetWord << endl;
}

int main() {
    guessWord();
    return 0;
}