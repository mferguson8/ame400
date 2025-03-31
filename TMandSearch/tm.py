#!/usr/bin/env python3
"""
tm.py

This file contains implementations of two Turing Machines:
1. Unary Addition – adds two numbers written in unary (e.g. 3+2 is represented as "_111+11")
2. Unary Subtraction – subtracts one unary number from another (always subtracting the smaller from the larger)

Each machine is implemented as an instance of the TuringMachine class.
"""

class TuringMachine:
    def __init__(self, tape, blank_symbol, initial_state, final_states, transitions):
        # Initialize tape (as a list), head position, state, etc.
        self.tape = list(tape)
        self.blank = blank_symbol
        self.state = initial_state
        self.final_states = final_states
        self.transitions = transitions
        self.head = 0

    def step(self):
        # Ensure the tape is long enough in case of head movement out of current bounds.
        if self.head < 0:
            self.tape.insert(0, self.blank)
            self.head = 0
        elif self.head >= len(self.tape):
            self.tape.append(self.blank)
        current_symbol = self.tape[self.head]
        key = (self.state, current_symbol)
        if key in self.transitions:
            new_state, new_symbol, direction = self.transitions[key]
            self.tape[self.head] = new_symbol
            self.state = new_state
            if direction == 'R':
                self.head += 1
            elif direction == 'L':
                self.head -= 1
            # 'N' means no movement.
            return True
        else:
            return False  # No applicable transition; halt.

    def run(self, max_steps=1000):
        steps = 0
        while self.state not in self.final_states and steps < max_steps:
            if not self.step():
                break
            steps += 1
        # Return the tape as a string with leading/trailing blanks removed.
        return "".join(self.tape).strip(self.blank)

# ---------------------------------------
# Turing Machine for Addition (Unary Addition)
#
# The idea:
# 1. Scan right (state q0) until the '+' symbol is found.
# 2. When '+' is found, change it to a marker 'P' and switch to state q1.
# 3. In q1, find a '1' in the second number, mark it as 'X', then go left (state q2).
# 4. In q2, move left to the left end (blank), then move right, write a '1' (thus appending
#    to the first number), and transition to q3.
# 5. In q3, move right until the marked 'X' is reached; convert it back to '1' and return to q1.
# 6. When no more '1's are found in the second number, the machine halts.

transitions_add = {
    ('q0', '1'): ('q0', '1', 'R'),
    ('q0', '+'): ('q1', 'P', 'R'),  # mark '+' as 'P'
    
    ('q1', '1'): ('q2', 'X', 'L'),
    ('q1', '_'): ('q_accept', '_', 'N'),  # finished processing
    
    ('q2', '1'): ('q2', '1', 'L'),
    ('q2', 'P'): ('q2', 'P', 'L'),
    ('q2', '_'): ('q3', '1', 'R'),  # write a '1' to the left number
    
    ('q3', '1'): ('q3', '1', 'R'),
    ('q3', 'P'): ('q3', 'P', 'R'),
    ('q3', 'X'): ('q1', '1', 'R'),  # restore 'X' to '1' and return to process the next digit
}

# ---------------------------------------
# Turing Machine for Subtraction (Unary Subtraction)
#
# This machine subtracts the second number from the first.
# The input is of the form: _[minuend]-[subtrahend]
# The machine works by “canceling” one '1' from the minuend for each '1'
# in the subtrahend.
#
# The process:
# 1. In state q0, scan right until '-' is found; then mark it with 'P' and switch to q1.
# 2. In q1, look for a '1' in the subtrahend. When found, mark it as 'X' and move left (q2).
# 3. In q2, move left until you find a '1' in the minuend and cancel it by replacing it with a blank.
# 4. In q3, move right until the marked 'X' is reached; convert it back to '1' and return to q1.
# 5. When no more '1's exist in the subtrahend, the machine halts.


transitions_sub = {
    ('q0', '1'): ('q0', '1', 'R'),
    ('q0', '-'): ('q1', 'P', 'R'),  # mark '-' as 'P'
    
    ('q1', '1'): ('q2', 'X', 'L'),
    ('q1', '_'): ('q_accept', '_', 'N'),  # finished processing
    
    ('q2', '1'): ('q3', '_' , 'R'),  # cancel one '1' from the minuend (replace with blank)
    ('q2', 'P'): ('q2', 'P', 'L'),
    ('q2', 'X'): ('q2', 'X', 'L'),
    ('q2', '_'): ('q_error', '_', 'N'),  # error state (should not occur if minuend >= subtrahend)
    
    ('q3', '1'): ('q3', '1', 'R'),
    ('q3', 'P'): ('q3', 'P', 'R'),
    ('q3', 'X'): ('q1', '1', 'R'),  # restore 'X' to '1' and continue
}

if __name__ == '__main__':
    # -----------------------
    # Test the Addition Turing Machine:
    print("Turing Machine for Addition (Unary Addition)")
    # Example: 3+2 is represented as _111+11
    tm_add1 = TuringMachine(tape="_111+11", blank_symbol='_', initial_state='q0',
                            final_states={'q_accept'}, transitions=transitions_add)
    result1 = tm_add1.run()
    print("Input: _111+11  -->  Output:", result1)  # Expected: 11111

    # Another example: 4+1 is represented as _1111+1
    tm_add2 = TuringMachine(tape="_1111+1", blank_symbol='_', initial_state='q0',
                            final_states={'q_accept'}, transitions=transitions_add)
    result2 = tm_add2.run()
    print("Input: _1111+1  -->  Output:", result2)  # Expected: 11111

    # -----------------------
    # Test the Subtraction Turing Machine:
    print("\nTuring Machine for Subtraction (Unary Subtraction)")
    # Example: 3-2 is represented as _111-11.
    tm_sub1 = TuringMachine(tape="_111-11", blank_symbol='_', initial_state='q0',
                            final_states={'q_accept'}, transitions=transitions_sub)
    result_sub1 = tm_sub1.run()
    print("Input: _111-11  -->  Output:", result_sub1)  # Expected result as per design (remaining ones)

    # Example: 3-1 is represented as _111-1.
    tm_sub2 = TuringMachine(tape="_111-1", blank_symbol='_', initial_state='q0',
                            final_states={'q_accept'}, transitions=transitions_sub)
    result_sub2 = tm_sub2.run()
    print("Input: _111-1  -->  Output:", result_sub2)  # Expected result as per design
