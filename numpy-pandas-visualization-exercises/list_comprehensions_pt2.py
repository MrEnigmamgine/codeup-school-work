import math
import pandas as pd

# 17 list comprehension problems in python

# **Challenge yourself to be able to...**

# - solve each using vanilla python.
    
# - solve each using list comprehensions.
    
# - solve each by using a pandas Series for the data structure instead of lists and using vectorized operations instead of loops and list comprehensions.

fruits = ['mango', 'kiwi', 'strawberry', 'guava', 'pineapple', 'mandarin orange']

numbers = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 17, 19, 23, 256, -8, -4, -2, 5, -9]


pdFruits = pd.Series(fruits)
pdNumbers = pd.Series(numbers)

# Example for loop solution to add 1 to each number in the list
numbers_plus_one = []
for number in numbers:
    numbers_plus_one.append(number + 1)

# Example of using a list comprehension to create a list of the numbers plus one.
numbers_plus_one = [number + 1 for number in numbers]

# Example code that creates a list of all of the list of strings in fruits and uppercases every string
output = []
for fruit in fruits:
    output.append(fruit.upper())
    


# Exercise 1 - rewrite the above example code using list comprehension syntax. 
# Make a variable named uppercased_fruits to hold the output of the list comprehension. Output should be ['MANGO', 'KIWI', etc...]
uppercased_fruits = [fruit.upper() for fruit in fruits]
print(uppercased_fruits)

print( pdFruits.str.upper() )

# Exercise 2 - create a variable named capitalized_fruits and use list comprehension syntax to produce output like ['Mango', 'Kiwi', 'Strawberry', etc...]
capitalized_fruits = [fruit.capitalize() for fruit in fruits]
print(capitalized_fruits)

print( pdFruits.str.capitalize() )

###################
## Boolean Masks ##
###################

# Exercise 3 - Use a list comprehension to make a variable named fruits_with_more_than_two_vowels. Hint: You'll need a way to check if something is a vowel.
def countvowels(string):
    vowels = 0
    for char in string:
        if char in "aeiouAEIOU":
           vowels += 1
    return vowels

fruits_with_more_than_two_vowels = [fruit for fruit in fruits if countvowels(fruit) > 2]
print(fruits_with_more_than_two_vowels)

print(pdFruits[pdFruits.str.count(r'[aeiouAEIOU]') > 2])

# Exercise 4 - make a variable named fruits_with_only_two_vowels. The result should be ['mango', 'kiwi', 'strawberry']
fruits_with_only_two_vowels = [fruit for fruit in fruits if countvowels(fruit) == 2]
print(fruits_with_only_two_vowels)

print(pdFruits[pdFruits.str.count(r'[aeiouAEIOU]') == 2])


# Exercise 5 - make a list that contains each fruit with more than 5 characters
print([fruit for fruit in fruits if len(fruit) > 5])


print(pdFruits[pdFruits.str.len() > 5])

# Exercise 6 - make a list that contains each fruit with exactly 5 characters
print([fruit for fruit in fruits if len(fruit) == 5])

print(pdFruits[pdFruits.str.len() == 5])

# Exercise 7 - Make a list that contains fruits that have less than 5 characters
print([fruit for fruit in fruits if len(fruit) < 5])

print(pdFruits[pdFruits.str.len() < 5])

# Exercise 8 - Make a list containing the number of characters in each fruit. Output would be [5, 4, 10, etc... ]
print([len(fruit) for fruit in fruits])

print(pdFruits.str.len())

# Exercise 9 - Make a variable named fruits_with_letter_a that contains a list of only the fruits that contain the letter "a"
fruits_with_letter_a = [fruit for fruit in fruits if 'a' in list(fruit)]
print(fruits_with_letter_a)

print( pdFruits[ pdFruits.str.contains('a') ] )

# Exercise 10 - Make a variable named even_numbers that holds only the even numbers 
even_numbers = [n for n in numbers if n % 2 == 0]
print(even_numbers)

# Exercise 11 - Make a variable named odd_numbers that holds only the odd numbers
odd_numbers = [n for n in numbers if n % 2 == 1]
print(odd_numbers)

# Exercise 12 - Make a variable named positive_numbers that holds only the positive numbers
positive_numbers = [n for n in numbers if n >= 0]
print(positive_numbers)

# Exercise 13 - Make a variable named negative_numbers that holds only the negative numbers
negative_numbers = [n for n in numbers if n <= 0]
print(negative_numbers)

####################
## /Boolean Masks ##
####################


# Exercise 14 - use a list comprehension w/ a conditional in order to produce a list of numbers with 2 or more numerals

## I want to test if negative two digit numbers works.
numbers.append(-22)
print([n for n in numbers if len(str(abs(n))) > 1])

pdNumbers.append(pd.Series([-22]))  ## Still want to test this with pandas method
pdNumbers                           ## Oh.. rude.. append doesn't actually save. Are these things immutable?
                                    ## Google says no they *are* mutable. But you can't change their length. 
pdNumbers = pd.Series(numbers)      ## That's a technicality IMO, anyway we'll just overwrite.
## This is still a boolean mask, but it involved converting the datatype of the series which made it interesting.
print(pdNumbers[pdNumbers.abs().astype(str).str.len() > 1])
## Let's break it down.
step1 = pdNumbers.abs()     ## Remove the sign. Doesn't mutate the data so when we eventually mask it on the original we'll still get the negatives.
step2 = step1.astype(str)   ## Covert the series data to a string.
step3 = step2.str.len()     ## Use the string accessor to get the number of digits
finalStep = step3 > 1       ## Finally convert the whole thing into a boolean mask.
pdNumbers[finalStep]        ## Profit


# Exercise 15 - Make a variable named numbers_squared that contains the numbers list with each element squared. Output is [4, 9, 16, etc...]
numbers_squared = [n**2 for n in numbers]
print(numbers_squared)

## Vectorized operation
print(pdNumbers**2)

# Exercise 16 - Make a variable named odd_negative_numbers that contains only the numbers that are both odd and negative.
odd_negative_numbers = [n for n in odd_numbers if n < 0]
print(odd_negative_numbers)

## Another boolean mask. But this question asks you to remember the bitwize operators.
pdNumbers[(pdNumbers < 0) & (pdNumbers % 2 == 1)]

# Exercise 17 - Make a variable named numbers_plus_5. In it, return a list containing each number plus five. 
numbers_plus_5 = [n+5 for n in numbers]
print(numbers_plus_5)

# BONUS Make a variable named "primes" that is a list containing the prime numbers in the numbers list. 
# *Hint* you may want to make or find a helper function that determines if a given number is prime or not.
def isprime(x):
    if x < 1: return False
    a=2
    while a <= math.sqrt(x):
        if x%a<1:
            return False
        a=a+1
    return x>1

primes = [n for n in numbers if isprime(n)]
print(primes)

# Another boolean mask.  But this one asks you to apply a function.
pdNumbers[pdNumbers.apply(isprime)]