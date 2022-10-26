# Exercises
# Create a file named function_exercises.py for this exercise. After creating each function specified below, write the necessary code in order to test your function.

# Define a function named is_two. It should accept one input and return True if the passed input is either the number or the string 2, False otherwise.

## Checks if a value can be evaluate as the integer 2.
# @param {string OR int} x
##



def is_two(x):
    # Trying to convert "two" into an integer will cause an error to be thrown.
    # To work around that we can use try: / except:
    try:      
        # This code will TRY to run
        int(x)
        return int(x) == 2
    except:
        # If an error is thrown, this will run instead.
        return False

# print(is_two(2), is_two('2'))

# Define a function named is_vowel. It should return True if the passed string is a vowel, False otherwise.

## Checks if a string contains a single vowel.
# @param {string} s
##
def is_vowel(s):
  vowels = ['a','e','i','o','u']
  # You can ask python if something is in an array, and we can pass that directly to the return statement.
  return s.lower() in vowels

assert is_vowel("a") == True
assert is_vowel("e") == True
assert is_vowel("i") == True
assert is_vowel("o") == True
assert is_vowel("u") == True
assert is_vowel("A") == True
assert is_vowel("E") == True
assert is_vowel("I") == True
assert is_vowel("O") == True
assert is_vowel("U") == True
assert is_vowel("banana") == False
assert is_vowel("Q") == False
assert is_vowel("y") == False
assert is_vowel("aei") == False
assert is_vowel("iou") == False

# Define a function named is_consonant. It should return True if the passed string is a consonant, False otherwise. Use your is_vowel function to accomplish this.

## Checks if a string contains a single consonant.
# @param {string} s
##
def is_consonant(s):
    # We aren't asking if something is in an array this time because of an arbitrary requirement,
    # so instead we will want to first make sure that we are receiving only a single character,
    if len(s) != 1 or s.isdigit(): return False        # and stopping the function if not.
    # While not completely accurate due to characters like [$,%,#,@], we can surmise that if something is
    # not a vowel, then it is probably a consonant.
    return not is_vowel(s)

print(is_consonant("banana"))

assert is_consonant("O") == False
assert is_consonant("U") == False
assert is_consonant("banana") == False
assert is_consonant("Q") == True
assert is_consonant("y") == True

# Define a function that accepts a string that is a word. The function should capitalize the first letter of the word if the word starts with a consonant.

## Checks if a string starts with a consonant and is not just a single character. If so capitalize it.
# @param {string} word
##
def capitalize_word(word):
    if len(word) > 1 and is_consonant(word[0]):
        return word.lower().capitalize()
    return word

assert capitalize_word('apple') == 'apple'
assert capitalize_word('banana') == 'Banana'


# Define a function named calculate_tip. It should accept a tip percentage (a number between 0 and 1) and the bill total, and return the amount to tip.

## Calculates the tip amount based on the percentage of the total bill.
# @param {float} perc
# @param {float} bill
##
def calculate_tip(perc, bill):
    return round((perc * bill), 2)

print(calculate_tip(.15, 23.88))

# Define a function named apply_discount. It should accept a original price, and a discount percentage, and return the price after the discount is applied.

## Calculates resulting price after applying a discount presented in a percentage.
# @param {float} price
# @param {float} perc
##
def apply_discout(price, perc):
    return round((price - (price * perc)), 2)

print(apply_discout(23.88, .10))


# Define a function named handle_commas. It should accept a string that is a number that contains commas in it as input, and return a number as output.


def handle_commas(str):
    # The replace method can remove unwanted characters nicely.
    # Also we want to return a number. I chose to use float over int because I didn't want to assume that a number would be a whole number.
    return float(str.replace(',' , ''))

print(handle_commas('1,000,000.23'))

# Define a function named get_letter_grade. It should accept a number and return the letter grade associated with that number (A-F).

# Here we can leverage dictionaries and ranges to presnt the prerequisite data in an easy to understand format.
def get_letter_grade(num):
    gradeKey = { # Be aware that the range will not include the second number in the argument. But it does include the first.
                'A+': range(100,200),
                'A' : range(89 ,100),
                'A-': range(88 , 89),
                'B+': range(87 , 88),
                'B' : range(81 , 87),
                'B-': range(80 , 81),
                'C+': range(79 , 80),
                'C' : range(68 , 79),
                'C-': range(67 , 68),
                'D+': range(66 , 67),
                'D' : range(61 , 66),
                'D-': range(60 , 61),
                'F' : range( 0 , 60)
                }
    for grade, v in gradeKey.items():
        if num in v:
            return grade

grade = 89
get_letter_grade(grade)

# Define a function named remove_vowels that accepts a string and returns a string with all the vowels removed.
def remove_vowels(str):
    # Trying to remove something from an iterable while we are iterating through it is usually a bad idea.
    # Instead we'll create a new string from scratch based on the original, but we'll ignore any vowels.
    output = ''
    for l in str:
        if not is_vowel(l):
            output += l
    return(output)

remove_vowels('Define a function named remove_vowels that accepts a string and returns a string with all the vowels removed.')

# Define a function named normalize_name. It should accept a string and return a valid python identifier, that is:
    # anything that is not a valid python identifier should be removed
    # leading and trailing whitespace should be removed
    # everything should be lowercase
    # spaces should be replaced with underscores
    # for example:
    # Name will become name
    # First Name will become first_name
    # % Completed will become completed

# def normalize_name(str):
#     # This problem is just string manipulation with extra steps.  We can break down the steps by calling the methods in sequence.
#     return str.lower().replace('%','').strip().replace(' ', '_')

def remove_special_char(string):
    return ''.join([c for c in string if c.isalnum() or c == ' '])


def normalize_name(string):
    special_char_removed = remove_special_char(string)
    return special_char_removed.lower().strip().replace(' ', '_')

normalize_name('Name')
normalize_name('First Name')
normalize_name('% Completed')


# Write a function named cumulative_sum that accepts a list of numbers and returns a list that is the cumulative sum of the numbers in the list.
    # cumulative_sum([1, 1, 1]) returns [1, 2, 3]
    # cumulative_sum([1, 2, 3, 4]) returns [1, 3, 6, 10]

def cumulative_sum(arr):
    output = []
    for n in arr:
        # Without declaring a holder variable to track the sum we need to use a slightly different logic on the first loop.
        # To do so requires an if statement, which means that this isn't the most effecient loop since there is an extra step that runs every loop.
        if len(output) > 0:
            output.append(n + output[-1])
        else: output.append(n)
    return output

print(cumulative_sum([1, 1, 1]))
cumulative_sum([1, 2, 3, 4])
cumulative_sum(range(1,40))


# Additional Exercise
# Once you've completed the above exercises, follow the directions from https://gist.github.com/zgulde/ec8ed80ad8216905cda83d5645c60886 in order to thouroughly comment your code to explain your code.
# 
# Bonus
# Create a function named twelveto24. It should accept a string in the format 10:45am or 4:30pm and return a string that is the representation of the time in a 24-hour format. Bonus write a function that does the opposite.
# Create a function named col_index. It should accept a spreadsheet column name, and return the index number of the column.
# col_index('A') returns 1
# col_index('B') returns 2
# col_index('AA') returns 27