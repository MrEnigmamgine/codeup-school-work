################################ Part 1 #####################################
#  You have rented some movies for your kids: 
#   -  The little mermaid (for 3 days), 
#   -  Brother Bear (for 5 days, they love it), 
#   -  and Hercules (1 day, you don't know yet if they're going to like it). 
#  If price for a movie per day is 3 dollars, how much will you have to pay?


# Since the scope of this problem is limited to 3 movies and I don't expect any movies to be added to the list, and I don't care about the title or the audiences approval
# AND since the price per day is the same for all movies, I can store the relevant data in a list consisting of the days in integers.
rentalDays = [3, 5, 1]

# We can now easily calculate the total cost of rentals
rentalCost = sum(rentalDays) * 3
print(rentalCost)

################################ Part 2 #####################################
# Suppose you're working as a contractor for 3 companies: Google, Amazon and Facebook, they pay you a different rate per hour. 
#       Google pays 400 dollars per hour, 
#       Amazon 380, and 
#       Facebook 350. 
# How much will you receive in payment for this week? 
# You worked 10 hours for Facebook, 6 hours for Google and 4 hours for Amazon.

# Since we have multiple dimensions of data that we care about dictionaries make the most sense to me.
# I'll be using a dictionary with subdictionaries.

ledger = {
        'Google' : {
            'rate' : 400,
            'hours' : 6
        },
        'Amazon' : {
            'rate' : 380,
            'hours' : 4
        },
        'Facebook' : {
            'rate' : 350,
            'hours' : 10
        },
}

# Calculating it would look something like this
totalCheck = 0
for patron, timeSheet in ledger.items():
    check = timeSheet['rate'] * timeSheet['hours']
    print(f'{patron}: ${check}.00')
    totalCheck += check
print(f'Total: ${totalCheck}.00')

################################ Part 3 #####################################
# A student can be enrolled to a class only if the class is not full and the class schedule does not conflict with her current schedule.

isClassFull = False
isScheduleConflicting = False
canEnroll = not isClassFull and not isScheduleConflicting

################################ Part 4 #####################################
# A product offer can be applied only if people buys more than 2 items, and the offer has not expired. Premium members do not need to buy a specific amount of products.

cartItemCount = 3
isPremiumMember = False
isOfferExpired = False
isCartEligible = cartItemCount > 2

canOfferApply = not isOfferExpired and (isCartEligible or isPremiumMember)

################################ Part 4 #####################################
# GIVEN
username = 'codeup'
password = 'notastrongpassword'
# Create a variable that holds a boolean value for each of the following conditions:
# - the password must be at least 5 characters
# - the username must be no more than 20 characters
# - the password must not be the same as the username
# - bonus neither the username or password can start or end with whitespace

pass_len_valid = len(password) >= 5
name_len_valid = len(username) <= 20
name_pass_not_match = username.lower().strip() != password.lower().strip()  # https://pypi.org/project/jellyfish/0.5.4/ for a more robust solution
name_not_whitespace = not username[0].isspace() and not username[-1].isspace()
pass_not_whitespace = not password[0].isspace() and not password[-1].isspace()
name_pass_valid = pass_len_valid and name_len_valid and name_pass_not_match and name_not_whitespace and pass_not_whitespace
print(pass_len_valid, name_len_valid, name_pass_not_match, name_not_whitespace, pass_not_whitespace)
print(name_pass_valid)