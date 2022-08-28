"""
Problem 1
Assume s is a string of lower case characters.
Write a program that counts up the numbers of vowels
contained in the string s. Valid vowels are:
'a', 'e', 'i', 'o', 'u'.

For example, if s = 'azcbobobegghakl', your program should
print:

Number of vowels: 5
"""
s = 'fjdakluineauogoe' # This changes as test input
vowels = ["A", "E", "I", "O", "U"]
vowel_count = 0
s_upper = s.upper()
for char in s_upper:
    if char in vowels:
        vowel_count +=1
    
print('Number of vowels: {0}'.format(vowel_count))


"""
Problem 2
Assume s is a string of lower case characters.

Write a program that prints the number of times the string 'bob' occurs in s. 
For example, if s = 'azcbobobegghakl', then your program should print:
Number of times bob occurs is: 2
"""

pattern = 'bob'

